"""Prompt 拼装可视化工具。

传入任意 BenchmarkKit，查看框架如何将 Kit prompt 拼装成 VLM 消息。
默认用通用占位符（Observation 1, Action 1），与具体 benchmark 无关。

用法:
    from anything2skill.tests.mock_prompt_viewer import view_prompts
    from anything2skill.benchmarks.osworld.kit import OSWorldKit

    view_prompts(OSWorldKit())                              # 全部场景
    view_prompts(OSWorldKit(), scenarios=["planner"])        # 指定场景
    view_prompts(OSWorldKit(), obs={"screenshot": bytes})   # 传入真实观测
    view_prompts(OSWorldKit(), export=True)                 # 导出 JSON
"""

from __future__ import annotations

import json
from pathlib import Path

from anything2skill.agent.message_builder import MessageBuilder
from anything2skill.agent.planner import PlanAction, PlanDecision
from anything2skill.agent.state import AgentState
from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.parser.data_types import Skill, Skills


# ── 通用 fixtures（与 benchmark 无关）────────────────────────────────────

_DEFAULT_SKILLS = Skills(
    task_id="demo-task",
    instruction="Complete the demo task",
    skills=[
        Skill(name="skill-a", description="First skill", content="## Steps\n1. Do step A1\n2. Do step A2", images=[]),
        Skill(name="skill-b", description="Second skill", content="## Steps\n1. Do step B1\n2. Do step B2", images=[]),
    ],
)

_PLACEHOLDER_OBS = [{"type": "text", "text": "<Observation: current environment state>"}]

_PLACEHOLDER_HISTORY = [
    {
        "obs_content": [{"type": "text", "text": "<Observation 1>"}],
        "response": "<LLM Response 1>\nAction 1 code here",
        "action": "Action 1",
    },
    {
        "obs_content": [{"type": "text", "text": "<Observation 2>"}],
        "response": "<LLM Response 2>\nAction 2 code here",
        "action": "Action 2",
    },
]


# ── 格式化 ───────────────────────────────────────────────────────────────

def _fmt(messages: list[dict], title: str) -> str:
    lines = ["", "=" * 72, f"  {title}", "=" * 72]
    for i, msg in enumerate(messages):
        lines.append(f"\n--- Message {i} [{msg['role']}] ---")
        for block in msg.get("content", []):
            if block.get("type") == "text":
                lines.append(block["text"])
            elif block.get("type") == "image_url":
                url = block["image_url"]["url"]
                if len(url) > 80:
                    url = url[:40] + "..." + url[-20:]
                lines.append(f"[IMAGE] {url}")
    lines.append("\n" + "-" * 72)
    return "\n".join(lines)


# ── 核心 API ─────────────────────────────────────────────────────────────

def view_prompts(
    kit: BenchmarkKit,
    *,
    obs: dict | None = None,
    skills: Skills | None = None,
    scenarios: list[str] | None = None,
    export: bool = False,
) -> None:
    """传入 Kit，打印各场景下 VLM 收到的完整消息。

    Args:
        kit: 任意 BenchmarkKit 实例。
        obs: 真实观测 dict（会通过 kit.encode_observation 编码）。
             默认使用通用文本占位符，不调用 kit.encode_observation。
        skills: 自定义 Skills，默认用内置示例。
        scenarios: 要展示的场景列表，默认全部。
            可选: simple, vanilla, executor, reflect, planner, extraction
        export: 是否导出 JSON 到 prompt_snapshots/。
    """
    mb = MessageBuilder(kit)
    skills = skills or _DEFAULT_SKILLS

    # obs_content: 真实编码 or 通用占位符
    if obs is not None:
        obs_content = kit.encode_observation(obs)
        history = [
            {"obs_content": kit.encode_observation(obs), "response": "<LLM Response 1>", "action": "Action 1"},
            {"obs_content": kit.encode_observation(obs), "response": "<LLM Response 2>", "action": "Action 2"},
        ]
    else:
        obs_content = _PLACEHOLDER_OBS
        history = _PLACEHOLDER_HISTORY

    builders = {
        "simple":     lambda: _build_simple(mb, skills, obs_content, history),
        "vanilla":    lambda: _build_vanilla(mb, skills, obs_content, history),
        "executor":   lambda: _build_executor(mb, obs_content),
        "reflect":    lambda: _build_reflect(mb, skills, obs_content, history),
        "planner":    lambda: _build_planner(mb, skills, obs_content, history),
        "extraction": lambda: _build_extraction(mb, skills),
    }

    chosen = scenarios or list(builders.keys())
    all_results = {}
    for name in chosen:
        if name not in builders:
            print(f"Unknown scenario: {name} (available: {list(builders.keys())})")
            continue
        all_results.update(builders[name]())

    if export:
        out_dir = Path(__file__).parent / "prompt_snapshots"
        out_dir.mkdir(exist_ok=True)
        for name, msgs in all_results.items():
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(msgs, f, indent=2, ensure_ascii=False)
        print(f"\nJSON exported to: {out_dir}")


# ── 场景构建 ─────────────────────────────────────────────────────────────

def _build_simple(mb, skills, obs_content, history):
    msgs1 = mb.build_action_messages(skills, obs_content, history=[], instruction=skills.instruction)
    print(_fmt(msgs1, "SimpleAgent (无历史)"))
    msgs2 = mb.build_action_messages(skills, obs_content, history=history, instruction=skills.instruction)
    print(_fmt(msgs2, "SimpleAgent (2 轮历史)"))
    return {"simple_no_history": msgs1, "simple_with_history": msgs2}


def _build_vanilla(mb, skills, obs_content, history):
    msgs1 = mb.build_vanilla_messages(obs_content, history=[], instruction=skills.instruction)
    print(_fmt(msgs1, "VanillaAgent (无历史)"))
    msgs2 = mb.build_vanilla_messages(obs_content, history=history, instruction=skills.instruction)
    print(_fmt(msgs2, "VanillaAgent (2 轮历史)"))
    return {"vanilla_no_history": msgs1, "vanilla_with_history": msgs2}


def _build_executor(mb, obs_content):
    d = PlanDecision(PlanAction.EXECUTE, reasoning="State matches step 1.", guidance="Execute step B1.")
    msgs = mb.build_executor_messages(obs_content, "Complete the demo task", d)
    print(_fmt(msgs, "Executor (EXECUTE)"))
    return {"executor": msgs}


def _build_reflect(mb, skills, obs_content, history):
    d = PlanDecision(PlanAction.REFLECT, reasoning="Previous action failed.", guidance="Re-examine SOP.")
    msgs = mb.build_reflect_messages(skills, obs_content, history=history, instruction=skills.instruction, decision=d)
    print(_fmt(msgs, "Reflector (REFLECT)"))
    return {"reflect": msgs}


def _build_planner(mb, skills, obs_content, history):
    state = AgentState()
    for entry in history:
        state.record(entry["obs_content"], entry["response"], entry["action"])
    msgs = mb.build_planner_messages(skills, state, obs_content, skills.instruction)
    print(_fmt(msgs, "Planner"))
    return {"planner": msgs}


def _build_extraction(mb, skills):
    tutorial = "<h1>Tutorial</h1>\n<p>Step 1: Do X.</p>\n<p>Step 2: Do Y.</p>"
    msgs = mb.build_skill_extraction_messages(tutorial, skills.instruction, [])
    print(_fmt(msgs, "Skill Extraction"))
    return {"extraction": msgs}


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from anything2skill.benchmarks.osworld.kit import OSWorldKit
    view_prompts(OSWorldKit())
