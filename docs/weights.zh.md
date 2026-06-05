# RLCard 人机权重

RLCard benchmark 让 VLM agent（座位 0）与其余座位上训练好的 RL 对手（人机）对战。
三个 benchmark 游戏各自固定一种人机类型：

| 游戏 | 人机类型 | 目录 | 权重文件 |
|------|---------|------|---------|
| `doudizhu`（斗地主） | **DMC** | `rlcard_models/doudizhu_dmc/` | `checkpoint_dmc.pt`、`checkpoint_dmc_p0.pt`、`checkpoint_dmc_p1.pt`、`checkpoint_dmc_p2.pt` |
| `mahjong`（麻将） | **NFSP** | `rlcard_models/mahjong_nfsp/` | `checkpoint_nfsp.pt` |
| `no-limit-holdem`（无限注德州） | **DQN** | `rlcard_models/no-limit-holdem_dqn/` | `checkpoint_dqn.pt` |

只有 **doudizhu** 和 **mahjong** 计入 success-inferable 的 *Strategy* 评测
（30 + 30 = 60 手）。**无限注德州是私有信息诊断项**：它的 payoff 取决于对手底牌，
而底牌可能从不出现在 agent 可见的轨迹里，因此单独汇报、不计入核心分数。
DQN 权重仍随仓库提供，以便复现该诊断。LLM 评测期间对手**保持固定**（不接收任何反馈）。

`random` 对手始终可用，**不需要**任何权重。目录与文件命名是
[`anything2skill/benchmarks/rlcard_common.py`](../anything2skill/benchmarks/rlcard_common.py)
读取的约定：`create_opponent()` 会去找 `rlcard_models/{game}_{type}/{checkpoint_file}`，
所以上表的布局必须严格保持。

## 为什么权重放在 GitHub Release

`rlcard_models/` 在 `.gitignore` 里——`.pt` 权重文件太大，不能进 git 仓库，
改为作为 **GitHub Release** 的附件分发。刚 `git clone` 下来的仓库，
`rlcard_models/` 目录是空的（甚至不存在），所以跑 RLCard benchmark 前
要先把权重下载到对应目录里。

每个游戏目录会被打成一个 `<game>_<agent>.tar.gz` 附件
（例如 `doudizhu_dmc.tar.gz`），解压到 `rlcard_models/` 后即可还原出
该目录及其中的 `.pt` 文件。

## 获取权重（下载）

```bash
# 从默认 release tag（weights-v1）拉取全部权重包
bash scripts/rlcard/download_weights.sh

# 指定 release 或其他仓库
bash scripts/rlcard/download_weights.sh --tag weights-v2
bash scripts/rlcard/download_weights.sh --repo NJU-LINK/MMG2Skill

# 只下载某一个游戏
bash scripts/rlcard/download_weights.sh --game doudizhu_dmc
```

脚本会把每个 tar 包解压回 `rlcard_models/<game>_<agent>/`。
需要 GitHub CLI（`gh auth login`；公开仓库只需读权限）。

手动方式——从 release 页面下载附件后执行：

```bash
mkdir -p rlcard_models
tar xzf doudizhu_dmc.tar.gz -C rlcard_models   # → rlcard_models/doudizhu_dmc/*.pt
```

## 发布权重（上传，维护者）

先把权重训练好（或放好）到 `rlcard_models/` 下
（`bash scripts/rlcard/setup.sh`），然后：

```bash
# 打包 rlcard_models/*/*.pt 并上传到默认 release tag（weights-v1）
bash scripts/rlcard/upload_weights.sh

# 自定义 tag / 仓库 / 单个目录
bash scripts/rlcard/upload_weights.sh --tag weights-v2
bash scripts/rlcard/upload_weights.sh --repo NJU-LINK/MMG2Skill
bash scripts/rlcard/upload_weights.sh --dir doudizhu_dmc
```

release 不存在时脚本会自动创建，并以 `--clobber` 覆盖同名附件，
所以重新训练后再跑一次即可覆盖旧的 tar 包。
需要 `gh` 已认证且对目标仓库有 **push** 权限。

## 权重如何训练

对手用 RLCard 官方实现训练后冻结；`scripts/rlcard/setup.sh` 复现了这些默认值：

| 游戏 / 对手 | 训练预算 | 设置 |
|------------|---------|------|
| `doudizhu` / DMC | 10⁸ frames | 地主与两个农民座位各自独立的 checkpoint（即 `checkpoint_dmc_p{0,1,2}.pt`） |
| `mahjong` / NFSP | 5×10⁴ episodes | 座位 0 为学习者，其余座位用 `RandomAgent` |
| `no-limit-holdem` / DQN | 5×10⁴ episodes | 两人单挑（heads-up） |

这些与 `setup.sh` 直接对应（DMC 的 `TOTAL_FRAMES=100000000`，
NFSP/DQN 的 `NUM_EPISODES=DQN_EPISODES=50000`），也与该脚本里固定的
`(game → opponent)` 表一致。

## 运行时如何选择权重

人机类型在 `configs/benchmark/<game>.yaml` 的 `env.opponent_type`
（`dmc | nfsp | dqn | random | auto`）中按游戏指定。设为 `auto` 时，
加载器会按 `GAME_OPPONENTS` 顺序尝试已训练的权重，全都缺失则回退到
`random`。若显式指定的类型缺少权重，会抛出 `FileNotFoundError` 并提示回到
`setup.sh`——这时跑一下下载脚本即可修复。

人机的训练方式见 `scripts/rlcard/setup.sh` 与 `scripts/rlcard/train_agent.py`。
