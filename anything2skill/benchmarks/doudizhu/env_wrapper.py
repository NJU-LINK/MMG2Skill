"""Doudizhu environment wrapper."""

from __future__ import annotations

import logging

from anything2skill.benchmarks.rlcard_common import BaseRLCardEnvWrapper, ensure_rlcard_path

logger = logging.getLogger("anything2skill.benchmarks.doudizhu.env")


class DoudizhuEnvWrapper(BaseRLCardEnvWrapper):
    """Wraps RLCard Doudizhu (3 players: 1 Landlord + 2 Peasants)."""

    game_name = "doudizhu"
    _DIFFICULTY_OPPONENT = {"easy": "random", "hard": "dmc"}

    def __init__(self, seed: int | None = None, opponent_type: str = "auto"):
        super().__init__(seed=seed, opponent_type=opponent_type)

    def evaluate(self) -> float:
        # Doudizhu payoffs are 0/1, not -1/1: losing side gets 0, not negative.
        if not self._done:
            payoffs = self._env.get_payoffs()
            self._payoff = float(payoffs[0])
        if self._payoff > 0:
            return 1.0
        return 0.0

    def _build_obs(self, state: dict) -> dict:
        raw = state.get("raw_obs", {})

        current_hand: str = raw.get("current_hand", "")
        others_hand: str = raw.get("others_hand", "")
        trace: list = raw.get("trace", [])
        played_cards: dict = raw.get("played_cards", {})
        num_cards_left: list[int] = raw.get("num_cards_left", [])
        player_id: int = raw.get("self", 0)
        legal_actions: list[str] = state.get("raw_legal_actions", [])

        role = "landlord" if player_id == 0 else "peasant"

        # Build recent action history from trace
        recent_actions: list[dict[str, str]] = []
        for pid, action_str in trace[-10:]:
            role_name = "landlord" if pid == 0 else f"peasant_{pid}"
            recent_actions.append({"player": role_name, "action": action_str})

        return {
            "game": "doudizhu",
            "hand": current_hand,
            "role": role,
            "player_id": player_id,
            "recent_actions": recent_actions,
            "others_hand": others_hand,
            "played_cards": played_cards,
            "num_cards_left": num_cards_left,
            "legal_actions": legal_actions,
            "raw_state": state,
        }

    def _action_str_to_id(self, action: str) -> int | None:
        ensure_rlcard_path()
        from rlcard.games.doudizhu.utils import ACTION_2_ID  # noqa: PLC0415

        action = action.strip()
        if action in ACTION_2_ID:
            return ACTION_2_ID[action]

        logger.warning("Doudizhu action '%s' not found in ACTION_2_ID", action)
        return None

    def _decode_action_id(self, action_id: int) -> str:
        ensure_rlcard_path()
        from rlcard.games.doudizhu.utils import ID_2_ACTION  # noqa: PLC0415

        if 0 <= action_id < len(ID_2_ACTION):
            return ID_2_ACTION[action_id]
        return str(action_id)
