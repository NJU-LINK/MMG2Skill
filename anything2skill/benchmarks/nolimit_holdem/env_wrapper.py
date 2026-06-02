"""No-Limit Hold'em environment wrapper."""

from __future__ import annotations

import logging

from anything2skill.benchmarks.rlcard_common import BaseRLCardEnvWrapper, ensure_rlcard_path

logger = logging.getLogger("anything2skill.benchmarks.nolimit_holdem.env")

ACTION_MAP = {
    "fold": 0,
    "check_call": 1,
    "raise_half_pot": 2,
    "raise_pot": 3,
    "all_in": 4,
}

ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


class NolimitHoldemEnvWrapper(BaseRLCardEnvWrapper):
    """Wraps RLCard No-Limit Hold'em (2 players, 100 chips each)."""

    game_name = "no-limit-holdem"
    _DIFFICULTY_OPPONENT = {"easy": "random", "hard": "dqn"}

    def __init__(self, seed: int | None = None, opponent_type: str = "auto"):
        ensure_rlcard_path()
        super().__init__(seed=seed, num_players=2, opponent_type=opponent_type)
        self.actions = ["fold", "check_call", "raise_half_pot", "raise_pot", "all_in"]

    def _build_obs(self, state: dict) -> dict:
        raw = state.get("raw_obs", {})
        hand: list[str] = raw.get("hand", [])
        public_cards: list[str] = raw.get("public_cards", [])
        my_chips: int = raw.get("my_chips", 0)
        all_chips: list = raw.get("all_chips", [])
        pot: int = raw.get("pot", 0)
        stakes: list = raw.get("stakes", [])

        # raw_legal_actions contains Action enum objects; convert to strings
        raw_legal = state.get("raw_legal_actions", [])
        legal_actions: list[str] = []
        for a in raw_legal:
            # Action enum has .name attribute; convert to lowercase
            if hasattr(a, "name"):
                legal_actions.append(a.name.lower())
            else:
                legal_actions.append(str(a).lower())

        # Determine the current betting round
        num_public = len(public_cards)
        if num_public == 0:
            round_name = "Preflop"
        elif num_public == 3:
            round_name = "Flop"
        elif num_public == 4:
            round_name = "Turn"
        else:
            round_name = "River"

        return {
            "game": "nolimit_holdem",
            "hand": hand,
            "public_cards": public_cards,
            "round": round_name,
            "my_chips": my_chips,
            "all_chips": all_chips,
            "pot": pot,
            "stakes": stakes,
            "legal_actions": legal_actions,
            "opponent_actions": self._last_opponent_actions,
            "raw_state": state,
        }

    def _action_str_to_id(self, action: str) -> int | None:
        action = action.strip().lower()
        return ACTION_MAP.get(action)

    def _decode_action_id(self, action_id: int) -> str:
        return ACTION_NAMES.get(action_id, str(action_id))
