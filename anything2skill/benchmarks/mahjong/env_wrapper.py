"""Mahjong environment wrapper."""

from __future__ import annotations

import logging
from typing import Any

from anything2skill.benchmarks.rlcard_common import BaseRLCardEnvWrapper, ensure_rlcard_path

logger = logging.getLogger("anything2skill.benchmarks.mahjong.env")


class MahjongEnvWrapper(BaseRLCardEnvWrapper):
    """Wraps RLCard Mahjong (4 players, fixed)."""

    game_name = "mahjong"
    _DIFFICULTY_OPPONENT = {"easy": "random", "hard": "dmc"}

    def __init__(self, seed: int | None = None, opponent_type: str = "auto"):
        super().__init__(seed=seed, opponent_type=opponent_type)

    def _build_obs(self, state: dict) -> dict:
        ensure_rlcard_path()
        from rlcard.games.mahjong.utils import card_decoding_dict  # noqa: PLC0415

        raw = state.get("raw_obs", {})
        hand = raw.get("current_hand", [])
        hand_strs = [c.get_str() if hasattr(c, "get_str") else str(c) for c in hand]
        table = raw.get("table", [])
        table_strs = [c.get_str() if hasattr(c, "get_str") else str(c) for c in table]

        players_pile = raw.get("players_pile", {})
        piles: dict[int, list[str]] = {}
        for pid, pile in players_pile.items():
            pile_cards = []
            for group in pile:
                for c in group:
                    pile_cards.append(c.get_str() if hasattr(c, "get_str") else str(c))
            piles[pid] = pile_cards

        # Use integer legal_actions + decoding dict instead of raw_legal_actions,
        # because raw_legal_actions contains Card objects (not action strings)
        # during claim scenarios (pong/chow/gong/stand).
        legal_ids = list(state.get("legal_actions", {}).keys())
        legal_strs = [card_decoding_dict.get(a, str(a)) for a in legal_ids]

        return {
            "game": "mahjong",
            "hand": hand_strs,
            "hand_size": len(hand_strs),
            "table": table_strs,
            "player_piles": piles,
            "legal_actions": legal_strs,
            "num_players": self._num_players,
            "opponent_actions": self._last_opponent_actions,
            "raw_state": state,
        }

    def _action_str_to_id(self, action: str) -> int | None:
        ensure_rlcard_path()
        from rlcard.games.mahjong.utils import card_encoding_dict  # noqa: PLC0415

        action = action.strip()
        if action in card_encoding_dict:
            return card_encoding_dict[action]

        action_lower = action.lower()
        for key, idx in card_encoding_dict.items():
            if key.lower() == action_lower:
                return idx

        logger.warning("Mahjong action '%s' not found in encoding dict", action)
        return None

    def _decode_action_id(self, action_id: int) -> str:
        ensure_rlcard_path()
        from rlcard.games.mahjong.utils import card_decoding_dict  # noqa: PLC0415
        return card_decoding_dict.get(action_id, str(action_id))
