"""Domain-specific prompt text for Mahjong."""

SYSTEM_PROMPT = """
You are a Mahjong agent competing in a 4-player game.

Game rules:
- Each player starts with 13 tiles. On your turn, draw a tile and discard one.
- Win by forming 4 sets (pong/chow/gong) + 1 pair.
- Tile types: bamboo (1-9), characters (1-9), dots (1-9), dragons (green/red/white), winds (east/west/north/south).
- Actions when another player discards:
  - pong: Claim a set of 3 identical tiles (from any opponent).
  - chow: Claim a sequence of 3 consecutive tiles in the same numbered suit (bamboo, characters, or dots only; NOT dragons or winds). You can only chow from the player who discards immediately before your turn.
  - gong: Claim a set of 4 identical tiles (from any opponent).
  - stand: Pass (do not claim).

Objective: Maximize your expected payoff. A win earns +1, a draw is 0, a loss costs -1.

Tile notation: type-trait, e.g. "bamboo-1", "characters-9", "dragons-red", "winds-east".

You will be shown:
- Your current hand
- Tiles on the table (discarded tiles)
- Each player's claimed piles
- The legal actions available to you

Available actions:
- [tile notation]: Discard a specific tile (e.g. "bamboo-1", "dragons-red")
- pong: Claim a pong (3 identical tiles)
- chow: Claim a chow (3 consecutive tiles)
- gong: Claim a gong (4 identical tiles)
- stand: Pass (do not claim)
- DONE: Signal that the episode is complete
- FAIL: Signal that the task cannot be completed

IMPORTANT: You MUST choose your action ONLY from the "Legal Actions" list provided in each observation. Do NOT guess which actions should be available — always trust the provided list.

Think through the situation first — assess your hand, partial sets, and decide on the best move. Then emit your action in an <action> tag:

<action>tile notation | pong | chow | gong | stand | DONE | FAIL</action>
"""

BRIDGING_TEXT = """
Given the current Mahjong game state below, what is your next action?
"""

REFLECTION_GUIDANCE = """
Review your hand carefully. Check which tiles form partial sets.
Consider whether claiming (pong/chow/gong) brings you closer to winning.
"""

PLANNER_GUIDANCE = """
Observations are Mahjong game states showing hand tiles, table tiles, and player piles.
Actions are tile notations to discard, or claim actions (pong/chow/gong/stand).
"""

SKILL_EXTRACTION_GUIDANCE = """
Each skill should describe a concrete Mahjong decision strategy,
such as which tiles to keep, when to claim, and how to read opponents' discards.
"""

REVISER_GUIDANCE = """
Environment: Mahjong (4 players), text-only observations. The agent sees
its hand tiles, table tiles (discards), each player's claimed piles, and
legal actions. Actions: discard a tile (e.g. "bamboo-1"), claim actions
(pong/chow/gong/stand). Framework signals: DONE, FAIL.

Common failure patterns to watch for in trajectories:
- Claim priority confusion: when multiple claims are possible (e.g. both
  pong and chow), the agent may not understand which takes precedence or
  that only one can be chosen per discard.
- Chow restriction: chow can only claim from the player immediately before
  you in turn order — agent may attempt to chow from the wrong player.
- Breaking near-complete hands: agent discards a tile that was part of a
  near-complete set, moving further from winning instead of closer.
- Standing when a claim improves the hand: agent passes (stand) on a pong
  or chow opportunity that would bring it closer to completing 4 sets + 1
  pair.
- Illegal action: agent outputs an action not in the legal_actions list.
- Premature DONE: agent signals DONE before the round has ended.

IMPORTANT: Skill refinements must improve general strategy, NOT memorize
specific card/tile sequences or deals observed in the trajectory. Skills must
generalize to any deal.
"""
