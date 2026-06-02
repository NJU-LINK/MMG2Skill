"""Domain-specific prompt text for No-Limit Hold'em."""

SYSTEM_PROMPT = """
You are a No-Limit Hold'em poker agent competing against one opponent.

Game rules:
- No-Limit Hold'em uses a standard 52-card deck.
- Each player starts with 100 chips and is dealt two private cards (hole cards).
- Blinds: small blind = 1 chip, big blind = 2 chips.
- There are four betting rounds:
  1. Preflop: After dealing hole cards, before any community cards.
  2. Flop: Three community cards are revealed.
  3. Turn: A fourth community card is revealed.
  4. River: A fifth community card is revealed.
- Unlike Limit Hold'em, bet sizes are not fixed. You choose from discrete sizing options:
  fold, check/call, raise half pot, raise full pot, or all-in.
- After the final round, the player with the best 5-card hand wins the pot.

Objective: Maximize your expected payoff. Your payoff equals the net chips won or lost (range: -100 to +100).

Hand rankings (highest to lowest):
- Royal Flush: A, K, Q, J, 10 of the same suit
- Straight Flush: Five consecutive cards of the same suit
- Four of a Kind: Four cards of the same rank
- Full House: Three of a kind plus a pair
- Flush: Five cards of the same suit
- Straight: Five consecutive cards
- Three of a Kind: Three cards of the same rank
- Two Pair: Two different pairs
- One Pair: Two cards of the same rank
- High Card: Highest card when no other hand is made

You will be shown:
- Your hole cards
- The community cards (as they are revealed)
- The pot total, your chips bet, all players' chips bet, and remaining stacks
- The legal actions available to you

Available actions:
- fold: Surrender your hand and forfeit the pot
- check_call: Check (no bet pending) or call (match the current bet)
- raise_half_pot: Raise by half the pot
- raise_pot: Raise by the full pot
- all_in: Bet all your remaining chips
- DONE: Signal that the episode is complete
- FAIL: Signal that the task cannot be completed

IMPORTANT: You MUST choose your action ONLY from the "Legal Actions" list provided in each observation. Do NOT guess which actions should be available — always trust the provided list.

Think through the situation first — assess your hand strength, pot odds, stack sizes, and decide on the best move. Then emit your action in an <action> tag:

<action>fold | check_call | raise_half_pot | raise_pot | all_in | DONE | FAIL</action>
"""

BRIDGING_TEXT = """
Given the current No-Limit Hold'em game state below, what is your next action?
"""

REFLECTION_GUIDANCE = """
Review your No-Limit Hold'em hand carefully:
- What is your current hand strength given the community cards?
- What are the community cards and how do they relate to your hole cards?
- What actions has your opponent taken?
"""

PLANNER_GUIDANCE = """
Observations are No-Limit Hold'em game states showing your hole cards, community cards,
chip counts, and legal actions.
Actions are: 'fold', 'check_call', 'raise_half_pot', 'raise_pot', or 'all_in'.
"""

SKILL_EXTRACTION_GUIDANCE = """
Each skill should describe a concrete No-Limit Hold'em decision strategy,
such as preflop hand ranges, bet sizing strategies, when to go all-in,
bluffing frequencies, and stack management principles.
"""

REVISER_GUIDANCE = """
Environment: No-Limit Hold'em (2 players, 52-card deck, 100 chip stacks),
text-only observations. The agent sees its hole cards, community cards
(flop/turn/river as revealed), pot total, chip counts, and legal actions.
Actions: fold, check_call, raise_half_pot, raise_pot, all_in. Framework
signals: DONE, FAIL.

Common failure patterns to watch for in trajectories:
- Over-committing with weak hands: agent goes all-in or raises pot-size
  with marginal holdings when the board is dangerous.
- Passive play with strong hands: agent merely check-calls with premium
  hands when raising would extract more value from the opponent.
- Ignoring stack-to-pot ratio: agent makes large bets when short-stacked,
  effectively committing all chips without explicitly choosing all_in.
- Board reading failure: agent does not recognize that community cards
  have changed the relative strength of its hand (e.g. a flush draw
  completing on the river for the opponent).
- Bluffing without fold equity: agent bluffs when the opponent is
  pot-committed and will call regardless.
- Illegal action: agent outputs an action not in the legal_actions list.
- Premature DONE: agent signals DONE before the hand is resolved.

IMPORTANT: Skill refinements must improve general strategy, NOT memorize
specific card sequences or deals observed in the trajectory. Skills must
generalize to any deal.
"""
