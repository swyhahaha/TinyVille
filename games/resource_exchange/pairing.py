import random
from typing import Dict, List, Tuple


class PairingManager:
    """
    Generates 14-round pairing schedule for 4 players (2 per team) with rules:
    - First 3 rounds are teammate pairings (players are told only first 2).
    - Total teammate pairings: 7 rounds.
    - Two distinct opponents: pair counts 3 and 4 (order randomized).
    - No more than 3 consecutive rounds with the same partner.

    players: list of player_ids length 4
    teams: dict player_id -> team_id
    """

    def __init__(self, players: List[str], teams: Dict[str, str], seed: int = None):
        assert len(players) == 4, "This game assumes exactly 4 players."
        self.players = players
        self.teams = teams
        self.seed = seed
        self.rng = random.Random(seed)
        self.schedule: List[Dict[str, str]] = []

        # Precompute teammates/opponents map
        self.teammate: Dict[str, str] = {}
        self.opponents: Dict[str, List[str]] = {p: [] for p in players}
        for p in players:
            team = teams[p]
            mates = [x for x in players if x != p and teams[x] == team]
            opps = [x for x in players if teams[x] != team]
            self.teammate[p] = mates[0]
            self.opponents[p] = opps

    def generate(self) -> List[Dict[str, str]]:
        # Round 1-3: teammate
        rounds: List[Dict[str, str]] = []
        for _ in range(3):
            rounds.append(self._pair_teammates())

        # Remaining counts
        teammate_remaining = 7 - 3  # 4 more teammate rounds
        opp_counts = self._opponent_counts_template()

        # Build remaining 11 rounds by sampling partner choices respecting counts/constraints
        while len(rounds) < 14:
            round_plan = self._sample_round(rounds, teammate_remaining, opp_counts)
            rounds.append(round_plan)
            # Update counts
            if self._is_teammate_round(round_plan):
                teammate_remaining -= 1
            else:
                self._decrement_opponent_counts(round_plan, opp_counts)

        self.schedule = rounds
        return rounds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pair_teammates(self) -> Dict[str, str]:
        mapping = {}
        for p in self.players:
            mapping[p] = self.teammate[p]
        return mapping

    def _pair_opponent(self, p: str, opp: str) -> Dict[str, str]:
        mapping = {}
        pairs = [(p, opp)]
        others = [x for x in self.players if x not in (p, opp)]
        # The remaining two must be each other
        pairs.append((others[0], others[1]))
        for a, b in pairs:
            mapping[a] = b
            mapping[b] = a
        return mapping

    def _is_teammate_round(self, plan: Dict[str, str]) -> bool:
        # Check first player only (symmetric)
        p0 = self.players[0]
        return self.teams[p0] == self.teams[plan[p0]]

    def _opponent_counts_template(self) -> Dict[Tuple[str, str], int]:
        """
        Counts for each directed pairing with opponents:
        For each player p: three counts (3 vs opp1, 4 vs opp2) but to simplify,
        we track per unordered pair to decrement consistently.
        """
        # For team A player p, opponents list is length 2
        counts = {}
        # choose ordering stable by name
        for p in self.players:
            for opp in self.opponents[p]:
                pair = tuple(sorted([p, opp]))
                if pair not in counts:
                    counts[pair] = 0
                # Each unordered pair total count derives from one player's target:
                # Between the two opponents, counts are 3 and 4. We'll set them once below.
        # Determine opponent pairs globally
        opp_pairs = [tuple(sorted(self.opponents[self.players[0]]))]
        opp_a, opp_b = opp_pairs[0]
        # Set counts for pairs (teamA-player, opp_a) and (teamA-player, opp_b)
        # We don't know which player is teamA-player, but counts per unordered pair suffice.
        # Assign 3 to one opponent pair and 4 to the other; choose randomly.
        a_three = self.rng.choice([True, False])
        for pair in counts:
            if opp_a in pair and opp_b not in pair:
                counts[pair] = 3 if a_three else 4
            elif opp_b in pair and opp_a not in pair:
                counts[pair] = 4 if a_three else 3
        return counts

    def _remaining_partner_options(self, counts: Dict[Tuple[str, str], int]) -> Dict[str, List[str]]:
        options = {p: [] for p in self.players}
        for pair, cnt in counts.items():
            if cnt <= 0:
                continue
            a, b = pair
            options[a].append(b)
            options[b].append(a)
        return options

    def _decrement_opponent_counts(self, plan: Dict[str, str], counts: Dict[Tuple[str, str], int]):
        seen = set()
        for p, partner in plan.items():
            if p in seen:
                continue
            pair = tuple(sorted([p, partner]))
            if pair in counts and counts[pair] > 0:
                counts[pair] -= 1
            seen.add(p)
            seen.add(partner)

    def _sample_round(self, rounds: List[Dict[str, str]], teammate_remaining: int,
                      opp_counts: Dict[Tuple[str, str], int]) -> Dict[str, str]:
        max_retry = 1000
        for _ in range(max_retry):
            # Decide if this round is teammate or opponent-based
            remaining_rounds = 14 - len(rounds)
            need_teammate = teammate_remaining
            need_opponent = remaining_rounds - need_teammate
            choose_teammate = False
            if need_teammate > 0 and need_opponent > 0:
                choose_teammate = self.rng.random() < (need_teammate / remaining_rounds)
            elif need_teammate > 0:
                choose_teammate = True
            # Build plan
            if choose_teammate:
                plan = self._pair_teammates()
            else:
                options = self._remaining_partner_options(opp_counts)
                # pick a player with available options
                pivot = self.rng.choice([p for p, opts in options.items() if opts])
                partner = self.rng.choice(options[pivot])
                plan = self._pair_opponent(pivot, partner)

            if self._violates_consecutive(rounds, plan):
                continue
            return plan
        # Fallback: return teammate plan to avoid failure
        return self._pair_teammates()

    def _violates_consecutive(self, rounds: List[Dict[str, str]], plan: Dict[str, str]) -> bool:
        """
        No more than 3 consecutive rounds with same partner.
        """
        if len(rounds) < 3:
            return False
        for p in self.players:
            if all(r[p] == plan[p] for r in rounds[-3:]):
                return True
        return False


