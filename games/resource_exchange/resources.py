import itertools
import random
from typing import Dict, List, Tuple


class ResourceManager:
    """
    Handles initial allocation and exchanges with 2x gain for receiver.
    """

    def __init__(self, resource_types: List[str], seed: int = None):
        self.resource_types = resource_types
        self.rng = random.Random(seed)
        self.allocations: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------ #
    # Initial allocation
    # ------------------------------------------------------------------ #
    def generate_initial_allocations(self, teams: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Rules:
        - Each player total 7 points with pattern [3,2,1,1,0] over 5 resources.
        - No two players identical allocation.
        - For each team (2 players): the resource with 0 points for one has 3 for the teammate.
        """
        players = list(itertools.chain.from_iterable(teams.values()))
        assert len(players) == 4, "This game expects exactly 4 players."

        # helper to build a single allocation
        base_pattern = [3, 2, 1, 1, 0]

        team_allocs: Dict[str, Dict[str, int]] = {}
        for team, members in teams.items():
            assert len(members) == 2, "Each team must have 2 players."
            a, b = members

            # Player A
            perm_a = self._sample_unique_allocation(set(), base_pattern)
            # Player B: ensure complement on zero->3
            zero_res = [r for r, v in perm_a.items() if v == 0][0]

            # Build B pattern: set zero_res to 3, remaining shuffled pattern [2,1,1,0]
            remaining_pattern = [2, 1, 1, 0]
            perm_b = self._sample_unique_allocation(set([tuple(perm_a.items())]), remaining_pattern, fixed={zero_res: 3})

            team_allocs[a] = perm_a
            team_allocs[b] = perm_b

        # Ensure global uniqueness (across teams). If clash, resample.
        all_players = list(team_allocs.keys())
        attempts = 0
        while self._has_duplicates(team_allocs) and attempts < 200:
            attempts += 1
            # Resample entire set
            team_allocs.clear()
            for team, members in teams.items():
                a, b = members
                perm_a = self._sample_unique_allocation(set(), base_pattern)
                zero_res = [r for r, v in perm_a.items() if v == 0][0]
                remaining_pattern = [2, 1, 1, 0]
                perm_b = self._sample_unique_allocation(set([tuple(perm_a.items())]), remaining_pattern, fixed={zero_res: 3})
                team_allocs[a] = perm_a
                team_allocs[b] = perm_b

        if self._has_duplicates(team_allocs):
            raise RuntimeError("Failed to generate unique allocations after many attempts.")

        self.allocations = team_allocs
        return team_allocs

    def _sample_unique_allocation(
        self,
        disallow: set,
        pattern: List[int],
        fixed: Dict[str, int] = None,
    ) -> Dict[str, int]:
        fixed = fixed or {}
        remaining_resources = [r for r in self.resource_types if r not in fixed]
        # Shuffle resources for pattern assignment
        self.rng.shuffle(remaining_resources)
        pattern_copy = list(pattern)
        self.rng.shuffle(pattern_copy)

        allocation = fixed.copy()
        for r, v in zip(remaining_resources, pattern_copy):
            allocation[r] = v
        if tuple(allocation.items()) in disallow:
            # retry recursively
            return self._sample_unique_allocation(disallow, pattern, fixed=fixed)
        return allocation

    def _has_duplicates(self, allocs: Dict[str, Dict[str, int]]) -> bool:
        seen = set()
        for alloc in allocs.values():
            key = tuple(sorted(alloc.items()))
            if key in seen:
                return True
            seen.add(key)
        return False

    # ------------------------------------------------------------------ #
    # Exchange logic
    # ------------------------------------------------------------------ #
    def process_exchange(
        self, giver: str, receiver: str, resource: str, amount: int
    ) -> Dict[str, int]:
        # Validate resource and players; if invalid, do nothing but report error
        if resource not in self.resource_types:
            return {"giver_delta": 0, "receiver_delta": 0, "error": "unknown_resource", "resource": resource}
        if giver not in self.allocations or receiver not in self.allocations:
            return {"giver_delta": 0, "receiver_delta": 0, "error": "unknown_player"}
        if amount <= 0:
            return {"giver_delta": 0, "receiver_delta": 0, "error": "non_positive_amount"}

        giver_amt = self.allocations[giver].get(resource, 0)
        transfer = min(amount, giver_amt)
        self.allocations[giver][resource] = giver_amt - transfer
        self.allocations[receiver][resource] = self.allocations[receiver].get(resource, 0) + transfer * 2

        return {"giver_delta": -transfer, "receiver_delta": transfer * 2}

