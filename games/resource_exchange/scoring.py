from typing import Dict, List


class ScoreCalculator:
    """
    Computes provisional, penalty, and final team scores.
    """

    def calculate_team_score(
        self, team_players: List[str], allocations: Dict[str, Dict[str, int]]
    ) -> Dict[str, int]:
        # Aggregate resources across players
        total_by_resource: Dict[str, int] = {}
        for p in team_players:
            for res, val in allocations[p].items():
                total_by_resource[res] = total_by_resource.get(res, 0) + val

        provisional = sum(total_by_resource.values())
        if not total_by_resource:
            return {"provisional": 0, "penalty": 0, "final": 0}

        mx = max(total_by_resource.values())
        mn = min(total_by_resource.values())
        penalty = mx - mn
        final = provisional - penalty
        return {
            "provisional": provisional,
            "penalty": penalty,
            "final": final,
            "by_resource": total_by_resource,
        }

