from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ResourceExchangeConfig:
    """Configuration for the 4-player resource exchange game."""

    total_rounds: int = 14
    chat_duration_seconds: int = 180
    exchange_duration_seconds: int = 30
    feedback_duration_seconds: int = 20

    # Chat timesteps: chat phase is split into this many synchronous ticks
    chat_timesteps: int = 3

    # Players/teams
    players: List[str] = field(default_factory=lambda: ["alice", "bob", "carol", "dave"])
    teams: Dict[str, List[str]] = field(
        default_factory=lambda: {"team_a": ["alice", "bob"], "team_b": ["carol", "dave"]}
    )

    # Pairing constraints
    teammate_pairings: int = 7
    opponent_pairings: Tuple[int, int] = (4, 3)  # (opponent1, opponent2) for each player

    # Resources
    resource_types: List[str] = field(default_factory=lambda: ["meat", "grain", "water", "fruit", "fish"])
    initial_points_total: int = 7  # per player; distributed as 3,2,1,1,0 pattern

    # Vocabulary
    vocabulary_size: int = 19
    seed: Optional[int] = None

    # Names
    randomize_names: bool = True  # if True, use randomized display names in logs/output

    # Logging
    log_dir: str = "./logs"

    # LLM backend config passthrough
    llm_backend: Dict = field(default_factory=dict)

