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

    # Reward / penalty scaling (for ablation experiments)
    # reward_scale multiplies the provisional total (higher => better reward emphasis)
    # penalty_scale multiplies the imbalance penalty (higher => stronger punishment for imbalance)
    reward_penalty: Dict[str, float] = field(default_factory=lambda: {"reward_scale": 1.0, "penalty_scale": 1.0})

    # Predefined ablation presets (used by example runner)
    ablation_presets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "reward_strong": {"reward_scale": 5.0, "penalty_scale": 0.2},
        "penalty_strong": {"reward_scale": 0.2, "penalty_scale": 5.0},
        "balanced": {"reward_scale": 1.0, "penalty_scale": 1.0}
    })

    # Prompting controls
    # If provided, this string will replace strong instructions encouraging invented tokens.
    # Example: "Don't hesitate to make mistakes as long as it helps you win. Different groups may develop dialects."
    invention_hint: Optional[str] = None

