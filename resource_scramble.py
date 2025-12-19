"""
Resource Scramble Game Implementation

Zero-sum game with information asymmetry to study pidgin language emergence:
- Team A knows resource LOCATIONS (coordinates)
- Team B knows resource ATTRIBUTES (value, safety)
- Both must communicate using abstract tokens (tok1-tok20) to succeed
"""

import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ResourceColor(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"


class ResourceShape(Enum):
    CIRCLE = "circle"
    SQUARE = "square"


@dataclass
class Resource:
    """A resource in the environment."""
    id: int
    x: float
    y: float
    color: ResourceColor
    shape: ResourceShape
    value: int  # Reward value
    is_trap: bool  # Whether this is a trap
    
    def get_location_info(self) -> Dict[str, Any]:
        """Information visible to Team A (location holders)."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y
        }
    
    def get_attribute_info(self) -> Dict[str, Any]:
        """Information visible to Team B (attribute holders)."""
        return {
            "id": self.id,
            "color": self.color.value,
            "shape": self.shape.value,
            "value": self.value,
            "is_trap": self.is_trap
        }


@dataclass
class CurriculumPhase:
    """Configuration for a curriculum learning phase."""
    phase_id: int
    num_resources: int  # N
    feature_dimensions: int  # Number of feature types
    success_threshold: float = 0.8
    min_episodes: int = 20


class ResourceScrambleEnvironment:
    """
    Resource Scramble game environment with information asymmetry.
    """
    
    def __init__(self, phase: CurriculumPhase, seed: Optional[int] = None):
        self.phase = phase
        self.resources: List[Resource] = []
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
    def reset(self) -> Dict[str, Any]:
        """Generate new resources for an episode."""
        self.resources = []
        
        for i in range(self.phase.num_resources):
            # Random position
            x = self.rng.uniform(0, 10)
            y = self.rng.uniform(0, 10)
            
            # Random attributes
            color = self.rng.choice(list(ResourceColor))
            shape = self.rng.choice(list(ResourceShape))
            
            # Value and trap distribution
            # ~30% are traps (negative value), 70% are positive
            is_trap = self.rng.random() < 0.3
            if is_trap:
                value = -10
            else:
                value = self.rng.randint(1, 10)
            
            resource = Resource(
                id=i,
                x=x, y=y,
                color=color,
                shape=shape,
                value=value,
                is_trap=is_trap
            )
            self.resources.append(resource)
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            "phase": self.phase.phase_id,
            "num_resources": len(self.resources),
            "resources": self.resources
        }
    
    def get_team_a_view(self) -> List[Dict[str, Any]]:
        """Get Team A's view (locations only)."""
        return [r.get_location_info() for r in self.resources]
    
    def get_team_b_view(self) -> List[Dict[str, Any]]:
        """Get Team B's view (attributes only)."""
        return [r.get_attribute_info() for r in self.resources]
    
    def evaluate_choice(self, resource_id: int) -> Tuple[int, bool]:
        """
        Evaluate a team's choice.
        
        Returns:
            (reward, is_correct) tuple
        """
        if resource_id < 0 or resource_id >= len(self.resources):
            return -5, False  # Invalid choice penalty
        
        resource = self.resources[resource_id]
        return resource.value, not resource.is_trap and resource.value > 0
    
    def get_optimal_choice(self) -> int:
        """Get the ID of the best (highest value, non-trap) resource."""
        best_id = 0
        best_value = -float('inf')
        
        for resource in self.resources:
            if not resource.is_trap and resource.value > best_value:
                best_value = resource.value
                best_id = resource.id
        
        return best_id


class AbstractVocabulary:
    """
    Forced abstract vocabulary system.
    
    Enforces communication through meaningless tokens (tok1-tok20)
    to prevent use of pre-trained natural language.
    """
    
    def __init__(self, size: int = 20, max_length: int = 5):
        self.size = size
        self.max_length = max_length
        self.tokens = [f"tok{i+1}" for i in range(size)]
        
    def is_valid_message(self, message: str) -> bool:
        """Check if message uses only vocabulary tokens."""
        tokens = message.strip().split(',')
        tokens = [t.strip() for t in tokens]
        
        if len(tokens) > self.max_length:
            return False
        
        for token in tokens:
            if token not in self.tokens and token != "":
                return False
        
        return True
    
    def parse_message(self, message: str) -> List[str]:
        """Parse message into token list."""
        if not self.is_valid_message(message):
            return []
        
        tokens = message.strip().split(',')
        return [t.strip() for t in tokens if t.strip()]
    
    def get_prompt_vocabulary(self) -> str:
        """Get vocabulary description for LLM prompts."""
        return f"Available tokens: {', '.join(self.tokens)}"
    
    def sample_initial_subset(self, subset_size: int = 10) -> List[str]:
        """Sample a subset for initial team vocabulary."""
        return random.sample(self.tokens, subset_size)


@dataclass
class GameRound:
    """Record of a single round."""
    round_id: int
    team_a_message: str
    team_b_message: str
    team_a_choice: int
    team_b_choice: int
    team_a_reward: int
    team_b_reward: int
    winner: str  # "A", "B", or "tie"


@dataclass
class EpisodeResult:
    """Results from one episode."""
    episode_id: int
    rounds: List[GameRound] = field(default_factory=list)
    total_team_a_reward: int = 0
    total_team_b_reward: int = 0
    winner: str = "tie"
    converged_round: int = -1  # Round where optimal was found
    
    def add_round(self, round_data: GameRound):
        """Add a round result."""
        self.rounds.append(round_data)
        self.total_team_a_reward += round_data.team_a_reward
        self.total_team_b_reward += round_data.team_b_reward
    
    def finalize(self):
        """Determine final winner."""
        if self.total_team_a_reward > self.total_team_b_reward:
            self.winner = "A"
        elif self.total_team_b_reward > self.total_team_a_reward:
            self.winner = "B"
        else:
            self.winner = "tie"


class CurriculumManager:
    """
    Manages curriculum learning progression.
    """
    
    def __init__(self):
        self.phases = [
            CurriculumPhase(
                phase_id=1,
                num_resources=4,
                feature_dimensions=2,  # color, shape
                success_threshold=0.8,
                min_episodes=20
            ),
            CurriculumPhase(
                phase_id=2,
                num_resources=6,
                feature_dimensions=2,
                success_threshold=0.8,
                min_episodes=30
            ),
            CurriculumPhase(
                phase_id=3,
                num_resources=8,
                feature_dimensions=2,
                success_threshold=0.85,
                min_episodes=40
            )
        ]
        
        self.current_phase_idx = 0
        self.phase_results: List[List[EpisodeResult]] = [[] for _ in self.phases]
    
    def get_current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase."""
        return self.phases[self.current_phase_idx]
    
    def record_episode(self, result: EpisodeResult):
        """Record episode result for current phase."""
        self.phase_results[self.current_phase_idx].append(result)
    
    def should_advance(self) -> bool:
        """Check if should advance to next phase."""
        if self.current_phase_idx >= len(self.phases) - 1:
            return False  # Already at final phase
        
        phase = self.get_current_phase()
        results = self.phase_results[self.current_phase_idx]
        
        if len(results) < phase.min_episodes:
            return False
        
        # Calculate success rate over recent episodes
        recent_results = results[-phase.min_episodes:]
        successes = sum(1 for r in recent_results 
                       if r.winner != "tie" and r.converged_round >= 0)
        success_rate = successes / len(recent_results)
        
        return success_rate >= phase.success_threshold
    
    def advance_phase(self) -> bool:
        """Advance to next phase."""
        if self.current_phase_idx < len(self.phases) - 1:
            self.current_phase_idx += 1
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall curriculum statistics."""
        stats = {}
        for i, phase in enumerate(self.phases):
            results = self.phase_results[i]
            if results:
                successes = sum(1 for r in results if r.winner != "tie")
                stats[f"phase_{i+1}"] = {
                    "episodes": len(results),
                    "success_rate": successes / len(results),
                    "avg_rounds": np.mean([len(r.rounds) for r in results])
                }
        return stats
