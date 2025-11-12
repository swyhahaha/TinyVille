"""
Task Module for Multi-Agent Language Emergence Experiment

This module maintains natural language prompts describing agent tasks,
cooperation relationships, and game-specific objectives.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class GameType(Enum):
    """Defines different experimental game types."""
    RESOURCE_SCRAMBLE = "resource_scramble"      # Zero-sum resource competition
    GENERATIONAL_LABEL = "generational_label"    # Cultural transmission
    GOSSIP_FORAGER = "gossip_forager"           # Social language evolution
    COLORED_SHAPE_LENS = "colored_shape_lens"   # Linguistic relativity


class TaskDifficulty(Enum):
    """Curriculum learning difficulty levels."""
    PHASE_1_SIMPLE = 1      # N=3-4 resources, 2 features
    PHASE_2_MODERATE = 2    # N=6-8 resources, 3 features
    PHASE_3_COMPLEX = 3     # N=10+ resources, 4+ features


class PromptTemplate(ABC):
    """
    Abstract class for generating prompts for LLM-based agents.
    
    Prompts describe:
    - Agent's role and identity
    - Current task/objective
    - Cooperation and competition relationships
    - Communication constraints
    """
    
    @abstractmethod
    def get_system_prompt(self, agent_id: str, role: str) -> str:
        """
        Generate system prompt defining agent's identity and capabilities.
        
        Args:
            agent_id: Unique agent identifier
            role: Agent's role (e.g., "Team A member")
            
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def get_task_prompt(self, task_type: str, context: Dict[str, Any]) -> str:
        """
        Generate task-specific prompt.
        
        Args:
            task_type: Type of task (e.g., "find_resource", "teach_label")
            context: Current game state and relevant information
            
        Returns:
            Task description prompt
        """
        pass
    
    @abstractmethod
    def get_communication_prompt(self, mode: str, constraints: Dict[str, Any]) -> str:
        """
        Generate prompt for communication behavior.
        
        Args:
            mode: Communication mode ("intra_team" or "inter_team")
            constraints: Vocabulary size, message length limits
            
        Returns:
            Communication instruction prompt
        """
        pass
    
    @abstractmethod
    def get_strategic_prompt(self, is_zero_sum: bool, team_info: Dict[str, Any]) -> str:
        """
        Generate prompt for strategic behavior in competitive settings.
        
        Args:
            is_zero_sum: Whether the game is zero-sum
            team_info: Information about teammates and opponents
            
        Returns:
            Strategic guidance prompt
        """
        pass


class GameTask(ABC):
    """
    Abstract class representing a specific game/experiment task.
    
    Encapsulates:
    - Game rules and objectives
    - Reward structure
    - Win conditions
    - Information asymmetry setup
    """
    
    @abstractmethod
    def __init__(self, game_type: GameType, difficulty: TaskDifficulty):
        """
        Initialize a game task.
        
        Args:
            game_type: Type of experiment/game
            difficulty: Difficulty level for curriculum learning
        """
        pass
    
    @abstractmethod
    def get_objective(self, agent_id: str, role: str) -> str:
        """
        Get the objective description for a specific agent.
        
        Args:
            agent_id: Agent identifier
            role: Agent's role
            
        Returns:
            Natural language objective description
        """
        pass
    
    @abstractmethod
    def get_rules(self) -> List[str]:
        """
        Get list of game rules.
        
        Returns:
            List of rule statements
        """
        pass
    
    @abstractmethod
    def get_reward_structure(self) -> Dict[str, float]:
        """
        Get reward parameters.
        
        Returns:
            Dictionary with reward values (r_success, r_comm, r_privacy, etc.)
        """
        pass
    
    @abstractmethod
    def check_win_condition(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Check if a team/agent has won.
        
        Args:
            state: Current game state
            
        Returns:
            Winner ID if game is over, None otherwise
        """
        pass


class ResourceScrambleTask(GameTask):
    """
    Task for the "Resource Scramble" zero-sum game.
    
    Setup:
    - Team A has location clues
    - Team B has attribute clues (value, trap status)
    - Must communicate to identify high-value safe resources
    - Zero-sum: first to claim wins
    """
    
    @abstractmethod
    def generate_resources(self, num_resources: int) -> List[Dict[str, Any]]:
        """
        Generate random resource configuration.
        
        Args:
            num_resources: Number of resources to generate
            
        Returns:
            List of resource objects with positions, values, trap status
        """
        pass
    
    @abstractmethod
    def get_team_a_info(self) -> Dict[str, Any]:
        """
        Get information available to Team A (location clues).
        
        Returns:
            Dictionary with location information
        """
        pass
    
    @abstractmethod
    def get_team_b_info(self) -> Dict[str, Any]:
        """
        Get information available to Team B (attribute clues).
        
        Returns:
            Dictionary with value and safety information
        """
        pass
    
    @abstractmethod
    def validate_claim(self, agent_id: str, resource_id: str, 
                      state: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Validate a resource claim attempt.
        
        Args:
            agent_id: Agent making the claim
            resource_id: Resource being claimed
            state: Current game state
            
        Returns:
            Tuple of (success, reward)
        """
        pass


class GenerationalLabelTask(GameTask):
    """
    Task for the "Generational Label" cultural transmission experiment.
    
    Setup:
    - Chain of agents (G0 -> G1 -> G2 -> ... -> GN)
    - Must transmit concept-label mappings across generations
    - Measures conventionalization and drift
    """
    
    @abstractmethod
    def generate_concepts(self, num_concepts: int) -> List[str]:
        """
        Generate abstract concepts to be labeled.
        
        Args:
            num_concepts: Number of concepts
            
        Returns:
            List of concept identifiers (e.g., ["Shape-A", "Shape-B"])
        """
        pass
    
    @abstractmethod
    def initialize_generation_zero(self, agent: Any, concepts: List[str]):
        """
        Initialize G0 agent with arbitrary labels.
        
        Args:
            agent: G0 agent
            concepts: List of concepts to label
        """
        pass
    
    @abstractmethod
    def measure_transmission_fidelity(self, original: Dict[str, str], 
                                     current: Dict[str, str]) -> float:
        """
        Measure how faithfully labels were transmitted.
        
        Args:
            original: Original G0 concept-label mapping
            current: Current generation's mapping
            
        Returns:
            Fidelity score [0, 1]
        """
        pass


class CooperationRelationship(ABC):
    """
    Abstract class for defining cooperation/competition relationships.
    """
    
    @abstractmethod
    def get_relationship_type(self, agent_a: str, agent_b: str) -> str:
        """
        Get relationship type between two agents.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            
        Returns:
            Relationship type: "teammate", "opponent", "neutral"
        """
        pass
    
    @abstractmethod
    def get_trust_level(self, agent_a: str, agent_b: str) -> float:
        """
        Get trust level between two agents.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            
        Returns:
            Trust level [0, 1]
        """
        pass
    
    @abstractmethod
    def update_trust(self, agent_a: str, agent_b: str, 
                    interaction_outcome: float):
        """
        Update trust based on interaction outcome.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            interaction_outcome: Outcome value (positive or negative)
        """
        pass
    
    @abstractmethod
    def get_communication_strategy(self, sender: str, receiver: str) -> str:
        """
        Get recommended communication strategy based on relationship.
        
        Args:
            sender: Sending agent ID
            receiver: Receiving agent ID
            
        Returns:
            Strategy: "honest", "strategic", "deceptive"
        """
        pass


class PromptLibrary:
    """
    Static library of prompt templates for different experimental scenarios.
    """
    
    @staticmethod
    def get_resource_scramble_system_prompt(agent_id: str, team: str) -> str:
        """System prompt for Resource Scramble game."""
        raise NotImplementedError
    
    @staticmethod
    def get_abstract_symbol_constraint_prompt(vocabulary: List[str], 
                                             max_length: int) -> str:
        """Prompt enforcing abstract symbol usage."""
        raise NotImplementedError
    
    @staticmethod
    def get_strategic_communication_prompt(is_teammate: bool) -> str:
        """Prompt for strategic vs. honest communication."""
        raise NotImplementedError
    
    @staticmethod
    def get_importance_rating_prompt(observation: str) -> str:
        """Prompt for rating memory importance (1-10 scale)."""
        raise NotImplementedError
