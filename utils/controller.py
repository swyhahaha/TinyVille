"""
Controller Module for Multi-Agent Language Emergence Experiment

This module manages the turn-based execution of agents, maintains global
system state, and coordinates interactions between agents and the environment.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import queue


class TurnOrder(Enum):
    """Defines different turn ordering strategies."""
    ROUND_ROBIN = "round_robin"          # Sequential turn-taking
    RANDOM = "random"                    # Random agent selection
    PRIORITY_BASED = "priority"          # Based on agent state/role
    SIMULTANEOUS = "simultaneous"        # All agents act simultaneously


class SimulationPhase(Enum):
    """Defines phases of the simulation."""
    INITIALIZATION = "init"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class GameController(ABC):
    """
    Abstract controller for managing multi-agent simulation.
    
    Responsibilities:
    - Determine agent turn order
    - Maintain global state (resources, positions, scores)
    - Process agent actions and update environment
    - Track game progress and termination conditions
    - Log interactions for analysis
    """
    
    @abstractmethod
    def __init__(self, agents: List[Any], task: Any, config: Dict[str, Any]):
        """
        Initialize the game controller.
        
        Args:
            agents: List of agent instances
            task: Game task definition
            config: Configuration parameters (turn_order, max_rounds, etc.)
        """
        pass
    
    # ==================== State Management ====================
    
    @abstractmethod
    def get_global_state(self) -> Dict[str, Any]:
        """
        Get complete global state.
        
        Returns:
            Dictionary containing:
            - current_round: Current round number
            - agent_states: Dictionary of agent states
            - environment_state: Resources, positions, etc.
            - scores: Team/agent scores
            - communication_log: History of messages
        """
        pass
    
    @abstractmethod
    def update_state(self, action_result: Dict[str, Any]):
        """
        Update global state based on action result.
        
        Args:
            action_result: Result from agent's execute_action()
        """
        pass
    
    @abstractmethod
    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Get partial state visible to a specific agent.
        
        Implements information asymmetry (e.g., Team A sees locations,
        Team B sees attributes).
        
        Args:
            agent_id: Agent requesting the view
            
        Returns:
            Filtered state dictionary
        """
        pass
    
    @abstractmethod
    def broadcast_observation(self, observation: str, exclude: Optional[List[str]] = None):
        """
        Broadcast an observation to all agents.
        
        Args:
            observation: Event description to broadcast
            exclude: Agent IDs to exclude from broadcast
        """
        pass
    
    # ==================== Turn Management ====================
    
    @abstractmethod
    def determine_turn_order(self) -> List[str]:
        """
        Determine the order of agent actions for current round.
        
        Returns:
            Ordered list of agent IDs
        """
        pass
    
    @abstractmethod
    def execute_round(self) -> Dict[str, Any]:
        """
        Execute one complete round of the simulation.
        
        Process:
        1. Determine turn order
        2. For each agent:
           a. Get agent's view of state
           b. Agent plans action
           c. Execute action
           d. Update state
           e. Broadcast observation
        3. Check termination conditions
        
        Returns:
            Round summary with actions taken and state changes
        """
        pass
    
    @abstractmethod
    def schedule_agent_action(self, agent_id: str, priority: int = 0):
        """
        Schedule an agent action (for event-driven execution).
        
        Args:
            agent_id: Agent to schedule
            priority: Priority level (higher = earlier)
        """
        pass
    
    @abstractmethod
    def process_simultaneous_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multiple agent actions that occur simultaneously.
        
        Handles conflicts (e.g., two agents claiming same resource).
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            Results for each action after conflict resolution
        """
        pass
    
    # ==================== Communication Management ====================
    
    @abstractmethod
    def route_message(self, sender: str, recipient: str, 
                     message: List[str], mode: str) -> bool:
        """
        Route a message from sender to recipient.
        
        Validates message format and applies communication costs.
        
        Args:
            sender: Sending agent ID
            recipient: Receiving agent ID
            message: Abstract symbol sequence
            mode: Communication mode ("intra_team" or "inter_team")
            
        Returns:
            True if message was delivered, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_message(self, message: List[str], vocabulary: List[str], 
                        max_length: int) -> Tuple[bool, Optional[str]]:
        """
        Validate message against vocabulary and length constraints.
        
        Args:
            message: Token sequence to validate
            vocabulary: Allowed vocabulary
            max_length: Maximum message length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def log_communication(self, sender: str, recipient: str, 
                         message: List[str], mode: str, success: bool):
        """
        Log a communication event for analysis.
        
        Args:
            sender: Sending agent ID
            recipient: Receiving agent ID
            message: Message content
            mode: Communication mode
            success: Whether message was successfully delivered
        """
        pass
    
    # ==================== Game Progress ====================
    
    @abstractmethod
    def check_termination(self) -> Tuple[bool, Optional[str]]:
        """
        Check if game should terminate.
        
        Termination conditions:
        - Max rounds reached
        - Win condition satisfied
        - All agents stuck/failed
        
        Returns:
            Tuple of (should_terminate, termination_reason)
        """
        pass
    
    @abstractmethod
    def calculate_final_scores(self) -> Dict[str, float]:
        """
        Calculate final scores for all agents/teams.
        
        Returns:
            Dictionary mapping agent/team ID to final score
        """
        pass
    
    @abstractmethod
    def get_game_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive game statistics for analysis.
        
        Returns:
            Dictionary with:
            - total_rounds: Number of rounds played
            - communication_stats: Message counts, vocabulary usage
            - language_evolution: Symbol-meaning mappings over time
            - success_rates: Task completion rates
            - reward_breakdown: r_success, r_comm, r_privacy totals
        """
        pass
    
    # ==================== Curriculum Learning ====================
    
    @abstractmethod
    def adjust_difficulty(self, performance_metrics: Dict[str, float]):
        """
        Adjust task difficulty based on agent performance.
        
        Implements curriculum learning:
        - If success_rate > 0.8, increase difficulty
        - If success_rate < 0.3, decrease difficulty
        
        Args:
            performance_metrics: Recent performance data
        """
        pass
    
    @abstractmethod
    def should_advance_phase(self) -> bool:
        """
        Determine if simulation should advance to next difficulty phase.
        
        Returns:
            True if ready to advance, False otherwise
        """
        pass


class ActionQueue(ABC):
    """
    Abstract class for managing scheduled agent actions.
    """
    
    @abstractmethod
    def enqueue(self, agent_id: str, action: str, priority: int = 0):
        """
        Add an action to the queue.
        
        Args:
            agent_id: Agent performing the action
            action: Action to perform
            priority: Priority level
        """
        pass
    
    @abstractmethod
    def dequeue(self) -> Optional[Tuple[str, str]]:
        """
        Remove and return the next action.
        
        Returns:
            Tuple of (agent_id, action) or None if queue empty
        """
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all actions from queue."""
        pass


class ConflictResolver(ABC):
    """
    Abstract class for resolving conflicts in simultaneous actions.
    """
    
    @abstractmethod
    def resolve_resource_conflict(self, claims: List[Tuple[str, str]]) -> str:
        """
        Resolve conflict when multiple agents claim same resource.
        
        Args:
            claims: List of (agent_id, resource_id) tuples
            
        Returns:
            ID of agent who successfully claims the resource
        """
        pass
    
    @abstractmethod
    def resolve_communication_conflict(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve conflict when agents send conflicting messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Processed messages after conflict resolution
        """
        pass
    
    @abstractmethod
    def apply_tie_breaker(self, candidates: List[str], context: Dict[str, Any]) -> str:
        """
        Apply tie-breaking rule when multiple valid options exist.
        
        Args:
            candidates: List of candidate agent IDs
            context: Additional context for decision
            
        Returns:
            Selected agent ID
        """
        pass


class StateLogger(ABC):
    """
    Abstract class for logging simulation state and events.
    """
    
    @abstractmethod
    def log_round(self, round_num: int, state: Dict[str, Any]):
        """
        Log complete state for a round.
        
        Args:
            round_num: Round number
            state: Global state snapshot
        """
        pass
    
    @abstractmethod
    def log_action(self, agent_id: str, action: str, result: Dict[str, Any]):
        """
        Log an agent action and its result.
        
        Args:
            agent_id: Agent performing action
            action: Action description
            result: Action outcome
        """
        pass
    
    @abstractmethod
    def log_language_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log language-related events for analysis.
        
        Event types:
        - "symbol_mapping_update": New token-concept association
        - "pidgin_emergence": Cross-team shared symbol
        - "language_differentiation": Internal vs. external vocabulary split
        
        Args:
            event_type: Type of language event
            data: Event-specific data
        """
        pass
    
    @abstractmethod
    def export_logs(self, format: str = "json") -> str:
        """
        Export all logs for analysis.
        
        Args:
            format: Export format ("json", "csv", "pickle")
            
        Returns:
            File path of exported logs
        """
        pass
