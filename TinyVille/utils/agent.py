"""
Agent Module for Multi-Agent Language Emergence Experiment

This module implements agents with memory, planning, and execution capabilities
for simulating emergent language in zero-sum game environments.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .memory import MemoryStream, AbstractSymbolMemory


class AgentRole(Enum):
    """Defines agent roles in the experiment."""
    TEAM_A_MEMBER = "team_a"
    TEAM_B_MEMBER = "team_b"
    NEUTRAL = "neutral"


class CommunicationMode(Enum):
    """Defines communication modes."""
    INTRA_TEAM = "intra_team"  # Internal team communication
    INTER_TEAM = "inter_team"  # Cross-team communication (pidgin)
    BROADCAST = "broadcast"    # Public announcement


class BaseAgent(ABC):
    """
    Abstract base class for agents in the language emergence experiment.
    
    Core capabilities:
    - Memory: Maintain and retrieve experiences
    - Plan: Generate action plans based on goals and context
    - Execute: Perform actions in the environment
    """
    
    @abstractmethod
    def __init__(self, agent_id: str, role: AgentRole, vocabulary: List[str]):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent's role (Team A, Team B, etc.)
            vocabulary: Abstract symbol vocabulary for this agent
        """
        pass
    
    # ==================== Memory Functions ====================
    
    @abstractmethod
    def perceive(self, observation: str, metadata: Optional[Dict] = None):
        """
        Perceive and store an observation in memory.
        
        Args:
            observation: Natural language description of observed event
            metadata: Additional context (e.g., who performed action, location)
        """
        pass
    
    @abstractmethod
    def retrieve_memories(self, query: str, top_k: int = 10) -> List[Any]:
        """
        Retrieve relevant memories for decision-making.
        
        Args:
            query: Query for memory retrieval
            top_k: Number of memories to retrieve
            
        Returns:
            List of relevant MemoryObjects
        """
        pass
    
    @abstractmethod
    def reflect(self) -> List[str]:
        """
        Generate high-level reflections from recent experiences.
        Synthesizes observations into broader insights.
        
        Returns:
            List of reflection statements
        """
        pass
    
    # ==================== Planning Functions ====================
    
    @abstractmethod
    def plan(self, goal: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate a plan to achieve a goal.
        
        Args:
            goal: High-level objective (e.g., "find high-value resource")
            context: Current state information
            
        Returns:
            Ordered list of actions to execute
        """
        pass
    
    @abstractmethod
    def evaluate_plan(self, plan: List[str], context: Dict[str, Any]) -> float:
        """
        Evaluate the expected utility of a plan.
        
        Args:
            plan: Sequence of actions
            context: Current state
            
        Returns:
            Expected reward/utility score
        """
        pass
    
    @abstractmethod
    def replan(self, failed_action: str, context: Dict[str, Any]) -> List[str]:
        """
        Revise plan when an action fails.
        
        Args:
            failed_action: The action that failed
            context: Updated state
            
        Returns:
            Revised action sequence
        """
        pass
    
    # ==================== Execution Functions ====================
    
    @abstractmethod
    def execute_action(self, action: str, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a single action.
        
        Args:
            action: Action to perform (e.g., "communicate", "move", "claim_resource")
            target: Target of the action (e.g., another agent, a resource)
            
        Returns:
            Result dictionary with status, effects, and observations
        """
        pass
    
    @abstractmethod
    def communicate(self, recipient: str, message: List[str], 
                   mode: CommunicationMode = CommunicationMode.INTER_TEAM) -> bool:
        """
        Send a message using abstract symbols.
        
        Args:
            recipient: ID of the receiving agent
            message: Sequence of abstract tokens (e.g., ['tok3', 'tok7'])
            mode: Communication mode (intra-team vs inter-team)
            
        Returns:
            True if message was valid and sent, False otherwise
        """
        pass
    
    @abstractmethod
    def receive_message(self, sender: str, message: List[str], 
                       mode: CommunicationMode) -> str:
        """
        Receive and decode a message.
        
        Args:
            sender: ID of the sending agent
            message: Sequence of abstract tokens
            mode: Communication mode
            
        Returns:
            Interpreted meaning of the message
        """
        pass
    
    # ==================== Language Evolution Functions ====================
    
    @abstractmethod
    def encode_intent(self, intent: str, mode: CommunicationMode) -> List[str]:
        """
        Encode an intent into abstract symbols.
        
        Strategic encoding: In INTER_TEAM mode, may hide critical info.
        
        Args:
            intent: What the agent wants to communicate
            mode: Communication mode (affects honesty/transparency)
            
        Returns:
            Sequence of abstract tokens
        """
        pass
    
    @abstractmethod
    def decode_message(self, message: List[str], sender_role: AgentRole) -> str:
        """
        Decode abstract symbols into meaning.
        
        Args:
            message: Sequence of abstract tokens
            sender_role: Role of the sender (affects interpretation)
            
        Returns:
            Interpreted meaning
        """
        pass
    
    @abstractmethod
    def update_language_model(self, token: str, concept: str, 
                            success_feedback: float):
        """
        Update internal symbol-meaning associations based on interaction outcomes.
        
        Args:
            token: Abstract token (e.g., 'tok7')
            concept: Associated concept
            success_feedback: Reward signal from the interaction
        """
        pass
    
    # ==================== Utility Functions ====================
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get agent's current internal state.
        
        Returns:
            Dictionary with memory summary, current goal, beliefs, etc.
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """
        Calculate reward from action outcome.
        
        Includes:
        - r_success: Task completion reward
        - r_comm: Communication cost
        - r_privacy: Information hiding bonus (in zero-sum context)
        
        Args:
            outcome: Result of executed action
            
        Returns:
            Total reward value
        """
        pass


class TeamAgent(BaseAgent):
    """
    Abstract class for team-based agents in zero-sum games.
    
    Extends BaseAgent with team-specific capabilities:
    - Intra-team coordination
    - Strategic inter-team communication
    - Language differentiation (pidgin vs. internal jargon)
    """
    
    @abstractmethod
    def get_teammates(self) -> List[str]:
        """
        Get list of teammate agent IDs.
        
        Returns:
            List of teammate IDs
        """
        pass
    
    @abstractmethod
    def share_information(self, teammate: str, info: Dict[str, Any]):
        """
        Share information with a teammate (internal communication).
        
        Uses full/honest encoding since there's no conflict of interest.
        
        Args:
            teammate: Teammate agent ID
            info: Information to share
        """
        pass
    
    @abstractmethod
    def negotiate(self, opponent: str, proposal: str) -> Tuple[List[str], float]:
        """
        Negotiate with an opponent using strategic communication.
        
        Args:
            opponent: Opponent agent ID
            proposal: What to propose
            
        Returns:
            Tuple of (encoded message, expected_opponent_compliance)
        """
        pass
    
    @abstractmethod
    def detect_deception(self, message: List[str], sender: str) -> float:
        """
        Estimate likelihood that a message is deceptive.
        
        Args:
            message: Received message
            sender: Sender agent ID
            
        Returns:
            Deception probability [0, 1]
        """
        pass


class GenerationalAgent(BaseAgent):
    """
    Abstract class for agents in generational transmission experiments.
    
    Used in "Generational Label" game to study cultural transmission
    and conventionalization of language.
    """
    
    @abstractmethod
    def teach(self, student: 'GenerationalAgent', concept: str, label: str) -> float:
        """
        Teach a student the label for a concept.
        
        Args:
            student: The student agent
            concept: The concept to teach (e.g., "Shape-A")
            label: The label to teach (e.g., "tok7")
            
        Returns:
            Learning success rate [0, 1]
        """
        pass
    
    @abstractmethod
    def learn_from(self, teacher: 'GenerationalAgent', concept: str) -> str:
        """
        Learn the label for a concept from a teacher.
        
        Args:
            teacher: The teacher agent
            concept: The concept to learn
            
        Returns:
            The learned label
        """
        pass
    
    @abstractmethod
    def measure_fidelity(self, original_mapping: Dict[str, str]) -> float:
        """
        Measure how faithfully the language was transmitted.
        
        Args:
            original_mapping: Original concept-to-label mapping
            
        Returns:
            Fidelity score [0, 1]
        """
        pass
