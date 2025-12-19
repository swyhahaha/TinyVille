"""
Base Classes for Language Games

Provides abstract base classes that can be extended for various language emergence experiments.

Key Abstractions:
- BaseAgent: Agent with memory, vocabulary, and decision-making
- BaseGame: Game state management and turn execution
- VocabularyManager: Token-meaning mapping and team knowledge
- GameConfig: Configuration for games
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import random
import json
from datetime import datetime

from .protocol import Message, Observation, Action


# =============================================================================
# Game Configuration
# =============================================================================

@dataclass
class TeamConfig:
    """Configuration for a team."""
    team_id: str
    agent_ids: List[str]
    known_token_categories: List[str]  # e.g., ["coordinates", "query"]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameConfig:
    """
    Universal game configuration.
    
    Attributes:
        vocabulary_size: Total number of tokens
        token_categories: Dict mapping category name to list of tokens
        teams: List of team configurations
        max_rounds: Maximum game rounds
        seed: Random seed for reproducibility
    """
    vocabulary_size: int = 30
    token_categories: Dict[str, List[str]] = field(default_factory=dict)
    teams: List[TeamConfig] = field(default_factory=list)
    max_rounds: int = 20
    seed: Optional[int] = None
    
    # Communication settings
    allow_inter_team_communication: bool = True
    allow_intra_team_communication: bool = True
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_agents(self) -> List[str]:
        """Get all agent IDs across all teams."""
        agents = []
        for team in self.teams:
            agents.extend(team.agent_ids)
        return agents
    
    def get_team_for_agent(self, agent_id: str) -> Optional[str]:
        """Get team ID for an agent."""
        for team in self.teams:
            if agent_id in team.agent_ids:
                return team.team_id
        return None


# =============================================================================
# Vocabulary Management
# =============================================================================

class VocabularyManager:
    """
    Manages token-meaning mappings with support for:
    - Random shuffled mappings (prevents inference from token names)
    - Team-specific knowledge
    - Custom/invented meanings ("黑话")
    """
    
    def __init__(self, config: GameConfig = None, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Base vocabulary: token -> category
        self.token_categories: Dict[str, str] = {}
        
        # Ground truth mappings: category -> {value -> token}
        self.mappings: Dict[str, Dict[Any, str]] = {}
        
        # Reverse mappings: category -> {token -> value}
        self.reverse_mappings: Dict[str, Dict[str, Any]] = {}
        
        # Team knowledge: team_id -> {token -> known_meaning}
        self.team_knowledge: Dict[str, Dict[str, str]] = {}
        
        # Custom definitions: token -> [(definer, meaning, team)]
        self.custom_definitions: Dict[str, List[Tuple[str, str, str]]] = {}
        
        # Shared tokens (everyone knows)
        self.shared_tokens: Dict[str, str] = {}
        
        if config:
            self._init_from_config(config)
    
    def _init_from_config(self, config: GameConfig):
        """Initialize from game config."""
        for team in config.teams:
            self.team_knowledge[team.team_id] = {}
    
    def register_category(self, 
                         category: str,
                         tokens: List[str],
                         values: List[Any],
                         shuffle: bool = True) -> Dict[Any, str]:
        """
        Register a category with random mapping.
        
        Args:
            category: Category name (e.g., "x_coord", "value")
            tokens: List of tokens to use
            values: List of values to map to
            shuffle: Whether to randomly shuffle the mapping
            
        Returns:
            Mapping from value to token
        """
        if len(tokens) != len(values):
            raise ValueError(f"tokens and values must have same length")
        
        token_list = list(tokens)
        if shuffle:
            random.shuffle(token_list)
        
        mapping = {v: t for v, t in zip(values, token_list)}
        reverse = {t: v for v, t in mapping.items()}
        
        self.mappings[category] = mapping
        self.reverse_mappings[category] = reverse
        
        for token in tokens:
            self.token_categories[token] = category
        
        return mapping
    
    def register_shared(self, token: str, meaning: str):
        """Register a token that all teams know."""
        self.shared_tokens[token] = meaning
    
    def assign_team_knowledge(self, team_id: str, categories: List[str]):
        """
        Assign knowledge of certain categories to a team.
        
        Args:
            team_id: Team identifier
            categories: List of categories this team knows
        """
        if team_id not in self.team_knowledge:
            self.team_knowledge[team_id] = {}
        
        # Add shared tokens
        self.team_knowledge[team_id].update(self.shared_tokens)
        
        # Add category-specific knowledge
        for category in categories:
            if category in self.reverse_mappings:
                for token, value in self.reverse_mappings[category].items():
                    self.team_knowledge[team_id][token] = str(value)
    
    def define_custom(self, token: str, meaning: str, definer: str, team: str):
        """
        Agent defines a custom meaning for a token (黑话).
        
        Args:
            token: Token being defined
            meaning: Custom meaning
            definer: Agent ID who defined it
            team: Team this definition belongs to
        """
        if token not in self.custom_definitions:
            self.custom_definitions[token] = []
        self.custom_definitions[token].append((definer, meaning, team))
        
        # Add to team knowledge
        if team in self.team_knowledge:
            self.team_knowledge[team][token] = f"[{meaning}]"
    
    def get_agent_vocabulary(self, agent_id: str, team_id: str) -> Dict[str, str]:
        """
        Get the vocabulary an agent knows.
        
        Returns dict of {token: meaning} for tokens this agent understands.
        """
        vocab = {}
        
        # Add shared tokens
        vocab.update(self.shared_tokens)
        
        # Add team knowledge
        if team_id in self.team_knowledge:
            vocab.update(self.team_knowledge[team_id])
        
        # Add custom definitions for this team
        for token, defs in self.custom_definitions.items():
            for definer, meaning, def_team in defs:
                if def_team == team_id:
                    vocab[token] = f"[{meaning}]"
        
        return vocab
    
    def translate(self, tokens: List[str], team_id: str = None) -> str:
        """
        Translate tokens to meanings.
        
        Args:
            tokens: List of tokens to translate
            team_id: If provided, translate from team's perspective
            
        Returns:
            Space-separated meanings
        """
        if team_id:
            vocab = self.get_agent_vocabulary("", team_id)
        else:
            # Ground truth translation
            vocab = {}
            vocab.update(self.shared_tokens)
            for category, reverse in self.reverse_mappings.items():
                for token, value in reverse.items():
                    vocab[token] = str(value)
        
        meanings = []
        for token in tokens:
            if token in vocab:
                meanings.append(vocab[token])
            else:
                meanings.append(f"[?{token}?]")
        
        return " ".join(meanings)
    
    def value_to_token(self, category: str, value: Any) -> Optional[str]:
        """Get token for a value in a category."""
        if category in self.mappings:
            return self.mappings[category].get(value)
        return None
    
    def token_to_value(self, category: str, token: str) -> Optional[Any]:
        """Get value for a token in a category."""
        if category in self.reverse_mappings:
            return self.reverse_mappings[category].get(token)
        return None
    
    def export_mappings(self) -> Dict:
        """Export all mappings for logging/reproducibility."""
        return {
            "mappings": {k: {str(vk): vv for vk, vv in v.items()} 
                        for k, v in self.mappings.items()},
            "shared": self.shared_tokens,
            "custom": {k: [(d, m, t) for d, m, t in v] 
                      for k, v in self.custom_definitions.items()},
            "seed": self.seed,
        }


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for language game agents.
    
    Subclasses must implement:
    - act(): Core decision-making method
    - _build_prompt(): How to construct LLM prompts (if using LLM)
    """
    
    def __init__(self,
                 agent_id: str,
                 team_id: str,
                 vocab_manager: VocabularyManager = None):
        self.agent_id = agent_id
        self.team_id = team_id
        self.vocab_manager = vocab_manager
        
        # Memory
        self.memory: List[str] = []
        self.message_history: List[Message] = []
        self.round_number: int = 0
        
        # Custom definitions this agent has made
        self.custom_definitions: Dict[str, str] = {}
    
    @abstractmethod
    def act(self,
            observations: List[Observation],
            messages: List[Message],
            env_state: Dict[str, Any]) -> Tuple[Optional[Action], Optional[Message]]:
        """
        Core decision-making method.
        
        Args:
            observations: What the agent perceives this round
            messages: Messages received this round
            env_state: Agent's view of the environment
            
        Returns:
            Tuple of (Action to take, Message to send)
            Either can be None.
        """
        pass
    
    def get_vocabulary(self) -> Dict[str, str]:
        """Get tokens this agent knows."""
        if self.vocab_manager:
            return self.vocab_manager.get_agent_vocabulary(self.agent_id, self.team_id)
        return {}
    
    def translate_message(self, tokens: List[str]) -> str:
        """Translate received tokens to meanings this agent understands."""
        vocab = self.get_vocabulary()
        meanings = []
        for token in tokens:
            if token in vocab:
                meanings.append(vocab[token])
            elif token in self.custom_definitions:
                meanings.append(f"[{self.custom_definitions[token]}]")
            else:
                meanings.append(f"[?{token}?]")
        return " ".join(meanings)
    
    def remember(self, content: str):
        """Add something to memory."""
        self.memory.append(f"[R{self.round_number}] {content}")
    
    def get_recent_memory(self, n: int = 5) -> List[str]:
        """Get n most recent memories."""
        return self.memory[-n:] if self.memory else []
    
    def define_token(self, token: str, meaning: str):
        """Define a custom meaning for a token."""
        self.custom_definitions[token] = meaning
        if self.vocab_manager:
            self.vocab_manager.define_custom(token, meaning, self.agent_id, self.team_id)


# =============================================================================
# Base Game
# =============================================================================

class BaseGame(ABC):
    """
    Abstract base class for language games.
    
    Subclasses must implement:
    - get_agent_view(): What each agent can see
    - process_action(): How to handle agent actions
    - check_win_condition(): When the game ends
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.vocab_manager = VocabularyManager(config, seed=config.seed)
        
        # Game state
        self.round_number: int = 0
        self.game_over: bool = False
        self.winner: Optional[str] = None
        
        # History
        self.action_history: List[Tuple[int, str, Action]] = []  # (round, agent_id, action)
        self.message_history: List[Tuple[int, Message]] = []  # (round, message)
        
        # Agents
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the game."""
        self.agents[agent.agent_id] = agent
        agent.vocab_manager = self.vocab_manager
    
    @abstractmethod
    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Get what an agent can observe.
        
        This implements information asymmetry - different agents see different things.
        
        Args:
            agent_id: Agent requesting the view
            
        Returns:
            Dictionary of observable state
        """
        pass
    
    @abstractmethod
    def process_action(self, agent_id: str, action: Action) -> Dict[str, Any]:
        """
        Process an agent's action.
        
        Args:
            agent_id: Agent taking the action
            action: The action to process
            
        Returns:
            Result dictionary
        """
        pass
    
    @abstractmethod
    def check_win_condition(self) -> Tuple[bool, Optional[str]]:
        """
        Check if game has ended.
        
        Returns:
            Tuple of (is_game_over, winner_team_id or None)
        """
        pass
    
    def step(self, agent_order: List[str] = None) -> Dict[str, Any]:
        """
        Execute one round of the game.
        
        Args:
            agent_order: Order in which agents act (default: all agents)
            
        Returns:
            Round summary
        """
        self.round_number += 1
        
        if agent_order is None:
            agent_order = list(self.agents.keys())
        
        round_actions = []
        round_messages = []
        
        for agent_id in agent_order:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            agent.round_number = self.round_number
            
            # Get agent's view
            env_state = self.get_agent_view(agent_id)
            
            # Get pending messages for this agent
            pending_messages = self._get_pending_messages(agent_id)
            
            # Agent decides
            observations = [Observation(agent_id, env_state, "environment")]
            action, message = agent.act(observations, pending_messages, env_state)
            
            # Process action
            if action:
                result = self.process_action(agent_id, action)
                action.result = result
                self.action_history.append((self.round_number, agent_id, action))
                round_actions.append((agent_id, action))
            
            # Handle message
            if message:
                self.message_history.append((self.round_number, message))
                round_messages.append((agent_id, message))
            
            # Check win condition
            game_over, winner = self.check_win_condition()
            if game_over:
                self.game_over = True
                self.winner = winner
                break
        
        return {
            "round": self.round_number,
            "actions": round_actions,
            "messages": round_messages,
            "game_over": self.game_over,
            "winner": self.winner,
        }
    
    def _get_pending_messages(self, agent_id: str) -> List[Message]:
        """Get messages pending for an agent."""
        team_id = self.config.get_team_for_agent(agent_id)
        pending = []
        
        for round_num, msg in self.message_history:
            if round_num != self.round_number:
                continue
            
            # Check if agent should receive this message
            if agent_id in msg.receivers:
                pending.append(msg)
            elif "*" in msg.receivers:
                pending.append(msg)
            elif any(r.startswith("group:") for r in msg.receivers):
                for r in msg.receivers:
                    if r.startswith("group:") and r.split(":")[1] == team_id:
                        pending.append(msg)
                        break
        
        return pending
    
    def run(self, max_rounds: int = None) -> Dict[str, Any]:
        """
        Run the full game.
        
        Args:
            max_rounds: Override config max_rounds
            
        Returns:
            Final game state
        """
        max_rounds = max_rounds or self.config.max_rounds
        
        while not self.game_over and self.round_number < max_rounds:
            self.step()
        
        return {
            "total_rounds": self.round_number,
            "winner": self.winner,
            "action_count": len(self.action_history),
            "message_count": len(self.message_history),
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export full game state for logging."""
        return {
            "config": {
                "vocabulary_size": self.config.vocabulary_size,
                "max_rounds": self.config.max_rounds,
                "seed": self.config.seed,
                "teams": [{"team_id": t.team_id, "agents": t.agent_ids} 
                         for t in self.config.teams],
            },
            "vocab_mappings": self.vocab_manager.export_mappings(),
            "round_number": self.round_number,
            "game_over": self.game_over,
            "winner": self.winner,
            "actions": [(r, a, act.to_dict()) for r, a, act in self.action_history],
            "messages": [(r, m.to_dict()) for r, m in self.message_history],
        }


# =============================================================================
# LLM Agent Base
# =============================================================================

class LLMAgent(BaseAgent):
    """
    Base class for LLM-powered agents.
    
    Provides common functionality for building prompts and parsing responses.
    Subclasses should implement _build_system_prompt and _build_user_prompt.
    """
    
    def __init__(self,
                 agent_id: str,
                 team_id: str,
                 llm_backend,  # LLMBackend
                 vocab_manager: VocabularyManager = None,
                 action_space = None):  # ActionSpace
        super().__init__(agent_id, team_id, vocab_manager)
        self.llm = llm_backend
        self.action_space = action_space
        self.conversation_history: List[Dict[str, str]] = []
    
    def act(self,
            observations: List[Observation],
            messages: List[Message],
            env_state: Dict[str, Any]) -> Tuple[Optional[Action], Optional[Message]]:
        """Make decision using LLM."""
        self.round_number += 1
        
        # Update memory with received messages
        for msg in messages:
            tokens = msg.content if isinstance(msg.content, list) else str(msg.content).split()
            translation = self.translate_message(tokens)
            self.remember(f"{msg.sender}: {tokens} → {translation}")
        
        # Build prompts
        system_prompt = self._build_system_prompt(env_state)
        user_prompt = self._build_user_prompt(messages, env_state)
        
        # Call LLM
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add recent conversation history
        for hist in self.conversation_history[-4:]:
            chat_messages.insert(-1, hist)
        
        response = self.llm.chat(chat_messages)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Parse response and create action/message
        return self._parse_response(response)
    
    @abstractmethod
    def _build_system_prompt(self, env_state: Dict[str, Any]) -> str:
        """Build the system prompt for LLM."""
        pass
    
    @abstractmethod
    def _build_user_prompt(self, messages: List[Message], env_state: Dict[str, Any]) -> str:
        """Build the user prompt for LLM."""
        pass
    
    @abstractmethod
    def _parse_response(self, response: str) -> Tuple[Optional[Action], Optional[Message]]:
        """Parse LLM response into action and message."""
        pass

