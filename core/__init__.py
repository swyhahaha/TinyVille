"""
TinyVille Core Module

Lightweight multi-agent interaction framework for language emergence experiments.

Core Abstractions:
- BaseAgent, LLMAgent: Agent base classes
- BaseGame: Game state management
- VocabularyManager: Token-meaning mappings
- GameConfig: Configuration

Communication:
- Message, Observation, Action: Data structures
- Channel: Message routing and filtering
"""

# Protocol & Data Structures
from .protocol import Message, Observation, Action, LogEntry, GameLogger

# Channel & Communication
from .channel import Channel, MessageFilter, VocabularyFilter, PassthroughFilter, JSONFilter

# LLM Backends
from .llm import LLMBackend, DummyLLM
from .llm_backends import (
    OpenAIBackend,
    VLLMBackend,
    BatchLLMWrapper,
    create_llm_backend,
)

# Action Space
from .action_space import (
    ActionDef,
    ActionSpace,
    Parameter,
    FunctionCall,
    ActionParser,
    create_language_game_actions,
    create_negotiation_actions,
)

# Base Classes
from .base import (
    GameConfig,
    TeamConfig,
    VocabularyManager,
    BaseAgent,
    BaseGame,
    LLMAgent,
)

__all__ = [
    # Protocol
    "Message",
    "Observation", 
    "Action",
    "LogEntry",
    "GameLogger",
    
    # Channel
    "Channel",
    "MessageFilter",
    "VocabularyFilter",
    "PassthroughFilter",
    "JSONFilter",
    
    # LLM
    "LLMBackend",
    "DummyLLM",
    "OpenAIBackend",
    "VLLMBackend",
    "BatchLLMWrapper",
    "create_llm_backend",
    
    # Action Space
    "ActionDef",
    "ActionSpace",
    "Parameter",
    "FunctionCall",
    "ActionParser",
    "create_language_game_actions",
    "create_negotiation_actions",
    
    # Base Classes
    "GameConfig",
    "TeamConfig",
    "VocabularyManager",
    "BaseAgent",
    "BaseGame",
    "LLMAgent",
]
