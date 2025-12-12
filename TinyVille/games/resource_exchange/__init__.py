"""
Resource Exchange Game

Implements the 4-player, 2-team resource exchange game with timesteps per
round. Uses core Channel/VocabularyFilter/LLMAgent utilities.
"""

"""
Resource Exchange Game package.

Implements the 4-player, 2v2 resource exchange game with chat, exchange,
and feedback phases. Uses timestep-based chat to approximate real-time
messaging while keeping execution deterministic.
"""

from TinyVille.games.resource_exchange.config import ResourceExchangeConfig
from TinyVille.games.resource_exchange.pairing import PairingManager
from TinyVille.games.resource_exchange.vocabulary import AlienVocabularyGenerator
from TinyVille.games.resource_exchange.resources import ResourceManager
from TinyVille.games.resource_exchange.scoring import ScoreCalculator
from TinyVille.games.resource_exchange.agent import ResourceExchangeAgent, create_resource_exchange_action_space
from TinyVille.games.resource_exchange.game import ResourceExchangeGame

__all__ = [
    "ResourceExchangeConfig",
    "PairingManager",
    "AlienVocabularyGenerator",
    "ResourceManager",
    "ScoreCalculator",
    "ResourceExchangeAgent",
    "create_resource_exchange_action_space",
    "ResourceExchangeGame",
]

