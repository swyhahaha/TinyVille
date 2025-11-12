"""
Memory Module for Multi-Agent Language Emergence Experiment

This module implements a memory stream with retrieval mechanisms based on:
- Recency: Exponential decay over time
- Importance: Poignancy scoring (1-10 scale)
- Relevance: Cosine similarity with query embedding
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryObject:
    """
    Represents a single memory in the agent's memory stream.
    
    Attributes:
        description: Natural language description of the memory
        timestamp: When the memory was created
        last_access: When the memory was last retrieved
        importance: Poignancy score (1-10)
        embedding: Vector representation for relevance calculation
        metadata: Additional contextual information
    """
    description: str
    timestamp: datetime
    last_access: datetime
    importance: float  # 1-10 scale
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryStream(ABC):
    """
    Abstract class for managing an agent's memory stream.
    Maintains comprehensive record of agent's experiences.
    """
    
    @abstractmethod
    def add_observation(self, observation: str, metadata: Optional[Dict] = None) -> MemoryObject:
        """
        Add a new observation to the memory stream.
        
        Args:
            observation: Natural language description of the event
            metadata: Additional contextual information
            
        Returns:
            The created MemoryObject
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, 
                 alpha_recency: float = 1.0,
                 alpha_importance: float = 1.0,
                 alpha_relevance: float = 1.0) -> List[MemoryObject]:
        """
        Retrieve relevant memories based on weighted scoring.
        
        Score = α_recency * recency + α_importance * importance + α_relevance * relevance
        
        Args:
            query: The query memory for relevance calculation
            top_k: Number of memories to retrieve
            alpha_recency: Weight for recency score
            alpha_importance: Weight for importance score
            alpha_relevance: Weight for relevance score
            
        Returns:
            List of top-k relevant MemoryObjects
        """
        pass
    
    @abstractmethod
    def calculate_recency_score(self, memory: MemoryObject, current_time: datetime) -> float:
        """
        Calculate recency score using exponential decay.
        
        Args:
            memory: The memory object to score
            current_time: Current simulation time
            
        Returns:
            Normalized recency score [0, 1]
        """
        pass
    
    @abstractmethod
    def calculate_importance_score(self, description: str) -> float:
        """
        Calculate importance score using LLM.
        
        Prompt: "On the scale of 1 to 10, where 1 is purely mundane
        and 10 is extremely poignant, rate the likely poignancy..."
        
        Args:
            description: Natural language description of the memory
            
        Returns:
            Importance score [1, 10]
        """
        pass
    
    @abstractmethod
    def calculate_relevance_score(self, memory: MemoryObject, query_embedding: List[float]) -> float:
        """
        Calculate relevance score using cosine similarity.
        
        Args:
            memory: The memory object to score
            query_embedding: Embedding vector of the query
            
        Returns:
            Normalized relevance score [0, 1]
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using LLM.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def reflect(self) -> List[str]:
        """
        Generate high-level reflections from recent memories.
        Used to synthesize observations into broader insights.
        
        Returns:
            List of reflection statements
        """
        pass


class AbstractSymbolMemory(ABC):
    """
    Abstract class for managing abstract symbol vocabulary.
    Enforces constrained communication channel (tok1, tok2, ..., tokK).
    """
    
    @abstractmethod
    def get_vocabulary(self) -> List[str]:
        """
        Get the constrained abstract vocabulary.
        
        Returns:
            List of abstract tokens (e.g., ['tok1', 'tok2', ..., 'tok20'])
        """
        pass
    
    @abstractmethod
    def validate_message(self, message: List[str], max_length: int = 5) -> bool:
        """
        Validate if a message conforms to vocabulary and length constraints.
        
        Args:
            message: Sequence of tokens
            max_length: Maximum allowed message length
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def encode_meaning(self, concept: str) -> List[str]:
        """
        Encode a concept into abstract symbol sequence.
        
        Args:
            concept: High-level concept to encode
            
        Returns:
            Sequence of abstract tokens
        """
        pass
    
    @abstractmethod
    def decode_meaning(self, tokens: List[str]) -> str:
        """
        Decode abstract symbol sequence into meaning.
        
        Args:
            tokens: Sequence of abstract tokens
            
        Returns:
            Interpreted meaning
        """
        pass
    
    @abstractmethod
    def update_symbol_mapping(self, token: str, concept: str, strength: float = 1.0):
        """
        Update association between token and concept.
        Enables emergent language evolution.
        
        Args:
            token: Abstract token (e.g., 'tok7')
            concept: Associated concept
            strength: Association strength
        """
        pass
