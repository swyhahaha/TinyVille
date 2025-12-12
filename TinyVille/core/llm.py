"""
LLM Backend Module - Abstract interface for LLM integration.

Users should implement their own backend by subclassing LLMBackend.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    Subclass this to integrate with different LLM providers:
    - OpenAI API
    - Anthropic API
    - vLLM server
    - Local models (transformers, llama.cpp, etc.)
    
    Example implementation:
    
        class OpenAIBackend(LLMBackend):
            def __init__(self, model="gpt-4", api_key=None):
                self.client = OpenAI(api_key=api_key)
                self.model = model
            
            def generate(self, prompt, **kwargs):
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    **kwargs
                )
                return response.choices[0].text
            
            def chat(self, messages, **kwargs):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt string
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text string
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     e.g., [{"role": "system", "content": "You are..."},
                            {"role": "user", "content": "Hello"}]
            **kwargs: Additional parameters
            
        Returns:
            Assistant's response string
        """
        pass
    
    def embed(self, text: str) -> List[float]:
        """
        Generate text embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            NotImplementedError if not supported by backend
        """
        raise NotImplementedError("Embedding not supported by this backend")
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate completions for multiple prompts.
        
        Default implementation runs sequentially.
        Override for parallel/batched execution.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters
            
        Returns:
            List of generated texts
        """
        return [self.generate(p, **kwargs) for p in prompts]
    
    def batch_chat(self, conversations: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generate chat completions for multiple conversations.
        
        Default implementation runs sequentially.
        Override for parallel/batched execution.
        
        Args:
            conversations: List of message lists
            **kwargs: Additional parameters
            
        Returns:
            List of assistant responses
        """
        return [self.chat(conv, **kwargs) for conv in conversations]


class DummyLLM(LLMBackend):
    """
    Dummy LLM backend for testing.
    
    Returns predefined responses or echoes input.
    """
    
    def __init__(self, default_response: str = "I acknowledge your message."):
        self.default_response = default_response
        self.responses: Dict[str, str] = {}  # prompt -> response mapping
        self.call_history: List[Dict] = []
    
    def set_response(self, prompt_contains: str, response: str):
        """Set a specific response for prompts containing a string."""
        self.responses[prompt_contains] = response
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.call_history.append({"type": "generate", "prompt": prompt, "kwargs": kwargs})
        
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return self.default_response
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        self.call_history.append({"type": "chat", "messages": messages, "kwargs": kwargs})
        
        # Check last user message
        last_content = messages[-1].get("content", "") if messages else ""
        for key, response in self.responses.items():
            if key in last_content:
                return response
        return self.default_response
    
    def get_call_history(self) -> List[Dict]:
        """Get history of all LLM calls."""
        return self.call_history.copy()
    
    def clear_history(self):
        """Clear call history."""
        self.call_history = []

