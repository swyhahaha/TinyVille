"""
LLM Backends - Real LLM implementations using api.py utilities.

Provides:
- OpenAIBackend: Uses OpenAI API (via bean_gpt_api)
- VLLMBackend: Uses local vLLM server
- BatchLLMBackend: Wrapper for efficient batch processing
"""

import os
import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from .llm import LLMBackend

# Import API utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.api import api, vllm_api, generate_hash_uid
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    logging.warning("utils.api not available, some backends may not work")


class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend using bean_gpt_api.
    
    Usage:
        backend = OpenAIBackend(
            api_key="your-api-key",
            model="gpt-4o",
            temperature=0.7
        )
        response = backend.chat([{"role": "user", "content": "Hello"}])
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 num_workers: int = 4,
                 cache_dir: str = "./cache/openai",
                 use_cache: bool = True,
                 **kwargs):
        """
        Initialize OpenAI backend.
        
        Args:
            api_key: OpenAI API key (with 'Bearer ' prefix or raw key)
            model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            num_workers: Number of parallel workers for batch requests
            cache_dir: Directory for response caching
            use_cache: Whether to use caching
            **kwargs: Additional inference config (top_p, frequency_penalty, etc.)
        """
        if not API_AVAILABLE:
            raise ImportError("utils.api module required for OpenAIBackend")
        
        # Ensure api_key has proper format
        if not api_key.startswith("Bearer "):
            api_key = f"Bearer {api_key}"
        
        self.api_key = api_key
        self.model = model
        self.config = {
            'model': model,
            'num_workers': num_workers,
            'cache_dir': cache_dir,
            'use_cache': use_cache,
            'api_key': api_key,
            'infer_cfgs': {
                'temperature': temperature,
                'max_tokens': max_tokens,
                **kwargs
            }
        }
        
        self.call_count = 0
        self.total_tokens = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion for a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Single chat completion."""
        # Merge kwargs into config
        config = self._merge_config(kwargs)
        
        # Use batch API with single message
        results = api([messages], config)
        
        self.call_count += 1
        
        if results and len(results) > 0:
            return results[0] if isinstance(results[0], str) else str(results[0])
        return ""
    
    def batch_chat(self, conversations: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Batch chat completions - efficient for multiple requests."""
        config = self._merge_config(kwargs)
        
        results = api(conversations, config)
        
        self.call_count += len(conversations)
        
        return [r if isinstance(r, str) else str(r) for r in results]
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch text generation."""
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        return self.batch_chat(conversations, **kwargs)
    
    def _merge_config(self, kwargs: Dict) -> Dict:
        """Merge runtime kwargs into config."""
        config = self.config.copy()
        infer_cfgs = config['infer_cfgs'].copy()
        
        # Override with runtime kwargs
        for key in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if key in kwargs:
                infer_cfgs[key] = kwargs[key]
        
        config['infer_cfgs'] = infer_cfgs
        return config
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'model': self.model,
        }


class VLLMBackend(LLMBackend):
    """
    vLLM backend for local model serving.
    
    Requires vLLM server running at localhost:8000.
    
    Usage:
        # Start vLLM server first:
        # python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B
        
        backend = VLLMBackend(
            model="meta-llama/Llama-3-8B",
            temperature=0.7
        )
        response = backend.chat([{"role": "user", "content": "Hello"}])
    """
    
    def __init__(self,
                 model: str,
                 api_base: str = "http://localhost:8000/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 top_p: float = 1.0,
                 max_concurrent_requests: int = 10,
                 request_timeout: int = 120,
                 cache_dir: str = "./cache/vllm",
                 use_cache: bool = True,
                 use_async: bool = True,
                 **kwargs):
        """
        Initialize vLLM backend.
        
        Args:
            model: Model name (must match vLLM server's model)
            api_base: vLLM server API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling
            max_concurrent_requests: Max parallel requests
            request_timeout: Request timeout in seconds
            cache_dir: Directory for response caching
            use_cache: Whether to use caching
            use_async: Whether to use async requests (faster)
        """
        if not API_AVAILABLE:
            raise ImportError("utils.api module required for VLLMBackend")
        
        self.model = model
        self.api_base = api_base
        self.config = {
            'model': model,
            'max_concurrent_requests': max_concurrent_requests,
            'request_timeout': request_timeout,
            'cache_dir': cache_dir,
            'use_cache': use_cache,
            'use_async': use_async,
            'infer_cfgs': {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                **kwargs
            }
        }
        
        self.call_count = 0
        self.total_tokens = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion for a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Single chat completion."""
        config = self._merge_config(kwargs)
        
        results = vllm_api([messages], config)
        
        self.call_count += 1
        
        if results and len(results) > 0:
            result = results[0]
            if isinstance(result, dict):
                content = result.get('content', '')
                usage = result.get('usage', {})
                if usage:
                    self.total_tokens += usage.get('total_tokens', 0)
                return content
            return str(result)
        return ""
    
    def batch_chat(self, conversations: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Batch chat completions - very efficient with vLLM."""
        config = self._merge_config(kwargs)
        
        results = vllm_api(conversations, config)
        
        self.call_count += len(conversations)
        
        outputs = []
        for result in results:
            if isinstance(result, dict):
                content = result.get('content', '')
                usage = result.get('usage', {})
                if usage:
                    self.total_tokens += usage.get('total_tokens', 0)
                outputs.append(content)
            else:
                outputs.append(str(result))
        
        return outputs
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch text generation."""
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        return self.batch_chat(conversations, **kwargs)
    
    def _merge_config(self, kwargs: Dict) -> Dict:
        """Merge runtime kwargs into config."""
        config = self.config.copy()
        infer_cfgs = config['infer_cfgs'].copy()
        
        for key in ['temperature', 'max_tokens', 'top_p']:
            if key in kwargs:
                infer_cfgs[key] = kwargs[key]
        
        config['infer_cfgs'] = infer_cfgs
        return config
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'model': self.model,
        }


class BatchLLMWrapper:
    """
    Wrapper that collects LLM calls and executes them in batches.
    
    Useful for scenarios where multiple agents need to make LLM calls
    and batch processing would be more efficient.
    
    Usage:
        batch_llm = BatchLLMWrapper(backend)
        
        # Collect requests
        batch_llm.queue_chat(agent_id="alice", messages=[...])
        batch_llm.queue_chat(agent_id="bob", messages=[...])
        
        # Execute all at once
        results = batch_llm.execute()
        # results = {"alice": "response1", "bob": "response2"}
    """
    
    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.pending_requests: List[Dict[str, Any]] = []
    
    def queue_chat(self, agent_id: str, messages: List[Dict[str, str]], **kwargs):
        """Queue a chat request for batch execution."""
        self.pending_requests.append({
            'agent_id': agent_id,
            'messages': messages,
            'kwargs': kwargs
        })
    
    def queue_generate(self, agent_id: str, prompt: str, **kwargs):
        """Queue a generate request for batch execution."""
        messages = [{"role": "user", "content": prompt}]
        self.queue_chat(agent_id, messages, **kwargs)
    
    def execute(self) -> Dict[str, str]:
        """Execute all queued requests in a batch."""
        if not self.pending_requests:
            return {}
        
        # Extract conversations
        conversations = [req['messages'] for req in self.pending_requests]
        agent_ids = [req['agent_id'] for req in self.pending_requests]
        
        # Batch execute
        responses = self.backend.batch_chat(conversations)
        
        # Map results to agent IDs
        results = {}
        for agent_id, response in zip(agent_ids, responses):
            results[agent_id] = response
        
        # Clear queue
        self.pending_requests = []
        
        return results
    
    def clear(self):
        """Clear pending requests without executing."""
        self.pending_requests = []
    
    def pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self.pending_requests)


# =============================================================================
# Factory function for easy backend creation
# =============================================================================

def create_llm_backend(backend_type: str, **kwargs) -> LLMBackend:
    """
    Factory function to create LLM backends.
    
    Args:
        backend_type: "openai", "vllm", or "dummy"
        **kwargs: Backend-specific configuration
        
    Returns:
        LLMBackend instance
        
    Examples:
        # OpenAI
        llm = create_llm_backend("openai", api_key="sk-...", model="gpt-4o")
        
        # vLLM
        llm = create_llm_backend("vllm", model="meta-llama/Llama-3-8B")
        
        # Dummy for testing
        llm = create_llm_backend("dummy")
    """
    backend_type = backend_type.lower()
    
    if backend_type == "openai":
        return OpenAIBackend(**kwargs)
    elif backend_type == "vllm":
        return VLLMBackend(**kwargs)
    elif backend_type == "dummy":
        from .llm import DummyLLM
        return DummyLLM(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. "
                        f"Supported: openai, vllm, dummy")

