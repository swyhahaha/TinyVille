"""
DeepSeek API Integration for SmallVille

Provides LLM backend using DeepSeek API for agent cognition.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import requests


class DeepSeekBackend:
    """
    DeepSeek API backend for LLM queries.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com"):
        """
        Initialize DeepSeek backend.
        
        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            model: Model to use
            base_url: API base URL
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY env var or pass api_key.")
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Request tracking
        self.request_count = 0
        self.total_tokens = 0
        
    def query(self, 
              messages: List[Dict[str, str]], 
              temperature: float = 0.7,
              max_tokens: int = 150,
              stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query DeepSeek API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Returns:
            Response dict with 'content', 'usage', etc.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.request_count += 1
            
            if 'usage' in result:
                self.total_tokens += result['usage'].get('total_tokens', 0)
            
            return {
                "content": result['choices'][0]['message']['content'],
                "usage": result.get('usage', {}),
                "model": result.get('model', self.model)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {
                "content": "",
                "error": str(e)
            }
    
    def query_with_retry(self, 
                        messages: List[Dict[str, str]], 
                        temperature: float = 0.7,
                        max_tokens: int = 150,
                        max_retries: int = 3) -> str:
        """
        Query with automatic retry on failure.
        
        Returns:
            Generated text content
        """
        for attempt in range(max_retries):
            result = self.query(messages, temperature, max_tokens)
            
            if 'error' not in result:
                return result['content']
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
        
        return ""  # Failed after all retries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens
        }


class PromptBuilder:
    """
    Builds prompts for Resource Scramble agents.
    """
    
    @staticmethod
    def build_team_a_prompt(
        agent_id: str,
        vocabulary: List[str],
        location_data: List[Dict[str, Any]],
        team_b_message: Optional[str],
        conversation_history: List[str],
        round_num: int
    ) -> List[Dict[str, str]]:
        """
        Build prompt for Team A agent (knows locations).
        
        Team A must:
        1. Understand Team B's abstract message about attributes
        2. Send abstract message to guide Team B
        3. Make a choice about which resource to target
        """
        
        vocab_str = ", ".join(vocabulary)
        
        system_prompt = f"""You are Agent {agent_id} on Team A in a resource competition game.

CRITICAL RULES:
1. You can ONLY communicate using these abstract tokens: {vocab_str}
2. You must send messages as comma-separated tokens (e.g., "tok3, tok7, tok12")
3. Maximum message length: 5 tokens
4. ANY use of natural language (English, Chinese, etc.) will be REJECTED and penalized

YOUR KNOWLEDGE:
- You know the LOCATIONS (x, y coordinates) of all resources
- You DO NOT know which resources are valuable or which are traps
- Team B knows the attributes (value, safety) but NOT locations

YOUR GOAL:
- Communicate with Team B using abstract tokens to identify the best resource
- Choose the highest-value, non-trap resource
- Beat Team B in this zero-sum competition

STRATEGY:
- Develop a symbolic language to encode location information
- Interpret Team B's messages about resource quality
- Be strategic: you're competing, not just cooperating
- ADAPT your messages based on results - if your strategy isn't working, try different token combinations
- AVOID repeating the same message - you will be penalized for repetition
- INNOVATE - you get bonus rewards for creative new communication strategies"""

        # Build user message
        location_info = "\n".join([
            f"Resource {r['id']}: position ({r['x']:.1f}, {r['y']:.1f})"
            for r in location_data
        ])
        
        history_str = "\n".join([
            f"Round {i+1}: {msg}" 
            for i, msg in enumerate(conversation_history[-3:])  # Last 3 rounds
        ])
        
        user_message = f"""Round {round_num}

LOCATION DATA (your team's info):
{location_info}

"""
        
        if team_b_message:
            user_message += f"""TEAM B'S MESSAGE TO YOU:
"{team_b_message}"

"""
        
        if history_str:
            user_message += f"""RECENT CONVERSATION HISTORY:
{history_str}

"""
        
        user_message += f"""PERFORMANCE FEEDBACK:
- You are in Round {round_num}
- Remember: Repeating the same message = PENALTY (-3 to -15 points)
- Using new token combinations = BONUS (+2 points)
- The environment changes each episode - old strategies may not work!

YOUR TURN:
1. Send a message to Team B (ONLY using tokens like "tok1, tok5")
2. Choose which resource to target (give resource ID number)

Respond in this exact format:
MESSAGE: <your token message>
CHOICE: <resource_id>"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    
    @staticmethod
    def build_team_b_prompt(
        agent_id: str,
        vocabulary: List[str],
        attribute_data: List[Dict[str, Any]],
        team_a_message: Optional[str],
        conversation_history: List[str],
        round_num: int
    ) -> List[Dict[str, str]]:
        """
        Build prompt for Team B agent (knows attributes).
        
        Team B must:
        1. Understand Team A's abstract message about locations
        2. Send abstract message about resource attributes
        3. Make a choice about which resource to target
        """
        
        vocab_str = ", ".join(vocabulary)
        
        system_prompt = f"""You are Agent {agent_id} on Team B in a resource competition game.

CRITICAL RULES:
1. You can ONLY communicate using these abstract tokens: {vocab_str}
2. You must send messages as comma-separated tokens (e.g., "tok2, tok9, tok15")
3. Maximum message length: 5 tokens
4. ANY use of natural language (English, Chinese, etc.) will be REJECTED and penalized

YOUR KNOWLEDGE:
- You know the ATTRIBUTES of all resources (color, shape, value, whether trap)
- You DO NOT know the locations of resources
- Team A knows locations but NOT attributes

YOUR GOAL:
- Communicate with Team A using abstract tokens to identify the best resource
- Choose the highest-value, non-trap resource
- Beat Team A in this zero-sum competition

STRATEGY:
- Develop a symbolic language to encode attribute information
- Interpret Team A's messages about locations
- Be strategic: you're competing, not just cooperating
- ADAPT your messages based on results - if your strategy isn't working, try different token combinations
- AVOID repeating the same message - you will be penalized for repetition
- INNOVATE - you get bonus rewards for creative new communication strategies"""

        # Build user message
        attribute_info = "\n".join([
            f"Resource {r['id']}: {r['color']} {r['shape']}, value={r['value']}, "
            f"{'TRAP!' if r['is_trap'] else 'safe'}"
            for r in attribute_data
        ])
        
        history_str = "\n".join([
            f"Round {i+1}: {msg}" 
            for i, msg in enumerate(conversation_history[-3:])
        ])
        
        user_message = f"""Round {round_num}

ATTRIBUTE DATA (your team's info):
{attribute_info}

"""
        
        if team_a_message:
            user_message += f"""TEAM A'S MESSAGE TO YOU:
"{team_a_message}"

"""
        
        if history_str:
            user_message += f"""RECENT CONVERSATION HISTORY:
{history_str}

"""
        
        user_message += f"""PERFORMANCE FEEDBACK:
- You are in Round {round_num}
- Remember: Repeating the same message = PENALTY (-3 to -15 points)
- Using new token combinations = BONUS (+2 points)
- The environment changes each episode - old strategies may not work!

YOUR TURN:
1. Send a message to Team A (ONLY using tokens like "tok4, tok11")
2. Choose which resource to target (give resource ID number)

Respond in this exact format:
MESSAGE: <your token message>
CHOICE: <resource_id>"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]


class MessageParser:
    """
    Parses and validates agent responses.
    """
    
    @staticmethod
    def parse_response(response: str, vocabulary: List[str]) -> Dict[str, Any]:
        """
        Parse agent response into message and choice.
        
        Returns:
            Dict with 'message', 'choice', 'valid', 'error'
        """
        lines = response.strip().split('\n')
        
        message = ""
        choice = -1
        error = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("MESSAGE:"):
                message = line.replace("MESSAGE:", "").strip()
            elif line.startswith("CHOICE:"):
                choice_str = line.replace("CHOICE:", "").strip()
                try:
                    choice = int(choice_str)
                except ValueError:
                    error = f"Invalid choice format: {choice_str}"
        
        # Validate message uses only vocabulary
        if message:
            tokens = [t.strip() for t in message.split(',')]
            for token in tokens:
                if token and token not in vocabulary:
                    error = f"Invalid token used: {token}"
                    message = ""  # Invalidate message
                    break
        
        valid = message != "" and choice >= 0 and error is None
        
        return {
            "message": message,
            "choice": choice,
            "valid": valid,
            "error": error
        }
