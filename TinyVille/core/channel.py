"""
Channel Module - Communication channel and message filtering.

Provides:
- MessageFilter: Abstract class for constraining LLM outputs
- Channel: Message routing and delivery management
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import re

from .protocol import Message


class MessageFilter(ABC):
    """
    Abstract base class for filtering/transforming LLM raw outputs.
    
    Filters enforce communication constraints (e.g., vocabulary limits,
    format requirements) before messages are transmitted through the channel.
    """
    
    @abstractmethod
    def filter(self, raw_output: str, context: Dict[str, Any] = None) -> Tuple[Any, bool]:
        """
        Transform raw LLM output into valid channel content.
        
        Args:
            raw_output: Raw string output from LLM
            context: Optional context dict containing:
                     - sender: Agent ID
                     - receivers: Target agent IDs
                     - channel: Channel name
                     - mode: Communication mode (e.g., "intra_team", "inter_team")
                     
        Returns:
            Tuple of (filtered_content, is_valid)
            - filtered_content: Transformed content (can be any type)
            - is_valid: Whether the output could be successfully filtered
        """
        pass
    
    def get_constraint_description(self) -> str:
        """
        Return a natural language description of this filter's constraints.
        Useful for including in agent prompts.
        """
        return "No specific constraints."


class PassthroughFilter(MessageFilter):
    """Filter that passes through raw output unchanged."""
    
    def filter(self, raw_output: str, context: Dict[str, Any] = None) -> Tuple[str, bool]:
        return raw_output, True
    
    def get_constraint_description(self) -> str:
        return "You may communicate freely without format restrictions."


class VocabularyFilter(MessageFilter):
    """
    Filter that restricts output to a predefined vocabulary.
    
    Used for abstract symbol communication experiments where agents
    can only use tokens like [tok1, tok2, ..., tokK].
    
    IMPORTANT: This filter enforces that ONLY vocabulary tokens pass through.
    Any natural language (English, etc.) will be stripped out.
    """
    
    def __init__(self, vocabulary: List[str], max_length: int = 5, 
                 separator: str = " ", strict: bool = False,
                 extract_mode: str = "anywhere"):
        """
        Args:
            vocabulary: List of allowed tokens
            max_length: Maximum number of tokens in a message
            separator: Token separator in raw output
            strict: If True, reject messages with any invalid tokens
                   If False, filter out invalid tokens and keep valid ones
            extract_mode: How to extract tokens from output
                   - "anywhere": Find tokens anywhere in the text
                   - "line": Only from lines that look like token sequences
        """
        self.vocabulary = set(vocabulary)
        self.max_length = max_length
        self.separator = separator
        self.strict = strict
        self.extract_mode = extract_mode
    
    def filter(self, raw_output: str, context: Dict[str, Any] = None) -> Tuple[List[str], bool]:
        """
        Extract ONLY valid vocabulary tokens from output.
        
        This is strict: any English words, punctuation, etc. are discarded.
        Only exact matches to vocabulary tokens are kept.
        """
        if self.extract_mode == "anywhere":
            tokens = self._extract_anywhere(raw_output)
        else:
            tokens = self._extract_line(raw_output)
        
        if self.strict:
            # In strict mode, if we found fewer tokens than words, reject
            words = raw_output.strip().split()
            if len(tokens) < len([w for w in words if w.strip()]):
                return [], False
            valid_tokens = tokens[:self.max_length]
        else:
            valid_tokens = tokens[:self.max_length]
        
        is_valid = len(valid_tokens) > 0
        return valid_tokens, is_valid
    
    def _extract_anywhere(self, text: str) -> List[str]:
        """Extract valid tokens from anywhere in the text."""
        tokens = []
        # Split by common separators
        words = re.split(r'[\s,;:\[\]\(\)\{\}\"\']+', text)
        for word in words:
            word = word.strip()
            if word in self.vocabulary:
                tokens.append(word)
        return tokens
    
    def _extract_line(self, text: str) -> List[str]:
        """Extract tokens from lines that look like token sequences."""
        tokens = []
        for line in text.split('\n'):
            line = line.strip()
            # Skip lines that look like explanations
            if any(line.startswith(p) for p in ['#', '//', 'Note:', 'Reasoning:']):
                continue
            # Try to extract tokens from this line
            words = line.split()
            line_tokens = [w for w in words if w in self.vocabulary]
            if line_tokens:
                tokens.extend(line_tokens)
        return tokens
    
    def get_constraint_description(self) -> str:
        vocab_str = ", ".join(sorted(self.vocabulary)[:10])
        if len(self.vocabulary) > 10:
            vocab_str += f", ... ({len(self.vocabulary)} tokens total)"
        return (f"You must communicate using ONLY these tokens: [{vocab_str}]. "
                f"Maximum {self.max_length} tokens per message, separated by spaces. "
                f"NO English or other natural language - ONLY tokens!")


class JSONFilter(MessageFilter):
    """
    Filter that expects JSON-formatted output.
    """
    
    def __init__(self, required_fields: List[str] = None):
        """
        Args:
            required_fields: List of field names that must be present in JSON
        """
        self.required_fields = required_fields or []
    
    def filter(self, raw_output: str, context: Dict[str, Any] = None) -> Tuple[Dict, bool]:
        import json
        
        # Try to extract JSON from the output
        try:
            # First try direct parsing
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to find JSON in the output
            match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return {}, False
            else:
                return {}, False
        
        # Check required fields
        if self.required_fields:
            if not all(f in data for f in self.required_fields):
                return data, False
        
        return data, True
    
    def get_constraint_description(self) -> str:
        if self.required_fields:
            return f"Respond with valid JSON containing these fields: {self.required_fields}"
        return "Respond with valid JSON."


class Channel:
    """
    Communication channel managing message routing and delivery.
    
    Features:
    - Unicast, multicast, and broadcast messaging
    - Group-based communication
    - Message filtering
    - History tracking
    """
    
    def __init__(self, default_filter: MessageFilter = None):
        """
        Args:
            default_filter: Default filter applied to all messages.
                           If None, PassthroughFilter is used.
        """
        self.default_filter = default_filter or PassthroughFilter()
        self.message_queue: Dict[str, List[Message]] = defaultdict(list)
        self.groups: Dict[str, Set[str]] = {}  # group_id -> set of agent_ids
        self.agent_groups: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> set of group_ids
        self.history: List[Message] = []
        self.filters: Dict[str, MessageFilter] = {}  # channel_name -> filter
    
    def set_filter(self, channel_name: str, filter: MessageFilter):
        """Set a specific filter for a channel."""
        self.filters[channel_name] = filter
    
    def get_filter(self, channel_name: str) -> MessageFilter:
        """Get the filter for a channel."""
        return self.filters.get(channel_name, self.default_filter)
    
    def create_group(self, group_id: str, members: List[str]):
        """
        Create a communication group.
        
        Args:
            group_id: Unique group identifier
            members: List of agent IDs in the group
        """
        self.groups[group_id] = set(members)
        for member in members:
            self.agent_groups[member].add(group_id)
    
    def add_to_group(self, group_id: str, agent_id: str):
        """Add an agent to a group."""
        if group_id not in self.groups:
            self.groups[group_id] = set()
        self.groups[group_id].add(agent_id)
        self.agent_groups[agent_id].add(group_id)
    
    def remove_from_group(self, group_id: str, agent_id: str):
        """Remove an agent from a group."""
        if group_id in self.groups:
            self.groups[group_id].discard(agent_id)
        self.agent_groups[agent_id].discard(group_id)
    
    def get_group_members(self, group_id: str) -> Set[str]:
        """Get all members of a group."""
        return self.groups.get(group_id, set()).copy()
    
    def send(self, message: Message, apply_filter: bool = True) -> bool:
        """
        Send a message through the channel.
        
        Args:
            message: Message to send
            apply_filter: Whether to apply channel filter
            
        Returns:
            True if message was successfully sent
        """
        # Apply filter if needed
        if apply_filter and message.raw_content is None:
            message.raw_content = message.content
            filter = self.get_filter(message.channel)
            context = {
                "sender": message.sender,
                "receivers": message.receivers,
                "channel": message.channel,
                "metadata": message.metadata,
            }
            filtered_content, is_valid = filter.filter(str(message.content), context)
            if not is_valid:
                return False
            message.content = filtered_content
        
        # Resolve receivers
        actual_receivers = self._resolve_receivers(message.receivers, message.sender)
        
        # Deliver to each receiver
        for receiver in actual_receivers:
            self.message_queue[receiver].append(message)
        
        # Record in history
        self.history.append(message)
        
        return True
    
    def _resolve_receivers(self, receivers: List[str], sender: str) -> Set[str]:
        """Resolve receiver specifications to actual agent IDs."""
        actual = set()
        
        for r in receivers:
            if r == "*":
                # Broadcast to all known agents (from groups)
                for group_members in self.groups.values():
                    actual.update(group_members)
                actual.update(self.message_queue.keys())
            elif r.startswith("group:"):
                group_id = r.split(":", 1)[1]
                actual.update(self.groups.get(group_id, set()))
            else:
                actual.add(r)
        
        # Don't send to self
        actual.discard(sender)
        
        return actual
    
    def receive(self, agent_id: str, clear: bool = True) -> List[Message]:
        """
        Receive all pending messages for an agent.
        
        Args:
            agent_id: Agent requesting messages
            clear: Whether to clear messages after receiving
            
        Returns:
            List of messages
        """
        messages = self.message_queue[agent_id].copy()
        if clear:
            self.message_queue[agent_id] = []
        return messages
    
    def peek(self, agent_id: str) -> List[Message]:
        """Peek at pending messages without clearing."""
        return self.receive(agent_id, clear=False)
    
    def broadcast(self, sender: str, content: Any, channel: str = "default",
                  exclude: List[str] = None, metadata: Dict = None) -> Message:
        """
        Convenience method for broadcasting a message.
        
        Args:
            sender: Sender agent ID
            content: Message content
            channel: Channel name
            exclude: Agent IDs to exclude from broadcast
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        receivers = ["*"]
        if exclude:
            # We'll handle exclusion in _resolve_receivers or here
            pass
        
        message = Message(
            sender=sender,
            receivers=receivers,
            content=content,
            channel=channel,
            metadata=metadata or {},
        )
        self.send(message)
        return message
    
    def send_to_group(self, sender: str, group_id: str, content: Any,
                      channel: str = "default", metadata: Dict = None) -> Message:
        """
        Send a message to a group.
        
        Args:
            sender: Sender agent ID
            group_id: Target group ID
            content: Message content
            channel: Channel name
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        message = Message(
            sender=sender,
            receivers=[f"group:{group_id}"],
            content=content,
            channel=channel,
            metadata=metadata or {},
        )
        self.send(message)
        return message
    
    def get_history(self, 
                    filter_fn: Callable[[Message], bool] = None,
                    sender: str = None,
                    channel: str = None,
                    limit: int = None) -> List[Message]:
        """
        Get message history with optional filtering.
        
        Args:
            filter_fn: Custom filter function
            sender: Filter by sender
            channel: Filter by channel
            limit: Maximum number of messages to return
            
        Returns:
            List of messages matching criteria
        """
        result = self.history
        
        if sender:
            result = [m for m in result if m.sender == sender]
        if channel:
            result = [m for m in result if m.channel == channel]
        if filter_fn:
            result = [m for m in result if filter_fn(m)]
        if limit:
            result = result[-limit:]
        
        return result
    
    def clear_history(self):
        """Clear message history."""
        self.history = []
    
    def clear_queues(self):
        """Clear all message queues."""
        self.message_queue = defaultdict(list)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            "total_messages": len(self.history),
            "pending_messages": sum(len(q) for q in self.message_queue.values()),
            "num_groups": len(self.groups),
            "messages_by_channel": self._count_by_channel(),
            "messages_by_sender": self._count_by_sender(),
        }
    
    def _count_by_channel(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for m in self.history:
            counts[m.channel] += 1
        return dict(counts)
    
    def _count_by_sender(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for m in self.history:
            counts[m.sender] += 1
        return dict(counts)

