"""
Protocol Module - Core data structures for agent communication and logging.

Defines:
- Message, Observation, Action: Fundamental units of interaction
- LogEntry, GameLogger: Structured logging with token translations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import time
import uuid
import json


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class Message:
    """
    Universal message structure for agent communication.
    
    Attributes:
        sender: Agent ID of the sender
        receivers: List of recipient agent IDs
                   - ["agent_1"] for unicast
                   - ["agent_1", "agent_2"] for multicast
                   - ["*"] for broadcast
                   - ["group:team_a"] for group message
        content: Filtered message content (after passing through MessageFilter)
        raw_content: Original LLM output (for debugging/analysis)
        channel: Communication channel name
        id: Unique message identifier
        timestamp: Unix timestamp when message was created
        metadata: Additional context
        reply_to: Optional ID of message being replied to
    """
    sender: str
    receivers: List[str]
    content: Any
    raw_content: Any = None
    channel: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None
    
    def is_broadcast(self) -> bool:
        return "*" in self.receivers
    
    def is_group_message(self) -> bool:
        return any(r.startswith("group:") for r in self.receivers)
    
    def get_target_groups(self) -> List[str]:
        return [r.split(":", 1)[1] for r in self.receivers if r.startswith("group:")]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receivers": self.receivers,
            "content": self.content,
            "raw_content": self.raw_content,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "reply_to": self.reply_to,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**data)


@dataclass
class Observation:
    """
    Observation received by an agent.
    
    Attributes:
        agent_id: The agent receiving this observation
        content: Observation content (can be any type)
        source: Where the observation came from
        timestamp: When the observation was created
        metadata: Additional context
    """
    agent_id: str
    content: Any
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(**data)


@dataclass  
class Action:
    """
    Action taken by an agent.
    
    Attributes:
        agent_id: The agent taking this action
        action_type: Type of action (e.g., "communicate", "claim")
        target: Optional target of the action
        params: Action parameters
        timestamp: When the action was taken
        result: Result after execution
    """
    agent_id: str
    action_type: str
    target: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "target": self.target,
            "params": self.params,
            "timestamp": self.timestamp,
            "result": self.result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(**data)


# =============================================================================
# Logging
# =============================================================================

@dataclass
class LogEntry:
    """A single log entry with token translation."""
    timestamp: str
    round: int
    agent_id: str
    team: str
    action_type: str
    raw_content: Any
    translated_content: str
    target: Optional[str] = None
    result: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "round": self.round,
            "agent": self.agent_id,
            "team": self.team,
            "action": self.action_type,
            "raw": self.raw_content,
            "translated": self.translated_content,
            "target": self.target,
            "result": self.result,
            "metadata": self.metadata,
        }


class GameLogger:
    """
    Game logger with token translation support.
    
    Usage:
        logger = GameLogger(translate_fn=lambda tokens: " ".join(TOKEN_MEANINGS.get(t, t) for t in tokens))
        logger.log_action(round_num=1, agent_id="alice", team="team_a", ...)
    """
    
    def __init__(self, translate_fn: Callable[[List[str]], str] = None):
        """
        Args:
            translate_fn: Function to translate token list to readable string.
                         If None, tokens are joined with spaces.
        """
        self.translate_fn = translate_fn or (lambda tokens: " ".join(tokens))
        self.entries: List[LogEntry] = []
        self.custom_definitions: List[Dict] = []
    
    def log_action(self,
                   round_num: int,
                   agent_id: str,
                   team: str,
                   action_type: str,
                   content: Any,
                   target: str = None,
                   result: Dict = None,
                   metadata: Dict = None):
        """Log an action with translation."""
        # Parse content as tokens
        if isinstance(content, list):
            tokens = content
        elif isinstance(content, str):
            tokens = content.split()
        else:
            tokens = []
        
        # Translate
        translated = self.translate_fn(tokens) if tokens else str(content)
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            round=round_num,
            agent_id=agent_id,
            team=team,
            action_type=action_type,
            raw_content=tokens,
            translated_content=translated,
            target=target,
            result=result,
            metadata=metadata or {},
        )
        
        self.entries.append(entry)
        return entry
    
    def log_custom_definition(self,
                              round_num: int,
                              agent_id: str,
                              team: str,
                              token: str,
                              meaning: str):
        """Log when an agent invents a new token meaning."""
        self.custom_definitions.append({
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "agent_id": agent_id,
            "team": team,
            "token": token,
            "meaning": meaning,
        })
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            round=round_num,
            agent_id=agent_id,
            team=team,
            action_type="DEFINE_TOKEN",
            raw_content=[token],
            translated_content=f"defines {token} = '{meaning}'",
            metadata={"new_meaning": meaning}
        )
        self.entries.append(entry)
        return entry
    
    @property
    def definition_history(self) -> List[Dict]:
        """Get all custom token definitions."""
        return self.custom_definitions
    
    def print_transcript(self, show_raw: bool = True, show_translation: bool = True):
        """Print human-readable transcript."""
        print("\n" + "=" * 70)
        print("GAME TRANSCRIPT")
        print("=" * 70)
        
        current_round = -1
        for entry in self.entries:
            if entry.round != current_round:
                current_round = entry.round
                print(f"\n--- Round {current_round} ---")
            
            team_icon = "🔵" if entry.team == "team_a" else "🔴"
            
            if entry.action_type == "DEFINE_TOKEN":
                print(f"  {team_icon} {entry.agent_id} 📝 {entry.translated_content}")
            elif entry.action_type in ["talk_to_teammate", "talk_to_opponent", "broadcast"]:
                channel = "🏠" if "teammate" in entry.action_type else "🌐"
                print(f"  {team_icon} {entry.agent_id} {channel} → {entry.target}")
                if show_raw:
                    print(f"      Tokens: {entry.raw_content}")
                if show_translation:
                    print(f"      Meaning: {entry.translated_content}")
            elif entry.action_type == "claim_resource":
                result_icon = "✅" if entry.result and entry.result.get("success") else "❌"
                print(f"  {team_icon} {entry.agent_id} {result_icon} CLAIM {entry.target}")
                if entry.result:
                    print(f"      Result: {entry.result}")
    
    def export_json(self) -> str:
        """Export logs as JSON."""
        return json.dumps({
            "entries": [e.to_dict() for e in self.entries],
            "custom_definitions": self.custom_definitions,
        }, indent=2, ensure_ascii=False)
