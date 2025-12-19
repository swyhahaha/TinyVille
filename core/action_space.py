"""
Action Space Module - Define task-specific action spaces with function calling support.

Provides:
- ActionDef: Define individual actions with parameters
- ActionSpace: Collection of available actions
- FunctionCallingTemplate: Generate function-calling prompts
- ActionParser: Parse LLM responses into action calls
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import json
import re


@dataclass
class Parameter:
    """Definition of an action parameter."""
    name: str
    type: str  # "string", "int", "float", "bool", "enum", "list"
    description: str
    required: bool = True
    default: Any = None
    enum_values: List[str] = field(default_factory=list)  # For enum type
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format."""
        schema = {
            "type": self.type if self.type != "enum" else "string",
            "description": self.description,
        }
        if self.enum_values:
            schema["enum"] = self.enum_values
        return schema


@dataclass
class ActionDef:
    """
    Definition of a single action.
    
    Example:
        ActionDef(
            name="send_message",
            description="Send a message to another agent or group",
            parameters=[
                Parameter("target", "enum", "Who to send to", enum_values=["teammate", "opponent", "broadcast"]),
                Parameter("content", "string", "Message content (tokens separated by space)")
            ]
        )
    """
    name: str
    description: str
    parameters: List[Parameter] = field(default_factory=list)
    handler: Optional[Callable] = None  # Optional: function to execute this action
    
    def to_function_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def to_prompt_description(self) -> str:
        """Convert to human-readable description for prompt."""
        lines = [f"### {self.name}"]
        lines.append(f"Description: {self.description}")
        
        if self.parameters:
            lines.append("Parameters:")
            for param in self.parameters:
                req = "(required)" if param.required else "(optional)"
                enum_hint = f" Options: {param.enum_values}" if param.enum_values else ""
                lines.append(f"  - {param.name} ({param.type}) {req}: {param.description}{enum_hint}")
        
        return "\n".join(lines)


class ActionSpace:
    """
    Collection of available actions for a task.
    
    Usage:
        space = ActionSpace()
        space.add(ActionDef("stay_silent", "Do nothing this turn"))
        space.add(ActionDef("send_message", "Send a message", [...]))
        
        # Get all actions as prompt
        prompt = space.to_prompt()
        
        # Validate an action call
        is_valid = space.validate("send_message", {"target": "teammate"})
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.actions: Dict[str, ActionDef] = {}
    
    def add(self, action: ActionDef) -> "ActionSpace":
        """Add an action to the space."""
        self.actions[action.name] = action
        return self
    
    def remove(self, action_name: str) -> "ActionSpace":
        """Remove an action from the space."""
        self.actions.pop(action_name, None)
        return self
    
    def get(self, action_name: str) -> Optional[ActionDef]:
        """Get an action by name."""
        return self.actions.get(action_name)
    
    def list_actions(self) -> List[str]:
        """List all action names."""
        return list(self.actions.keys())
    
    def validate(self, action_name: str, params: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate an action call.
        
        Returns:
            (is_valid, error_message)
        """
        if action_name not in self.actions:
            return False, f"Unknown action: {action_name}"
        
        action = self.actions[action_name]
        params = params or {}
        
        # Check required parameters
        for param in action.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
            
            if param.name in params and param.enum_values:
                if params[param.name] not in param.enum_values:
                    return False, f"Invalid value for {param.name}: {params[param.name]}. Must be one of {param.enum_values}"
        
        return True, None
    
    def to_prompt(self, format: str = "simple") -> str:
        """
        Generate prompt describing all actions.
        
        Args:
            format: "simple" for human-readable, "json" for JSON schema
        """
        if format == "json":
            schemas = [a.to_function_schema() for a in self.actions.values()]
            return json.dumps(schemas, indent=2)
        else:
            parts = ["## Available Actions\n"]
            for action in self.actions.values():
                parts.append(action.to_prompt_description())
                parts.append("")
            return "\n".join(parts)
    
    def to_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-style function schemas."""
        return [a.to_function_schema() for a in self.actions.values()]


@dataclass
class FunctionCall:
    """Represents a parsed function/action call."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


class ActionParser:
    """
    Parse LLM responses into action calls.
    
    Supports multiple formats:
    - Simple format: "ACTION: action_name(param1=value1, param2=value2)"
    - JSON format: {"action": "action_name", "params": {...}}
    - Function call format (OpenAI style)
    """
    
    @staticmethod
    def parse(response: str, action_space: ActionSpace = None) -> Optional[FunctionCall]:
        """
        Parse response into a FunctionCall.
        
        Tries multiple formats in order:
        1. JSON format
        2. Simple ACTION: format
        3. Function-like format
        """
        response = response.strip()
        
        # Try JSON format
        result = ActionParser._try_json(response)
        if result:
            return result
        
        # Try simple format
        result = ActionParser._try_simple(response)
        if result:
            return result
        
        # Try function-like format
        result = ActionParser._try_function_like(response)
        if result:
            return result
        
        return None
    
    @staticmethod
    def _try_json(response: str) -> Optional[FunctionCall]:
        """Try to parse JSON format."""
        try:
            # First try direct parsing
            data = json.loads(response.strip())
            return FunctionCall(
                name=data.get("action", data.get("name", "")),
                arguments=data.get("params", data.get("arguments", {})),
                raw_response=response
            )
        except json.JSONDecodeError:
            pass
        
        try:
            # Try to find JSON object in response (handles nested objects)
            # Find first { and matching }
            start = response.find('{')
            if start == -1:
                return None
            
            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            json_str = response[start:end]
            data = json.loads(json_str)
            return FunctionCall(
                name=data.get("action", data.get("name", "")),
                arguments=data.get("params", data.get("arguments", {})),
                raw_response=response
            )
        except:
            pass
        return None
    
    @staticmethod
    def _try_simple(response: str) -> Optional[FunctionCall]:
        """
        Try to parse simple format:
        ACTION: action_name
        TARGET: value (optional)
        MESSAGE: content (optional)
        """
        lines = response.strip().split("\n")
        action_name = None
        arguments = {}
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                parts = line[7:].strip().split()
                if parts:
                    action_name = parts[0].lower()
                    if len(parts) > 1:
                        arguments["target"] = " ".join(parts[1:])
            elif line.upper().startswith("TARGET:"):
                arguments["target"] = line[7:].strip()
            elif line.upper().startswith("MESSAGE:"):
                msg = line[8:].strip()
                if msg.upper() != "NONE":
                    arguments["content"] = msg
            elif line.upper().startswith("CONTENT:"):
                arguments["content"] = line[8:].strip()
        
        if action_name:
            return FunctionCall(name=action_name, arguments=arguments, raw_response=response)
        return None
    
    @staticmethod
    def _try_function_like(response: str) -> Optional[FunctionCall]:
        """
        Try to parse function-like format:
        action_name(param1="value1", param2="value2")
        """
        match = re.search(r'(\w+)\s*\(([^)]*)\)', response)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            
            arguments = {}
            # Parse key=value pairs
            for pair in re.findall(r'(\w+)\s*=\s*["\']?([^"\'`,]+)["\']?', args_str):
                arguments[pair[0]] = pair[1].strip()
            
            return FunctionCall(name=name, arguments=arguments, raw_response=response)
        return None


# =============================================================================
# Pre-built Action Spaces for Common Games
# =============================================================================

def create_language_game_actions(
    teammates: List[str] = None,
    opponents: List[str] = None,
    vocabulary: List[str] = None
) -> ActionSpace:
    """
    Create action space for language/communication games.
    
    Actions:
    - stay_silent: Do nothing
    - talk_to_teammate: Send message to teammate(s)
    - talk_to_opponent: Send message to opponent(s)
    - broadcast: Send message to everyone
    - claim_resource: Claim a resource (end game action)
    """
    space = ActionSpace("language_game")
    
    # Build target enums
    teammate_targets = teammates or ["teammate"]
    opponent_targets = opponents or ["opponent"]
    all_targets = teammate_targets + opponent_targets + ["all"]
    
    # Stay silent
    space.add(ActionDef(
        name="stay_silent",
        description="Do nothing this turn. Use when you want to wait and observe.",
        parameters=[]
    ))
    
    # Talk to teammate
    space.add(ActionDef(
        name="talk_to_teammate",
        description="Send a message to one or more teammates. Use internal/honest communication.",
        parameters=[
            Parameter(
                "target", "enum", 
                "Which teammate to talk to",
                enum_values=teammate_targets + (["all_teammates"] if len(teammate_targets) > 1 else [])
            ),
            Parameter(
                "content", "string",
                f"Message content (tokens separated by space){' from vocabulary: ' + str(vocabulary[:5]) + '...' if vocabulary else ''}"
            ),
        ]
    ))
    
    # Talk to opponent
    space.add(ActionDef(
        name="talk_to_opponent",
        description="Send a message to opponent(s). Be strategic - you may want to hide or mislead.",
        parameters=[
            Parameter(
                "target", "enum",
                "Which opponent to talk to",
                enum_values=opponent_targets + (["all_opponents"] if len(opponent_targets) > 1 else [])
            ),
            Parameter(
                "content", "string",
                f"Message content (tokens separated by space){' from vocabulary: ' + str(vocabulary[:5]) + '...' if vocabulary else ''}"
            ),
        ]
    ))
    
    # Broadcast
    space.add(ActionDef(
        name="broadcast",
        description="Send a message to everyone (teammates and opponents).",
        parameters=[
            Parameter(
                "content", "string",
                "Message content (tokens separated by space)"
            ),
        ]
    ))
    
    # Claim resource (game-ending action)
    space.add(ActionDef(
        name="claim_resource",
        description="Claim a specific resource. This ends the game if successful. Only do this when confident about location AND value.",
        parameters=[
            Parameter(
                "resource_id", "string",
                "ID of the resource to claim (e.g., 'r1', 'r2')"
            ),
            Parameter(
                "reason", "string",
                "Brief explanation of why you believe this is the best resource",
                required=False
            ),
        ]
    ))
    
    return space


def create_negotiation_actions() -> ActionSpace:
    """Create action space for negotiation games."""
    space = ActionSpace("negotiation")
    
    space.add(ActionDef(
        name="propose",
        description="Make a proposal to another party",
        parameters=[
            Parameter("target", "string", "Who to propose to"),
            Parameter("offer", "string", "What you're offering"),
            Parameter("request", "string", "What you want in return"),
        ]
    ))
    
    space.add(ActionDef(
        name="accept",
        description="Accept a proposal",
        parameters=[
            Parameter("proposal_id", "string", "ID of the proposal to accept"),
        ]
    ))
    
    space.add(ActionDef(
        name="reject",
        description="Reject a proposal",
        parameters=[
            Parameter("proposal_id", "string", "ID of the proposal to reject"),
            Parameter("reason", "string", "Reason for rejection", required=False),
        ]
    ))
    
    space.add(ActionDef(
        name="counter_offer",
        description="Make a counter-offer",
        parameters=[
            Parameter("original_proposal", "string", "ID of the original proposal"),
            Parameter("new_offer", "string", "Your counter-offer"),
        ]
    ))
    
    return space

