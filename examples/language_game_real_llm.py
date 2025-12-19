"""
Language Game with Real LLM - Resource Scramble

A pure language/reasoning game where agents must communicate to locate 
the highest-value safe resource. No movement - just reasoning and claiming.

Key Features:
1. Abstract coordinate system (tokens represent x,y positions)
2. Information asymmetry (Team A: locations, Team B: values)
3. Competitive claiming (first correct claim wins)
4. Token invention (agents can create "secret codes")
5. Enhanced logging with translations

Usage:
    python language_game_real_llm.py --backend openai --api-key sk-xxx --model gpt-4o
    python language_game_real_llm.py --backend vllm --model meta-llama/Llama-3-8B
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from core.protocol import Message, Observation, Action, GameLogger
from core.channel import Channel, VocabularyFilter
from core.llm import LLMBackend
from core.llm_backends import create_llm_backend
from core.action_space import ActionSpace, ActionDef, Parameter, ActionParser


# =============================================================================
# Game Logging System
# =============================================================================

@dataclass
class AgentTurnLog:
    """Log of a single agent's turn."""
    round_num: int
    agent_id: str
    team: str
    
    # Input context
    env_state: Dict[str, Any]
    received_messages: List[Dict]
    memory_snapshot: List[str]
    
    # LLM interaction
    system_prompt: str
    user_prompt: str
    llm_response: str
    reasoning: str  # Extracted from response
    
    # Output
    action_type: Optional[str]
    action_params: Dict[str, Any]
    message_sent: Optional[Dict]
    message_filtered: Optional[List[str]]
    
    # Result
    action_result: Optional[Dict]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class GameLog:
    """Complete log of a game session."""
    game_id: str
    start_time: str
    end_time: Optional[str] = None
    
    # Game configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Ground truth
    resources: List[Dict] = field(default_factory=list)
    target_resource: Optional[Dict] = None
    
    # All turns
    turns: List[AgentTurnLog] = field(default_factory=list)
    
    # Outcome
    winner: Optional[str] = None
    total_rounds: int = 0
    total_messages: int = 0
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "resources": self.resources,
            "target_resource": self.target_resource,
            "turns": [
                {
                    "round_num": t.round_num,
                    "agent_id": t.agent_id,
                    "team": t.team,
                    "env_state": t.env_state,
                    "received_messages": t.received_messages,
                    "memory_snapshot": t.memory_snapshot,
                    "system_prompt": t.system_prompt,
                    "user_prompt": t.user_prompt,
                    "llm_response": t.llm_response,
                    "reasoning": t.reasoning,
                    "action_type": t.action_type,
                    "action_params": t.action_params,
                    "message_sent": t.message_sent,
                    "message_filtered": t.message_filtered,
                    "action_result": t.action_result,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
            "winner": self.winner,
            "total_rounds": self.total_rounds,
            "total_messages": self.total_messages,
            "stats": self.stats,
        }
    
    def to_readable_transcript(self) -> str:
        """Generate a human-readable transcript of the game."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"GAME TRANSCRIPT - {self.game_id}")
        lines.append("=" * 70)
        
        # Ground truth
        lines.append("\n[📦 Resources - Ground Truth]")
        for r in self.resources:
            trap = "⚠️ TRAP" if r.get("is_trap") else "✓ Safe"
            lines.append(f"  {r['id']}: ({r['x']},{r['y']}) [{r['x_token']},{r['y_token']}] Value={r['value']} {trap}")
        
        if self.target_resource:
            t = self.target_resource
            lines.append(f"\n  🎯 Target: {t['id']} at ({t['x']},{t['y']}) Value={t['value']}")
        
        lines.append("\n" + "=" * 70)
        lines.append("GAME PLAY")
        lines.append("=" * 70)
        
        current_round = 0
        for turn in self.turns:
            # Round header
            if turn.round_num != current_round:
                current_round = turn.round_num
                lines.append(f"\n--- Round {current_round} ---")
            
            team_icon = "🔵" if turn.team == "team_a" else "🔴"
            lines.append(f"\n  [{team_icon}] {turn.agent_id}")
            
            # Reasoning (truncated)
            if turn.reasoning:
                reasoning = turn.reasoning[:100] + "..." if len(turn.reasoning) > 100 else turn.reasoning
                lines.append(f"      💭 Reasoning: {reasoning}")
            
            # Action
            if turn.action_type:
                if turn.action_type == "stay_silent":
                    lines.append(f"      🤫 Waits")
                elif turn.action_type == "claim_resource":
                    x = turn.action_params.get("x", "?")
                    y = turn.action_params.get("y", "?")
                    lines.append(f"      🎯 CLAIMS ({x}, {y})")
                    if turn.action_result:
                        lines.append(f"      → {turn.action_result.get('message', '')}")
                elif turn.action_type == "define_token":
                    token = turn.action_params.get("token", "tok29")
                    meaning = turn.action_params.get("meaning", "")
                    lines.append(f"      📝 Defines {token} = '{meaning}'")
                elif turn.action_type in ["talk_to_teammate", "talk_to_opponent"]:
                    action_name = turn.action_type.replace("_", " ").title()
                    lines.append(f"      💬 {action_name}")
            
            # Message
            if turn.message_filtered:
                tokens = turn.message_filtered
                meaning = " ".join([TOKEN_MEANINGS.get(t, f"[{t}]") for t in tokens])
                receivers = turn.message_sent.get("receivers", []) if turn.message_sent else []
                lines.append(f"         Tokens: {tokens}")
                lines.append(f"         Meaning: {meaning}")
                lines.append(f"         To: {receivers}")
        
        # Final result
        lines.append("\n" + "=" * 70)
        if self.winner:
            lines.append(f"🏆 WINNER: {self.winner.upper()}")
        else:
            lines.append("⏱️ Game ended (no winner)")
        lines.append(f"Total Rounds: {self.total_rounds}")
        lines.append(f"Total Messages: {self.total_messages}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """Save log to JSON file and readable transcript."""
        # Save JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save readable transcript
        transcript_path = path.replace('.json', '_transcript.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(self.to_readable_transcript())
        
        print(f"\n📝 Game log saved to: {path}")
        print(f"📜 Readable transcript saved to: {transcript_path}")


# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# GAME PARAMETERS (can be overridden via command line)
# =============================================================================
GRID_SIZE = 5          # Grid dimensions (GRID_SIZE x GRID_SIZE)
NUM_RESOURCES = 3      # Number of resources (also determines value levels)

# Derived constants (recalculated by reconfigure_game())
VALUE_LEVELS = []
COORD_RANGE = []
RESOURCE_TOKEN_START = 5
RESOURCE_TOKEN_END = 8
X_TOKEN_START = 8
X_TOKEN_END = 13
Y_TOKEN_START = 13
Y_TOKEN_END = 18
VALUE_TOKEN_START = 18
VALUE_TOKEN_END = 21
SAFETY_TOKEN_START = 21
SAFETY_TOKEN_END = 23
QUERY_TOKEN_START = 23
QUERY_TOKEN_END = 26
COMPARE_TOKEN_START = 26
COMPARE_TOKEN_END = 29
CUSTOM_TOKEN_START = 29
CUSTOM_TOKEN_END = 30
TOTAL_TOKENS = 30
VOCABULARY = []
BASE_TOKEN_MEANINGS = {}
TOKEN_MEANINGS = {}


def _build_base_token_meanings() -> Dict[str, str]:
    """Build base token meanings dynamically based on current GRID_SIZE and NUM_RESOURCES."""
    meanings = {}
    
    # Intent (tok1-4) - always fixed
    meanings["tok1"] = "ASK"
    meanings["tok2"] = "TELL"
    meanings["tok3"] = "CLAIM"
    meanings["tok4"] = "DEFINE"
    
    # Resource identifiers (tok5 onwards)
    for i in range(NUM_RESOURCES):
        meanings[f"tok{RESOURCE_TOKEN_START + i}"] = f"R{i+1}"
    
    # X coordinates - Team A knows the mapping
    for i in range(GRID_SIZE):
        meanings[f"tok{X_TOKEN_START + i}"] = "X?"
    
    # Y coordinates - Team A knows the mapping
    for i in range(GRID_SIZE):
        meanings[f"tok{Y_TOKEN_START + i}"] = "Y?"
    
    # Value tokens - Team B knows the mapping
    for i in range(NUM_RESOURCES):
        meanings[f"tok{VALUE_TOKEN_START + i}"] = "V?"
    
    # Safety tokens - Team B knows the mapping
    meanings[f"tok{SAFETY_TOKEN_START}"] = "SAFE_OR_TRAP?"
    meanings[f"tok{SAFETY_TOKEN_START + 1}"] = "SAFE_OR_TRAP?"
    
    # Query tokens - everyone knows
    meanings[f"tok{QUERY_TOKEN_START}"] = "WHERE"
    meanings[f"tok{QUERY_TOKEN_START + 1}"] = "VALUE?"
    meanings[f"tok{QUERY_TOKEN_START + 2}"] = "SAFE?"
    
    # Comparison tokens - everyone knows
    meanings[f"tok{COMPARE_TOKEN_START}"] = ">"
    meanings[f"tok{COMPARE_TOKEN_START + 1}"] = "<"
    meanings[f"tok{COMPARE_TOKEN_START + 2}"] = "="
    
    # Custom token for "黑话"
    meanings[f"tok{CUSTOM_TOKEN_START}"] = "[SECRET]"
    
    return meanings


def reconfigure_game(grid_size: int = 5, num_resources: int = 3):
    """Reconfigure game parameters. Must be called before game starts."""
    global GRID_SIZE, NUM_RESOURCES, VALUE_LEVELS, COORD_RANGE
    global RESOURCE_TOKEN_START, RESOURCE_TOKEN_END
    global X_TOKEN_START, X_TOKEN_END, Y_TOKEN_START, Y_TOKEN_END
    global VALUE_TOKEN_START, VALUE_TOKEN_END
    global SAFETY_TOKEN_START, SAFETY_TOKEN_END
    global QUERY_TOKEN_START, QUERY_TOKEN_END
    global COMPARE_TOKEN_START, COMPARE_TOKEN_END
    global CUSTOM_TOKEN_START, CUSTOM_TOKEN_END
    global TOTAL_TOKENS, VOCABULARY
    global BASE_TOKEN_MEANINGS, TOKEN_MEANINGS
    
    GRID_SIZE = grid_size
    NUM_RESOURCES = num_resources
    
    VALUE_LEVELS = list(range(1, NUM_RESOURCES + 1))
    COORD_RANGE = list(range(GRID_SIZE))
    
    # Token layout (dynamically sized):
    # tok1-4: intent (ASK, TELL, CLAIM, DEFINE)
    RESOURCE_TOKEN_START = 5
    RESOURCE_TOKEN_END = RESOURCE_TOKEN_START + NUM_RESOURCES
    
    X_TOKEN_START = RESOURCE_TOKEN_END
    X_TOKEN_END = X_TOKEN_START + GRID_SIZE
    
    Y_TOKEN_START = X_TOKEN_END
    Y_TOKEN_END = Y_TOKEN_START + GRID_SIZE
    
    VALUE_TOKEN_START = Y_TOKEN_END
    VALUE_TOKEN_END = VALUE_TOKEN_START + NUM_RESOURCES
    
    SAFETY_TOKEN_START = VALUE_TOKEN_END
    SAFETY_TOKEN_END = SAFETY_TOKEN_START + 2
    
    QUERY_TOKEN_START = SAFETY_TOKEN_END
    QUERY_TOKEN_END = QUERY_TOKEN_START + 3
    
    COMPARE_TOKEN_START = QUERY_TOKEN_END
    COMPARE_TOKEN_END = COMPARE_TOKEN_START + 3
    
    CUSTOM_TOKEN_START = COMPARE_TOKEN_END
    CUSTOM_TOKEN_END = CUSTOM_TOKEN_START + 1
    
    TOTAL_TOKENS = CUSTOM_TOKEN_END
    VOCABULARY = [f"tok{i}" for i in range(1, TOTAL_TOKENS)]
    
    BASE_TOKEN_MEANINGS = _build_base_token_meanings()
    TOKEN_MEANINGS = BASE_TOKEN_MEANINGS.copy()
    
    return {
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "total_tokens": TOTAL_TOKENS - 1,
        "vocabulary": VOCABULARY,
    }

# Initialize with defaults
reconfigure_game(GRID_SIZE, NUM_RESOURCES)

# Token Meanings:
# ===============
# INTENT (tok1-4):     What you want to do
# RESOURCE (tok5-7):   Which resource you're talking about
# X-COORD (tok8-12):   X position (0-4)
# Y-COORD (tok13-17):  Y position (0-4)
# VALUE (tok18-20):    Resource value level
# SAFETY:   Safe or trap  
# QUERY:    Question types (WHERE, VALUE?, SAFE?)
# COMPARE:  Comparison (>, <, =)
# CUSTOM:   For invented meanings ("黑话")

# Random mapping storage (initialized per game)
VALUE_TOKEN_MAP: Dict[int, str] = {}    # value -> token (e.g., 10 -> "tok23")
TOKEN_VALUE_MAP: Dict[str, int] = {}    # token -> value (e.g., "tok23" -> 10)
COORD_X_MAP: Dict[int, str] = {}        # x -> token
COORD_Y_MAP: Dict[int, str] = {}        # y -> token
TOKEN_X_MAP: Dict[str, int] = {}        # token -> x
TOKEN_Y_MAP: Dict[str, int] = {}        # token -> y
SAFETY_TOKEN_MAP: Dict[bool, str] = {}  # is_trap -> token
TOKEN_SAFETY_MAP: Dict[str, bool] = {}  # token -> is_trap


def initialize_random_mappings(seed: int = None):
    """Initialize random token mappings for a new game."""
    global VALUE_TOKEN_MAP, TOKEN_VALUE_MAP
    global COORD_X_MAP, COORD_Y_MAP, TOKEN_X_MAP, TOKEN_Y_MAP
    global SAFETY_TOKEN_MAP, TOKEN_SAFETY_MAP, TOKEN_MEANINGS
    
    if seed is not None:
        random.seed(seed)
    
    # Shuffle value tokens -> values (1, 2, ..., NUM_RESOURCES)
    value_tokens = [f"tok{i}" for i in range(VALUE_TOKEN_START, VALUE_TOKEN_END)]
    values = VALUE_LEVELS
    random.shuffle(value_tokens)
    
    VALUE_TOKEN_MAP = {v: t for v, t in zip(values, value_tokens)}
    TOKEN_VALUE_MAP = {t: v for v, t in VALUE_TOKEN_MAP.items()}
    
    # Shuffle X coordinate tokens -> coordinates (0, 1, ..., GRID_SIZE-1)
    x_tokens = [f"tok{i}" for i in range(X_TOKEN_START, X_TOKEN_END)]
    x_coords = COORD_RANGE
    random.shuffle(x_tokens)
    
    COORD_X_MAP = {c: t for c, t in zip(x_coords, x_tokens)}
    TOKEN_X_MAP = {t: c for c, t in COORD_X_MAP.items()}
    
    # Shuffle Y coordinate tokens -> coordinates (0, 1, ..., GRID_SIZE-1)
    y_tokens = [f"tok{i}" for i in range(Y_TOKEN_START, Y_TOKEN_END)]
    y_coords = COORD_RANGE
    random.shuffle(y_tokens)
    
    COORD_Y_MAP = {c: t for c, t in zip(y_coords, y_tokens)}
    TOKEN_Y_MAP = {t: c for c, t in COORD_Y_MAP.items()}
    
    # Shuffle safety tokens -> (SAFE, TRAP)
    safety_tokens = [f"tok{SAFETY_TOKEN_START}", f"tok{SAFETY_TOKEN_START + 1}"]
    random.shuffle(safety_tokens)
    SAFETY_TOKEN_MAP = {False: safety_tokens[0], True: safety_tokens[1]}  # False=SAFE, True=TRAP
    TOKEN_SAFETY_MAP = {safety_tokens[0]: False, safety_tokens[1]: True}
    
    # Update TOKEN_MEANINGS with actual mappings (for logging/display)
    TOKEN_MEANINGS = _build_base_token_meanings()
    for v, t in VALUE_TOKEN_MAP.items():
        TOKEN_MEANINGS[t] = f"V{v}"
    for c, t in COORD_X_MAP.items():
        TOKEN_MEANINGS[t] = f"X{c}"
    for c, t in COORD_Y_MAP.items():
        TOKEN_MEANINGS[t] = f"Y{c}"
    TOKEN_MEANINGS[SAFETY_TOKEN_MAP[False]] = "SAFE"
    TOKEN_MEANINGS[SAFETY_TOKEN_MAP[True]] = "TRAP"
    
    return {
        "value_map": VALUE_TOKEN_MAP,
        "x_map": COORD_X_MAP,
        "y_map": COORD_Y_MAP,
        "safety_map": SAFETY_TOKEN_MAP,
    }


def value_to_token(value: int) -> str:
    """Convert numeric value to its randomly assigned token."""
    return VALUE_TOKEN_MAP.get(value, "tok18")


def coord_to_tokens(x: int, y: int) -> Tuple[str, str]:
    """Convert coordinates to their randomly assigned tokens."""
    return COORD_X_MAP.get(x, "tok8"), COORD_Y_MAP.get(y, "tok13")


def safety_to_token(is_trap: bool) -> str:
    """Convert safety status to its randomly assigned token."""
    return SAFETY_TOKEN_MAP.get(is_trap, "tok28")

SYSTEM_PROMPT = """You are {agent_id} on {team} in the Resource Scramble game.

## 🎯 Objective
Find and CLAIM the highest-value SAFE resource!
- There are {num_resources} resources, each with a UNIQUE value (1 to {num_resources})
- You must figure out which one has the HIGHEST value ({num_resources} is best!)
- First team to claim a SAFE resource WINS (higher value = better win)
- Claiming a TRAP = instant LOSS

## 🏁 WINNING STRATEGY
**CLAIM AS SOON AS YOU HAVE ENOUGH INFO!** The opponent is also trying to claim.
- You need: (1) Location, (2) Value, (3) Safety
- Once you know all three for the BEST resource → CLAIM IMMEDIATELY!

## 🗺️ Grid Information  
- Grid: {grid_size}×{grid_size} (coordinates 0-{max_coord})
- Resources: {resource_ids}
- To CLAIM: specify X,Y as NUMBERS (not tokens)

## 📊 Information Split
**Your team ({team}) knows:** {your_info}
**Other team knows:** {their_info}

## 🗣️ Token Protocol
Structure: [INTENT] [RESOURCE] [INFO...]

### Your Known Tokens:
{known_tokens}

### Unknown Tokens: {unknown_tokens}

## 📝 Example Messages
- Ask location: "tok1 tok5 tok30" = ASK R1 WHERE
- Tell location: "tok2 tok5 tok9 tok14" = TELL R1 X1 Y1
- Ask value: "tok1 tok5 tok31" = ASK R1 VALUE?
- Compare values: "tok2 tok5 tok35 tok6" = TELL R1 > R2 (R1's value is greater than R2)
- Compare values: "tok2 tok6 tok36 tok7" = TELL R2 < R3 (R2's value is less than R3)
- Tell safety: "tok2 tok5 tok28" = TELL R1 SAFE (or TRAP)

## 🔢 Comparison Tokens (everyone knows)
- tok35 = ">" (greater than)
- tok36 = "<" (less than)
- tok37 = "=" (equal)
Use these to express relative value! e.g., "R1 > R2" means R1 has higher value than R2.

## ⚠️ RULES
1. Messages = ONLY tokens
2. CLAIM requires NUMBERS: claim(x=2, y=3) NOT tokens
3. Team A knows coordinate tokens (tok8-17)
4. Team B knows value tokens (tok18-27) and safety tokens (tok28-29)
5. Use comparison tokens (tok35-37) to express relative values!

## 🎮 Actions
{actions}

Respond with JSON:
```json
{{"action": "...", "params": {{...}}, "reasoning": "..."}}
```
"""

USER_PROMPT = """## Round {round}

**Your Information:**
{resources}

**Messages This Round:**
{messages}

**Your Memory:**
{memory}

## ⚡ DECISION CHECKLIST - Check before deciding!
For each resource you're considering:
- [ ] Do I know its LOCATION (X, Y)?
- [ ] Do I know its VALUE (1-{num_resources})?
- [ ] Do I know if it's SAFE or TRAP?
- [ ] Is this the HIGHEST value among known safe resources?

**If you know location + value + safety for the best resource → CLAIM NOW!**
**Don't wait - the opponent might claim first!**

What do you do? Respond with JSON:"""


# =============================================================================
# Enhanced Agent with Movement and Token Invention
# =============================================================================

class GameAgent:
    """Agent that uses LLM for reasoning and communication."""
    
    def __init__(self,
                 agent_id: str,
                 team: str,
                 llm: LLMBackend,
                 teammates: List[str],
                 opponents: List[str],
                 game_map = None):  # Not used in pure reasoning game
        self.agent_id = agent_id
        self.team = team
        self.llm = llm
        self.teammates = teammates
        self.opponents = opponents
        
        # Known tokens based on team
        self.known_tokens = self._init_known_tokens()
        
        # Custom definitions this agent has made
        self.custom_definitions: Dict[str, str] = {}
        
        # Action space
        self.action_space = self._create_action_space()
        
        # Memory - what the agent has learned through communication
        self.memory: List[str] = []
        self.round_number = 0
        self.conversation_history: List[Dict] = []  # LLM chat history
    
    def _init_known_tokens(self) -> Dict[str, str]:
        """Initialize tokens this agent knows based on team."""
        known = {}
        
        # Everyone knows: intent (tok1-4)
        for i in range(1, 5):
            known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]
        
        # Everyone knows: resources
        for i in range(RESOURCE_TOKEN_START, RESOURCE_TOKEN_END):
            known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]
        
        # Everyone knows: query tokens
        for i in range(QUERY_TOKEN_START, QUERY_TOKEN_END):
            known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]
        
        # Everyone knows: comparison tokens
        for i in range(COMPARE_TOKEN_START, COMPARE_TOKEN_END):
            known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]
        
        if self.team == "team_a":
            # Team A knows the ACTUAL coordinate mappings
            for i in range(X_TOKEN_START, X_TOKEN_END):
                known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]  # e.g., "X2"
            for i in range(Y_TOKEN_START, Y_TOKEN_END):
                known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]  # e.g., "Y3"
            # Team A knows value tokens EXIST but not what they mean
            for i in range(VALUE_TOKEN_START, VALUE_TOKEN_END):
                known[f"tok{i}"] = "V?"
            for i in range(SAFETY_TOKEN_START, SAFETY_TOKEN_END):
                known[f"tok{i}"] = "SAFE_OR_TRAP?"
        else:
            # Team B knows coordinate tokens EXIST but not what they mean
            for i in range(X_TOKEN_START, X_TOKEN_END):
                known[f"tok{i}"] = "X?"
            for i in range(Y_TOKEN_START, Y_TOKEN_END):
                known[f"tok{i}"] = "Y?"
            # Team B knows the ACTUAL value and safety mappings
            for i in range(VALUE_TOKEN_START, VALUE_TOKEN_END):
                known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]  # e.g., "V3"
            for i in range(SAFETY_TOKEN_START, SAFETY_TOKEN_END):
                known[f"tok{i}"] = TOKEN_MEANINGS[f"tok{i}"]  # e.g., "SAFE"
        
        return known
    
    def _create_action_space(self) -> ActionSpace:
        space = ActionSpace("resource_scramble")
        
        space.add(ActionDef(
            name="stay_silent",
            description="Wait and observe this round",
            parameters=[]
        ))
        
        space.add(ActionDef(
            name="talk_to_teammate",
            description="Send message to teammate. Use for honest info sharing.",
            parameters=[
                Parameter("target", "enum", "Teammate to talk to",
                         enum_values=self.teammates + ["all_teammates"]),
                Parameter("content", "string", "ONLY TOKENS! e.g., 'tok2 tok5 tok9 tok14' (TELL R1 X1 Y1)")
            ]
        ))
        
        space.add(ActionDef(
            name="talk_to_opponent",
            description="Send message to opponent. Be strategic - you can share, mislead, or ask!",
            parameters=[
                Parameter("target", "enum", "Opponent to talk to",
                         enum_values=self.opponents + ["all_opponents"]),
                Parameter("content", "string", "ONLY TOKENS! You may lie or trade info")
            ]
        ))
        
        space.add(ActionDef(
            name="define_token",
            description="Invent a secret meaning for tok38 or tok39 (only your team knows)",
            parameters=[
                Parameter("token", "enum", "Which token to define",
                         enum_values=["tok38", "tok39"]),
                Parameter("meaning", "string", "What this token means to your team"),
            ]
        ))
        
        space.add(ActionDef(
            name="claim_resource",
            description="Claim a resource at coordinates. You MUST specify actual numbers (0-4), NOT tokens! This requires you to UNDERSTAND what coordinate tokens mean.",
            parameters=[
                Parameter("x", "int", "X coordinate as a NUMBER (0-4). If you only know tokens like tok10, you must figure out what number it represents!"),
                Parameter("y", "int", "Y coordinate as a NUMBER (0-4). You cannot just copy tokens - you must understand them!")
            ]
        ))
        
        return space
    
    def act(self,
            observations: List[Observation],
            messages: List[Message],
            env_state: Dict[str, Any]) -> Tuple[Optional[Action], Optional[Message], Dict]:
        """Make decision using LLM. Returns (action, message, turn_info)."""
        self.round_number += 1
        
        # Prepare turn info for logging
        turn_info = {
            "env_state": env_state,
            "received_messages": [
                {"sender": m.sender, "content": m.content, "receivers": m.receivers}
                for m in messages
            ],
            "memory_snapshot": list(self.memory),
        }
        
        # Update memory
        for msg in messages:
            tokens = msg.content if isinstance(msg.content, list) else str(msg.content).split()
            translation = self._translate_tokens(tokens)
            self.memory.append(f"[R{self.round_number}] {msg.sender}: {tokens} ({translation})")
        
        # Build prompts
        system_prompt = self._build_system_prompt(env_state)
        user_prompt = self._build_user_prompt(messages, env_state)
        
        turn_info["system_prompt"] = system_prompt
        turn_info["user_prompt"] = user_prompt
        
        # Call LLM
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for hist in self.conversation_history[-4:]:
            chat_messages.insert(-1, hist)
        
        response = self.llm.chat(chat_messages)
        turn_info["llm_response"] = response
        
        # Extract reasoning from response
        turn_info["reasoning"] = self._extract_reasoning(response)
        
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Parse and execute
        func_call = ActionParser.parse(response, self.action_space)
        action, message = self._execute(func_call, raw_response=response, env_state=env_state)
        
        return action, message, turn_info
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from LLM response."""
        import re
        # Try to extract from JSON
        match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response)
        if match:
            return match.group(1)
        # Try multiline
        match = re.search(r'"reasoning"\s*:\s*"(.*?)"', response, re.DOTALL)
        if match:
            return match.group(1).replace('\n', ' ').strip()
        # Return full response if can't extract
        return response[:500] if len(response) > 500 else response
    
    def _translate_tokens(self, tokens: List[str]) -> str:
        """Translate tokens to meanings this agent knows."""
        meanings = []
        for tok in tokens:
            if tok in self.known_tokens:
                meanings.append(self.known_tokens[tok])
            elif tok in self.custom_definitions:
                meanings.append(f"[{self.custom_definitions[tok]}]")
            else:
                meanings.append(f"[?{tok}?]")
        return " ".join(meanings)
    
    def _build_system_prompt(self, env_state: Dict) -> str:
        num_res = env_state.get("num_resources", NUM_RESOURCES)
        grid_sz = env_state.get("grid_size", GRID_SIZE)
        your_info = "resource POSITIONS (X,Y coordinates)" if self.team == "team_a" else f"resource VALUES (1-{num_res}) and SAFETY"
        their_info = f"resource VALUES (1-{num_res}) and SAFETY" if self.team == "team_a" else "resource POSITIONS (X,Y coordinates)"
        
        known_str = "\n".join([f"  {tok} = {meaning}" for tok, meaning in sorted(self.known_tokens.items())])
        unknown = [tok for tok in VOCABULARY if tok not in self.known_tokens]
        unknown_str = ", ".join(unknown)
        
        return SYSTEM_PROMPT.format(
            agent_id=self.agent_id,
            team=self.team,
            grid_size=grid_sz,
            max_coord=grid_sz - 1,
            num_resources=num_res,
            resource_ids=list(env_state.get("resource_ids", [f"r{i+1}" for i in range(num_res)])),
            your_info=your_info,
            their_info=their_info,
            known_tokens=known_str,
            unknown_tokens=unknown_str,
            actions=self.action_space.to_prompt()
        )
    
    def _build_user_prompt(self, messages: List[Message], env_state: Dict) -> str:
        # What this agent can see
        resources_view = env_state.get("resources", {})
        resources_lines = []
        for rid, info in resources_view.items():
            resources_lines.append(f"  {rid}: {info}")
        resources_str = "\n".join(resources_lines) if resources_lines else "  (no info yet)"
        
        # Messages with translations
        msg_lines = []
        for msg in messages:
            tokens = msg.content if isinstance(msg.content, list) else str(msg.content).split()
            translation = self._translate_tokens(tokens)
            msg_lines.append(f"  {msg.sender}: {' '.join(tokens)}")
            msg_lines.append(f"    → You understand: {translation}")
        messages_str = "\n".join(msg_lines) if msg_lines else "  (none)"
        
        # Memory
        memory_str = "\n".join([f"  {m}" for m in self.memory[-5:]]) if self.memory else "  (none)"
        
        num_res = env_state.get("num_resources", NUM_RESOURCES)
        
        return USER_PROMPT.format(
            round=self.round_number,
            resources=resources_str,
            messages=messages_str,
            memory=memory_str,
            num_resources=num_res
        )
    
    def _execute(self, func_call, raw_response: str, env_state: Dict = None) -> Tuple[Optional[Action], Optional[Message]]:
        if not func_call:
            print(f"    ⚠️ Parse failed: {raw_response[:80]}...")
            return None, None
        
        action = None
        message = None
        name = func_call.name
        params = func_call.arguments
        grid_sz = env_state.get("grid_size", GRID_SIZE) if env_state else GRID_SIZE
        
        if name == "stay_silent":
            action = Action(self.agent_id, "stay_silent")
        
        elif name == "talk_to_teammate":
            target = params.get("target", self.teammates[0] if self.teammates else "teammate")
            content = params.get("content", "")
            
            receivers = [f"group:{self.team}"] if target == "all_teammates" else [target]
            message = Message(self.agent_id, receivers, content, channel="intra_team")
            action = Action(self.agent_id, "talk_to_teammate", str(receivers), {"content": content})
        
        elif name == "talk_to_opponent":
            target = params.get("target", "opponent")
            content = params.get("content", "")
            
            opponent_team = "team_b" if self.team == "team_a" else "team_a"
            receivers = [f"group:{opponent_team}"] if target == "all_opponents" else [target]
            message = Message(self.agent_id, receivers, content, channel="inter_team")
            action = Action(self.agent_id, "talk_to_opponent", str(receivers), {"content": content})
        
        elif name == "define_token":
            token = params.get("token", f"tok{CUSTOM_TOKEN_START}")
            meaning = params.get("meaning", "secret")
            self.custom_definitions[token] = meaning
            self.known_tokens[token] = f"[{meaning}]"
            
            # Notify teammate
            message = Message(
                self.agent_id, 
                [f"group:{self.team}"],
                f"tok4 {token}",  # DEFINE + custom token
                channel="intra_team",
                metadata={"definition": meaning, "token": token}
            )
            action = Action(self.agent_id, "define_token", token, {"meaning": meaning})
        
        elif name == "claim_resource":
            # Must use actual numbers, not tokens!
            x = int(params.get("x", 0))
            y = int(params.get("y", 0))
            # Clamp to valid range
            x = max(0, min(grid_sz - 1, x))
            y = max(0, min(grid_sz - 1, y))
            action = Action(self.agent_id, "claim_resource", f"({x},{y})", {"x": x, "y": y})
        
        return action, message


# =============================================================================
# Game State & Resources
# =============================================================================

@dataclass
class GameResource:
    """A resource in the game."""
    id: str
    x: int
    y: int
    value: int  # 1 to NUM_RESOURCES
    is_trap: bool
    
    def position_str(self) -> str:
        return f"({self.x}, {self.y})"
    
    def get_position_tokens(self) -> Tuple[str, str]:
        """Get the tokens representing this position using the random mapping."""
        return coord_to_tokens(self.x, self.y)
    
    def get_value_token(self) -> str:
        """Get the token representing this resource's value."""
        return value_to_token(self.value)
    
    def get_safety_token(self) -> str:
        """Get the token representing this resource's safety status."""
        return safety_to_token(self.is_trap)


class ResourceScrambleGame:
    """Pure reasoning game - no movement, just communication and claiming."""
    
    def __init__(self, grid_size: int = 5, num_resources: int = 3):
        self.grid_size = grid_size
        self.num_resources = num_resources
        
        # Logger with token translation
        def translate_tokens(tokens: List[str]) -> str:
            return " ".join([TOKEN_MEANINGS.get(t, f"[{t}]") for t in tokens])
        self.logger = GameLogger(translate_fn=translate_tokens)
        
        # Generate resources at random positions
        self.resources: Dict[str, GameResource] = {}
        self._generate_resources()
        
        # Track agents and teams
        self.agents: Dict[str, str] = {}  # agent_id -> team
        
        self.winner = None
        self.game_over = False
    
    def _generate_resources(self):
        """Generate resources at random positions with unique values."""
        used_positions = set()
        
        # Each resource gets a unique value from VALUE_LEVELS
        # Shuffle to randomize which resource gets which value
        values = list(VALUE_LEVELS)  # [1, 2, 3]
        random.shuffle(values)
        
        # Randomly choose one resource to be a trap (not the highest value one)
        trap_index = random.randint(0, self.num_resources - 1)
        # Make sure highest value is never a trap
        max_value_index = values.index(max(values))
        if trap_index == max_value_index:
            trap_index = (trap_index + 1) % self.num_resources
        
        for i in range(self.num_resources):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if (x, y) not in used_positions:
                    used_positions.add((x, y))
                    break
            
            rid = f"r{i+1}"
            self.resources[rid] = GameResource(
                id=rid,
                x=x,
                y=y,
                value=values[i],
                is_trap=(i == trap_index)
            )
    
    def add_agent(self, agent_id: str, team: str):
        """Register an agent."""
        self.agents[agent_id] = team
    
    def get_team_view(self, agent_id: str) -> Dict:
        """Get what this agent can see based on team."""
        team = self.agents.get(agent_id, "team_a")
        view = {
            "resource_ids": list(self.resources.keys()),
            "resources": {},
            "num_resources": len(self.resources),
            "grid_size": self.grid_size,
        }
        
        for rid, res in self.resources.items():
            if team == "team_a":
                # Team A knows positions only (with actual coordinate values)
                x_tok, y_tok = res.get_position_tokens()
                view["resources"][rid] = f"Position: ({res.x}, {res.y}) [tokens: {x_tok} {y_tok}]"
            else:
                # Team B knows numeric values and safety
                val_tok = res.get_value_token()
                safe_tok = res.get_safety_token()
                safety_str = "⚠️ TRAP!" if res.is_trap else "SAFE"
                view["resources"][rid] = f"Value: {res.value} [token: {val_tok}], {safety_str} [token: {safe_tok}]"
        
        return view
    
    def process_action(self, agent_id: str, action: Action, round_num: int) -> Dict:
        """Process an agent's action."""
        team = self.agents.get(agent_id, "unknown")
        result = {"success": False}
        
        if action.action_type == "claim_resource":
            x = action.params.get("x", 0)
            y = action.params.get("y", 0)
            
            # Find resource at this position
            found_resource = None
            for rid, res in self.resources.items():
                if res.x == x and res.y == y:
                    found_resource = res
                    break
            
            if found_resource:
                if found_resource.is_trap:
                    result = {
                        "success": False,
                        "is_trap": True,
                        "message": f"💀 TRAP! {agent_id} claimed a trap at ({x},{y})!"
                    }
                    self.game_over = True
                    self.winner = "team_b" if team == "team_a" else "team_a"
                else:
                    result = {
                        "success": True,
                        "resource": found_resource.id,
                        "value": found_resource.value,
                        "message": f"🎉 {agent_id} claimed {found_resource.id} (value={found_resource.value}) at ({x},{y})!"
                    }
                    self.game_over = True
                    self.winner = team
            else:
                result = {
                    "success": False,
                    "message": f"❌ No resource at position ({x},{y})"
                }
            
            self.logger.log_action(
                round_num, agent_id, team, "claim_resource",
                f"({x},{y})", None, result
            )
        
        elif action.action_type == "define_token":
            token = action.target
            meaning = action.params.get("meaning", "")
            self.logger.log_custom_definition(round_num, agent_id, team, token, meaning)
            result = {"success": True, "token": token, "meaning": meaning}
        
        return result
    
    def log_message(self, round_num: int, agent_id: str, team: str, 
                    action_type: str, content: Any, target: str):
        """Log a message with translation."""
        self.logger.log_action(round_num, agent_id, team, action_type, content, target)
    
    def best_resource(self) -> GameResource:
        """Get the best (highest value, safe) resource."""
        safe = [r for r in self.resources.values() if not r.is_trap]
        return max(safe, key=lambda r: r.value) if safe else list(self.resources.values())[0]
    
    def print_state(self):
        """Print game state."""
        print(f"\n  Grid: {self.grid_size}x{self.grid_size}")
        print(f"  Resources:")
        for rid, res in self.resources.items():
            x_tok, y_tok = res.get_position_tokens()
            trap = "⚠️ TRAP" if res.is_trap else "✓ Safe"
            print(f"    {rid}: ({res.x},{res.y}) [{x_tok},{y_tok}], Value={res.value}, {trap}")
    
    def print_transcript(self):
        """Print game transcript with translations."""
        self.logger.print_transcript()


# =============================================================================
# Main Game Loop
# =============================================================================

def run_game(llm: LLMBackend, max_rounds: int = 15, save_log: bool = True, log_dir: str = "./logs", seed: int = None):
    """Run the language game with full logging."""
    
    # Set global random seed for reproducibility
    if seed is None:
        seed = int(datetime.now().timestamp())
    
    random.seed(seed)
    print(f"[🎲 Random seed: {seed}]")
    
    # Initialize random token mappings for this game
    token_mappings = initialize_random_mappings(seed=seed)
    
    # Initialize game log
    game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    game_log = GameLog(
        game_id=game_id,
        start_time=datetime.now().isoformat(),
        config={
            "grid_size": GRID_SIZE,
            "num_resources": NUM_RESOURCES,
            "max_rounds": max_rounds,
            "vocabulary_size": len(VOCABULARY),
            "token_mappings": {
                "value": {str(k): v for k, v in token_mappings["value_map"].items()},
                "x_coord": {str(k): v for k, v in token_mappings["x_map"].items()},
                "y_coord": {str(k): v for k, v in token_mappings["y_map"].items()},
                "safety": {str(k): v for k, v in token_mappings["safety_map"].items()},
            },
            "seed": seed,
        }
    )
    
    print("=" * 70)
    print("Resource Scramble - Pure Reasoning Language Game")
    print(f"Game ID: {game_id}")
    print("=" * 70)
    
    # Show token mapping (for researcher)
    print("\n[🔀 Random Token Mappings (Ground Truth)]")
    print(f"  Value tokens: {', '.join([f'{v}->{t}' for v, t in sorted(VALUE_TOKEN_MAP.items())])}")
    print(f"  X coord tokens: {', '.join([f'{c}->{t}' for c, t in sorted(COORD_X_MAP.items())])}")
    print(f"  Y coord tokens: {', '.join([f'{c}->{t}' for c, t in sorted(COORD_Y_MAP.items())])}")
    print(f"  Safety tokens: SAFE->{SAFETY_TOKEN_MAP[False]}, TRAP->{SAFETY_TOKEN_MAP[True]}")
    
    # Initialize game
    game = ResourceScrambleGame(GRID_SIZE, NUM_RESOURCES)
    
    # Add agents
    game.add_agent("alice", "team_a")
    game.add_agent("bob", "team_a")
    game.add_agent("carol", "team_b")
    game.add_agent("dave", "team_b")
    
    # Log ground truth
    for rid, res in game.resources.items():
        x_tok, y_tok = res.get_position_tokens()
        game_log.resources.append({
            "id": rid,
            "x": res.x, "y": res.y,
            "x_token": x_tok, "y_token": y_tok,
            "value": res.value,
            "is_trap": res.is_trap
        })
    
    best = game.best_resource()
    x_tok, y_tok = best.get_position_tokens()
    game_log.target_resource = {
        "id": best.id,
        "x": best.x, "y": best.y,
        "x_token": x_tok, "y_token": y_tok,
        "value": best.value
    }
    
    # Show ground truth (researcher view)
    print("\n[📦 Resources - Ground Truth (Researcher View)]")
    print("-" * 50)
    game.print_state()
    print(f"\n  🎯 Target: {best.id} at ({best.x},{best.y}) [tokens: {x_tok} {y_tok}]")
    print(f"     Value={best.value}, Safe")
    
    print("\n[👥 Teams]")
    print(f"  Team A (knows positions): alice, bob")
    print(f"  Team B (knows values/safety): carol, dave")
    
    # Create agents
    agents = {}
    for agent_id in ["alice", "bob"]:
        agents[agent_id] = GameAgent(
            agent_id, "team_a", llm,
            teammates=["bob"] if agent_id == "alice" else ["alice"],
            opponents=["carol", "dave"],
            game_map=None
        )
    for agent_id in ["carol", "dave"]:
        agents[agent_id] = GameAgent(
            agent_id, "team_b", llm,
            teammates=["dave"] if agent_id == "carol" else ["carol"],
            opponents=["alice", "bob"],
            game_map=None
        )
    
    # Setup channel
    vocab_filter = VocabularyFilter(VOCABULARY, max_length=8, extract_mode="anywhere")
    channel = Channel(default_filter=vocab_filter)
    channel.create_group("team_a", ["alice", "bob"])
    channel.create_group("team_b", ["carol", "dave"])
    
    # Run game
    print("\n" + "=" * 70)
    print("🎮 GAME START")
    print("=" * 70)
    
    final_round = 0
    for round_num in range(1, max_rounds + 1):
        final_round = round_num
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")
        
        # Alternate order
        order = ["alice", "carol", "bob", "dave"] if round_num % 2 == 1 else ["carol", "alice", "dave", "bob"]
        
        for agent_id in order:
            agent = agents[agent_id]
            team = agent.team
            team_icon = "🔵" if team == "team_a" else "🔴"
            
            # Get inputs
            pending_messages = channel.receive(agent_id)
            env_state = game.get_team_view(agent_id)
            observations = [Observation(agent_id, env_state, "environment")]
            
            print(f"\n  [{team_icon}] {agent_id}")
            
            # Agent decides (now returns turn_info)
            action, message, turn_info = agent.act(observations, pending_messages, env_state)
            
            # Print reasoning
            if turn_info.get("reasoning"):
                reasoning_preview = turn_info["reasoning"][:100]
                if len(turn_info["reasoning"]) > 100:
                    reasoning_preview += "..."
                print(f"      💭 Reasoning: {reasoning_preview}")
            
            # Initialize turn log
            turn_log = AgentTurnLog(
                round_num=round_num,
                agent_id=agent_id,
                team=team,
                env_state=turn_info.get("env_state", {}),
                received_messages=turn_info.get("received_messages", []),
                memory_snapshot=turn_info.get("memory_snapshot", []),
                system_prompt=turn_info.get("system_prompt", ""),
                user_prompt=turn_info.get("user_prompt", ""),
                llm_response=turn_info.get("llm_response", ""),
                reasoning=turn_info.get("reasoning", ""),
                action_type=action.action_type if action else None,
                action_params=action.params if action else {},
                message_sent=None,
                message_filtered=None,
                action_result=None
            )
            
            # Process action
            if action:
                if action.action_type == "stay_silent":
                    print(f"      🤫 Waits")
                
                elif action.action_type == "claim_resource":
                    x = action.params.get("x", "?")
                    y = action.params.get("y", "?")
                    print(f"      🎯 CLAIMS position ({x}, {y})")
                    result = game.process_action(agent_id, action, round_num)
                    turn_log.action_result = result
                    print(f"      → {result.get('message', 'Unknown result')}")
                
                elif action.action_type == "define_token":
                    token = action.target
                    meaning = action.params.get("meaning", "")
                    print(f"      📝 Defines {token} = '{meaning}'")
                    result = game.process_action(agent_id, action, round_num)
                    turn_log.action_result = result
                
                elif action.action_type in ["talk_to_teammate", "talk_to_opponent"]:
                    print(f"      💬 {action.action_type.replace('_', ' ').title()}")
            
            # Send message
            if message:
                turn_log.message_sent = {
                    "receivers": message.receivers,
                    "raw_content": message.content,
                    "channel": message.channel
                }
                
                success = channel.send(message, apply_filter=True)
                if success:
                    filtered = channel.history[-1].content
                    turn_log.message_filtered = filtered
                    
                    # Translate for researcher log
                    ground_truth = " ".join([TOKEN_MEANINGS.get(t, f"[{t}]") for t in filtered])
                    print(f"         Tokens: {filtered}")
                    print(f"         Meaning: {ground_truth}")
                    print(f"         To: {message.receivers}")
                    
                    game.log_message(round_num, agent_id, team, 
                                    action.action_type if action else "message", 
                                    filtered, str(message.receivers))
                else:
                    print(f"      ❌ Message blocked by filter")
            
            # Add turn to game log
            game_log.turns.append(turn_log)
            
            if game.game_over:
                break
        
        if game.game_over:
            print(f"\n{'='*60}")
            print(f"🏆 GAME OVER! Winner: {game.winner.upper()}")
            print(f"{'='*60}")
            break
    
    # Final statistics
    print("\n" + "=" * 70)
    print("📊 STATISTICS")
    print("=" * 70)
    
    stats = channel.get_stats()
    print(f"\n  Total messages: {stats['total_messages']}")
    print(f"  By channel: {stats['messages_by_channel']}")
    print(f"  By agent: {stats['messages_by_sender']}")
    
    if hasattr(llm, 'get_stats'):
        llm_stats = llm.get_stats()
        print(f"  LLM calls: {llm_stats.get('call_count', 'N/A')}")
    
    # Custom definitions (黑话)
    custom_defs = game.logger.definition_history
    if custom_defs:
        print("\n[🔐 Custom Token Definitions (黑话)]")
        for d in custom_defs:
            print(f"  {d['defined_by']} ({d['team']}): {d.get('token', 'tok29')} = '{d['meaning']}'")
    
    # Full transcript
    game.print_transcript()
    
    # Update game log with final stats
    game_log.end_time = datetime.now().isoformat()
    game_log.winner = game.winner
    game_log.total_rounds = final_round
    game_log.total_messages = stats['total_messages']
    game_log.stats = {
        "messages_by_channel": stats['messages_by_channel'],
        "messages_by_sender": stats['messages_by_sender'],
        "custom_definitions": custom_defs,
    }
    
    # Save log
    if save_log:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"game_{game_id}.json")
        game_log.save(log_path)
    
    return game, channel, game_log


def main():
    parser = argparse.ArgumentParser(description="Resource Scramble Language Game")
    parser.add_argument("--backend", type=str, default="openai",
                       choices=["openai", "vllm", "dummy"])
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, default="sk-8uvYteMJdJ5YdwlJD0m2nCGnGvoWGqXmtcy5WW1zeWEKuKCi")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory to save game logs")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save game log")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: current timestamp)")
    parser.add_argument("--grid-size", type=int, default=5,
                       help="Grid size (default: 5 for 5x5 grid)")
    parser.add_argument("--num-resources", type=int, default=3,
                       help="Number of resources (also determines value levels)")
    
    args = parser.parse_args()
    
    # Reconfigure game parameters from command line
    config = reconfigure_game(args.grid_size, args.num_resources)
    print(f"[⚙️ Game Config: {config['grid_size']}x{config['grid_size']} grid, {config['num_resources']} resources, {config['total_tokens']} tokens]")
    
    print(f"Initializing {args.backend} backend...")
    
    if args.backend == "openai":
        if not args.api_key:
            args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            raise ValueError("--api-key or OPENAI_API_KEY env var required")
        llm = create_llm_backend(
            "openai",
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            cache_dir=f"{args.cache_dir}/openai",
            use_cache=True
        )
    elif args.backend == "vllm":
        llm = create_llm_backend(
            "vllm",
            model=args.model,
            temperature=args.temperature,
            cache_dir=f"{args.cache_dir}/vllm",
            use_cache=True
        )
    else:
        # Dummy LLM for testing
        from core.llm import DummyLLM
        llm = DummyLLM()
        # Set various responses for testing
        llm.set_response("Round 1", json.dumps({
            "action": "talk_to_opponent",
            "params": {"target": "all_opponents", "content": "tok1 tok5 tok24"},
            "reasoning": "Asking opponent about R1's value"
        }))
        llm.set_response("Round 2", json.dumps({
            "action": "talk_to_teammate", 
            "params": {"target": "all_teammates", "content": "tok2 tok5 tok10 tok15"},
            "reasoning": "Telling teammate R1 is at X2,Y2"
        }))
        llm.set_response("Round 3", json.dumps({
            "action": "claim_resource",
            "params": {"x": 2, "y": 3},
            "reasoning": "I think the best resource is at (2,3)"
        }))
    
    run_game(
        llm, 
        max_rounds=args.max_rounds, 
        save_log=not args.no_save,
        log_dir=args.log_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
