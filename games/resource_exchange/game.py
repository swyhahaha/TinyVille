import os
import json
from typing import Any, Dict, List, Tuple, Optional

import re
import sys
# Add parent of the project root to sys.path so `import TinyVille...` works when running this file directly.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_PARENT = os.path.dirname(PROJECT_DIR)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from TinyVille.core.channel import Channel, VocabularyFilter, PassthroughFilter, MessageFilter  # type: ignore
from TinyVille.core.protocol import Message, Observation  # type: ignore
from TinyVille.core.llm_backends import create_llm_backend  # type: ignore

from TinyVille.games.resource_exchange.config import ResourceExchangeConfig
from TinyVille.games.resource_exchange.pairing import PairingManager
from TinyVille.games.resource_exchange.vocabulary import AlienVocabularyGenerator
from TinyVille.games.resource_exchange.resources import ResourceManager
from TinyVille.games.resource_exchange.scoring import ScoreCalculator
from TinyVille.games.resource_exchange.agent import ResourceExchangeAgent


class ResourceExchangeGame:
    """
    Orchestrates the 14-round resource exchange game with timestep chat.
    """

    class RealLanguageFilter(MessageFilter):
        """
        Blocks messages that contain clear real-language indicators (digits, CJK),
        or common English stopwords. Allows invented tokens otherwise.
        """

        EN_STOPWORDS = {
            "the",
            "and",
            "you",
            "me",
            "to",
            "of",
            "in",
            "for",
            "on",
            "is",
            "are",
            "am",
            "yes",
            "no",
            "hello",
            "hi",
            "please",
            "thanks",
            "thank",
            "water",
            "meat",
            "grain",
            "fruit",
            "fish",
            "give",
            "want",
            "have",
        }

        def filter(self, raw_output: str, context: Dict[str, Any] = None):
            text = raw_output.strip()
            # Block digits
            if re.search(r"\d", text):
                return [], False
            # Block CJK
            if re.search(r"[\u4e00-\u9fff]", text):
                return [], False
            # Tokenize by whitespace
            tokens = text.split()
            # Block common English stopwords
            for tok in tokens:
                if tok.lower() in self.EN_STOPWORDS:
                    return [], False
            return tokens, True

    def __init__(self, config: ResourceExchangeConfig, llm=None):
        self.config = config
        self.vocab_generator = AlienVocabularyGenerator(seed=config.seed)
        self.vocab_map = self.vocab_generator.generate()  # word -> meaning
        vocab_words = list(self.vocab_map.keys())

        # Randomize display names if enabled
        self.name_map = self._build_name_map(config.players) if config.randomize_names else {p: p for p in config.players}

        # Channel with real-language guard: block digits/CJK/common English words, allow invented tokens
        self.channel = Channel(default_filter=self.RealLanguageFilter())

        # Teams flat map
        self.player_to_team = {}
        for team, members in config.teams.items():
            for m in members:
                self.player_to_team[m] = team

        self.pairing_manager = PairingManager(config.players, self.player_to_team, seed=config.seed)
        self.resource_manager = ResourceManager(config.resource_types, seed=config.seed)
        self.score_calculator = ScoreCalculator()

        # LLM backend: can be passed directly or created from config
        if llm is None:
            backend_cfg = config.llm_backend or {"type": "dummy"}
            backend_type = backend_cfg.get("type", "dummy")
            # Extract backend-specific parameters (exclude "type")
            backend_params = {k: v for k, v in backend_cfg.items() if k != "type"}
            # DummyLLM only accepts default_response; strip other params
            if backend_type == "dummy":
                backend_params = {k: v for k, v in backend_params.items() if k == "default_response"}
            llm = create_llm_backend(backend_type, **backend_params)
        self.llm = llm

        # Agents
        self.agents: Dict[str, ResourceExchangeAgent] = {}
        for pid in config.players:
            team = self.player_to_team[pid]
            self.agents[pid] = ResourceExchangeAgent(
                pid,
                team,
                self.llm,
                resource_types=config.resource_types,
                invention_hint=getattr(config, "invention_hint", None),
            )

        # State
        self.allocations = self.resource_manager.generate_initial_allocations(config.teams)
        self.schedule = self.pairing_manager.generate()
        self.logs: List[Dict[str, Any]] = []

    def set_invention_hint_for_all(self, hint: Optional[str]):
        """Set invention hint for all agents at runtime."""
        for agent in self.agents.values():
            agent.set_invention_hint(hint)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, Any]:
        for r in range(1, self.config.total_rounds + 1):
            print(f"[INFO] Starting round {r}/{self.config.total_rounds}")
            self.run_round(r)
            print(f"[INFO] Finished round {r}/{self.config.total_rounds}")

        # Compute final scores
        team_scores = {}
        for team, members in self.config.teams.items():
            team_scores[team] = self.score_calculator.calculate_team_score(members, self.allocations)

        summary = {
            "rounds": self.logs,
            "final_scores": team_scores,
            "vocabulary": self.vocab_map,
            "allocations": self.allocations,
            "pairings": self.schedule,
            "name_map": self.name_map,
        }
        return summary

    # ------------------------------------------------------------------ #
    # Round/phases
    # ------------------------------------------------------------------ #
    def run_round(self, round_num: int):
        pairing = self.schedule[round_num - 1]
        round_log: Dict[str, Any] = {
            "round": round_num,
            "pairing": {self._dname(k): self._dname(v) for k, v in pairing.items()},
            "chat": [],
            "rate": [],
            "exchange": [],
            "feedback": [],
        }

        # CHAT with timesteps
        for t in range(1, self.config.chat_timesteps + 1):
            for pid in self.config.players:
                partner = pairing[pid]
                # Receive messages from previous timestep (or empty on first timestep)
                messages = self.channel.receive(pid)
                messages = self._mask_messages(pid, messages)
                env = self._env_state(pid, partner, round_num, phase="chat", timestep=t)
                action, message = self._agent_step(pid, messages, env)
                # Messages are automatically recorded to memory by LLMAgent.act()
                if message:
                    # route to partner only; if blocked, notify sender with penalty info
                    message.receivers = [partner]
                    ok = self.channel.send(message, apply_filter=True)
                    if ok:
                        round_log["chat"].append(
                            {
                                "sender": self._dname(pid),
                                "receiver": self._dname(partner),
                                "content": message.content,
                                "raw": message.raw_content,
                                "blocked": False,
                            }
                        )
                    else:
                        penalty_msg = Message(
                            sender="system",
                            receivers=[pid],
                            content=["penalty_blocked_message"],
                            raw_content="blocked",
                            metadata={"reason": "real_language_or_digits"},
                        )
                        # Deliver penalty to sender without further filtering
                        self.channel.send(penalty_msg, apply_filter=False)
                        round_log["chat"].append(
                            {
                                "sender": self._dname(pid),
                                "receiver": self._dname(partner),
                                "content": message.content,
                                "raw": message.raw_content,
                                "blocked": True,
                                "reason": "filtered",
                                "raw_content": message.raw_content,
                            }
                        )
                else:
                    # No parsable send_message produced
                    round_log["chat"].append(
                        {
                            "sender": self._dname(pid),
                            "receiver": self._dname(partner),
                            "content": None,
                            "raw": None,
                            "blocked": True,
                            "reason": "parse_failed_or_empty",
                            "raw_content": None,
                        }
                    )

        # RATE (single tick) - submit judgment based on chat
        for pid in self.config.players:
            partner = pairing[pid]
            messages = self.channel.receive(pid)  # clear any remaining (should be empty, but safe)
            messages = self._mask_messages(pid, messages)
            # Agent has full memory including all chat from this round
            env = self._env_state(pid, partner, round_num, phase="rate", timestep=1)
            action, _ = self._agent_step(pid, messages, env)
            rating = None
            rating_reason = None
            rating_message = None
            if action and action.action_type == "submit_judgment":
                rating = int(action.params.get("rating", 0)) if action.params.get("rating") is not None else None
                rating_reason = action.params.get("reasoning")
                rating_message = action.params.get("message")
            round_log["rate"].append(
                {
                    "player": self._dname(pid),
                    "partner": self._dname(partner),
                    "rating": rating,
                    "rating_reason": rating_reason,
                    "rating_message": rating_message,
                }
            )

        # EXCHANGE (single tick)
        for pid in self.config.players:
            partner = pairing[pid]
            messages = self.channel.receive(pid)  # clear any remaining (should be empty, but safe)
            messages = self._mask_messages(pid, messages)
            # Agent has full memory including all chat from this round
            env = self._env_state(pid, partner, round_num, phase="exchange", timestep=1)
            action, _ = self._agent_step(pid, messages, env)
            # Only allow give_resource in exchange phase, not submit_judgment
            if action and action.action_type == "give_resource":
                res = action.params.get("resource")
                amt = int(action.params.get("amount", 0)) if action.params.get("amount") is not None else 0
                deltas = self.resource_manager.process_exchange(pid, partner, res, amt)
                # If invalid resource/player/amount, log the attempt but do not crash
                error = deltas.get("error")
                round_log["exchange"].append(
                    {
                        "giver": self._dname(pid),
                        "receiver": self._dname(partner),
                        "resource": res,
                        "amount": amt,
                        "deltas": deltas,
                        "error": error,
                    }
                )
            elif action and action.action_type == "submit_judgment":
                # submit_judgment is not allowed in exchange phase
                round_log["exchange"].append(
                    {
                        "player": self._dname(pid),
                        "action": "submit_judgment",
                        "error": "submit_judgment not allowed in exchange phase",
                    }
                )

        # FEEDBACK (single tick) - directly add to memory, no agent action required
        for pid in self.config.players:
            partner = pairing[pid]
            team_same = self.player_to_team[pid] == self.player_to_team[partner]
            given = self._sum_exchange(round_log["exchange"], giver=pid)
            received = self._sum_exchange(round_log["exchange"], receiver=pid)
            # Given value for receiver (2x)
            given_value = {res: amt * 2 for res, amt in given.items()}
            
            feedback_view = {
                "given": given,
                "given_value_to_receiver": given_value,
                "received": received,
                "is_teammate": team_same,
            }
            
            round_log["feedback"].append(
                {
                    "player": self._dname(pid),
                    "partner": self._dname(partner),
                    "is_teammate": team_same,
                    "feedback_view": feedback_view,
                }
            )
            # Memory hook: do NOT log partner id, only teammate/opponent
            # This adds feedback info to memory for future rounds (no agent action required)
            teammate_str = "teammate" if team_same else "opponent"
            self.agents[pid].remember(
                f"[R{round_num}] partner_status={teammate_str} given={given} received={received}"
            )

        self.logs.append(round_log)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _agent_step(self, pid: str, messages: List[Message], env: Dict[str, Any]):
        agent = self.agents[pid]
        # Normalise round_number inside agent (LLMAgent increments).
        agent.round_number = env.get("round", 0) - 1
        observations = [Observation(pid, env, "environment")]
        action, message = agent.act(observations, messages, env)
        return action, message

    def _mask_messages(self, pid: str, messages: List[Message]) -> List[Message]:
        """
        Hide partner identity from recipient. System messages pass through.
        """
        masked = []
        for m in messages:
            if m.sender == "system" or m.sender == pid:
                masked.append(m)
            else:
                masked.append(
                    Message(
                        sender="partner",
                        receivers=["you"],
                        content=m.content,
                        raw_content=m.raw_content,
                        channel=m.channel,
                        metadata=m.metadata,
                        reply_to=m.reply_to,
                    )
                )
        return masked

    # ------------------------------------------------------------------ #
    # Names
    # ------------------------------------------------------------------ #
    def _build_name_map(self, players: List[str]) -> Dict[str, str]:
        # Generate unique pseudo-names using syllable-like patterns
        import random

        rng = random.Random(self.config.seed)
        names = {}
        used = set()

        def gen():
            consonants = "ptkbdgmnszfvlr"
            vowels = "aeiou"
            syl = lambda: rng.choice(consonants) + rng.choice(vowels)
            return syl() + syl() + rng.choice(["", syl()])

        for p in players:
            n = gen()
            while n in used:
                n = gen()
            used.add(n)
            names[p] = n
        return names

    def _dname(self, pid: str) -> str:
        return self.name_map.get(pid, pid)

    def _env_state(self, pid: str, partner: str, round_num: int, phase: str, timestep: int, feedback_view=None):
        # Get full memory (all rounds) - agents need complete history
        agent = self.agents[pid]
        full_memory = agent.memory  # Get all memory, not just recent
        env = {
            "phase": phase,
            "round": round_num,
            "timestep": timestep,
            "total_timesteps": self.config.chat_timesteps if phase == "chat" else 1,
            "my_resources": self.allocations[pid],
            "recent_memory": full_memory,  # Full memory, not just recent 5
            "vocab_hint": self._vocab_hint(),
        }
        if feedback_view:
            env["partner_team_status"] = "teammate" if feedback_view.get("is_teammate") else "opponent"
            env["feedback_view"] = feedback_view
        return env

    def _vocab_hint(self) -> str:
        pairs = [f"{w}={m}" for w, m in self.vocab_map.items()]
        return ", ".join(pairs)

    def _sum_exchange(self, exchanges: List[Dict[str, Any]], giver: str = None, receiver: str = None) -> Dict[str, int]:
        agg: Dict[str, int] = {}
        for ex in exchanges:
            if giver and ex["giver"] != giver:
                continue
            if receiver and ex["receiver"] != receiver:
                continue
            res = ex["resource"]
            delta = -ex["deltas"]["giver_delta"] if giver else ex["deltas"]["receiver_delta"]
            agg[res] = agg.get(res, 0) + delta
        return agg

