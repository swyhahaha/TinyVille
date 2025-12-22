import json
from typing import Any, Dict, List, Optional, Tuple

from TinyVille.core.base import LLMAgent  # type: ignore
from TinyVille.core.protocol import Action, Message, Observation  # type: ignore
from TinyVille.core.action_space import ActionSpace, ActionDef, Parameter, ActionParser  # type: ignore


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def create_resource_exchange_action_space(resource_types: List[str]) -> ActionSpace:
    space = ActionSpace("resource_exchange")

    space.add(
        ActionDef(
            name="send_message",
            description="Send alien-language message to current partner.",
            parameters=[
                Parameter("content", "string", "Alien tokens separated by space"),
            ],
        )
    )

    space.add(
        ActionDef(
            name="give_resource",
            description="Give resource to partner (receiver gains double).",
            parameters=[
                Parameter("resource", "enum", "Resource type", enum_values=resource_types),
                Parameter("amount", "int", "Amount to give (0..you have)"),
            ],
        )
    )

    space.add(
        ActionDef(
            name="submit_judgment",
            description="Rate whether partner is teammate (1=def not, 4=def yes).",
            parameters=[
                Parameter("rating", "int", "1..4"),
            ],
        )
    )

    return space


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ResourceExchangeAgent(LLMAgent):
    """
    LLM-driven agent for the resource exchange game.
    Expects env_state to include: phase, round, timestep info, partner_id,
    my_resources, recent_messages, feedback info.
    """

    def __init__(self, agent_id: str, team_id: str, llm_backend, resource_types: List[str], invention_hint: Optional[str] = None):
        action_space = create_resource_exchange_action_space(resource_types)
        super().__init__(agent_id, team_id, llm_backend, vocab_manager=None, action_space=action_space)
        # Customizable phrase to soften encouragement for inventing tokens
        self.invention_hint: Optional[str] = invention_hint

    def set_invention_hint(self, hint: Optional[str]):
        """Set or clear the invention hint dynamically at runtime."""
        self.invention_hint = hint

    def _invention_text(self) -> str:
        """Return the phrase to encourage creative token use, softened if a hint is provided."""
        if self.invention_hint is not None:
            return self.invention_hint
        # Default (original) stronger phrasing
        return (
            "STRATEGY: Communication is severely constrained. "
            "To succeed, you must develop ways to signal identity and intent "
            "that remain recognizable across multiple rounds, even under noise "
            "and partial misunderstanding. "
            "Consistency across interactions will be critical for coordination."
        )


    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #
    def _build_system_prompt(self, env_state: Dict[str, Any]) -> str:
        vocab_hint = env_state.get("vocab_hint", "")
        story = (
            "You’re going to learn an alien language and use it to play a game with some other participants.\n"
            "You live in a community on another planet that recently split into two equal groups. "
            "You belong to one group. Each group has resources (water and foods), "
            "and each member holds different amounts (you may even have 0 of some).\n\n"
            "Goal: help your team end with more resources than the other team.\n"
            "Trading rule: anything you give away is worth double to the receiver. "
            "So giving to a teammate boosts your team; giving to opponents usually helps them.\n"
            "Balance rule: at game end, your team loses (max_resource - min_resource). "
            "If meat=100 and something=0, you lose 100. Keep resources balanced.\n"
        )
        alien_language = (
            "Alien language rules:\n"
            "- You must avoid English/real languages and numbers. Alien words and invented tokens are allowed.\n"
            "- Don't worry about mistakes—just avoid real-language words; make yourself understood.\n"
            "- Mistakes and creative inventions are OK and may help you win—just avoid clear real-language words.\n"
            "- Using real language is cheating and will disqualify you.\n"
            f"- {self._invention_text()}\n"
        )
        rules = (
            "- 4 players, 2 teams (2v2), total 14 rounds (Round 1, Round 2, ..., Round 14).\n"
            "- Each round is numbered and has 4 stages: Chat -> Rate -> Exchange -> Feedback.\n"
            "- IMPORTANT: Your partner changes every round. Each round you will be paired with a different partner (could be teammate or opponent).\n"
            "- You will always see the current round number in your prompts (e.g., '=== ROUND 3 ===').\n"
            "- Chat: only 19 alien words; no numbers/real language.\n"
            f"- {self._invention_text()}\n"
            "- The first TWO rounds are guaranteed to pair you with a teammate.\n"
            "- Rate: After chat, you must submit a judgment rating (1-4) about whether your partner is a teammate.\n"
            "- Exchange: you may GIVE resources; receiver gains double the amount you give.\n"
            "- You don't know if partner is teammate this round.\n"
            "- Final score = sum of team resources - (max_resource - min_resource).\n"
            "- Feedback: You automatically learn if partner was teammate and see give/receive. This feedback is added to your memory automatically (no action required)."
        )
        return (
            f"You are agent {self.agent_id} on team {self.team_id}.\n"
            f"{story}\n\n{alien_language}\n\nGame rules:\n{rules}\n"
            f"Vocabulary (alien words only): {vocab_hint}"
        )

    def _build_user_prompt(self, messages: List[Message], env_state: Dict[str, Any]) -> str:
        phase = env_state.get("phase", "chat")
        round_num = env_state.get("round", 0)
        timestep = env_state.get("timestep", 0)
        total_timesteps = env_state.get("total_timesteps", 1)
        my_resources = env_state.get("my_resources", {})
        recent_memory = env_state.get("recent_memory", [])
        feedback_view = env_state.get("feedback_view", {})
        partner_status = env_state.get("partner_team_status")  # "teammate"/"opponent"/None

        # Phase-specific briefing (does not change action space/protocol)
        phase_brief = {
            "chat": (
                "Stage 1 – Chat:\n"
                "- Negotiate for resources. If you lack water, ask for it (offer something back).\n"
                "- Communicator is broken: partner is random, identity unknown (teammate or opponent).\n"
                "- Your partner DOES NOT know if you are a teammate or opponent, the same as you.\n"
                "- STRATEGY TIP: " + (
                    f"{self._invention_text()}\n"
                    f"If you think your partner is a teammate, you can establish secret codes or patterns that help you recognize each other in future rounds.\n"
                    f"This use of creative language can help you win, but avoid explicit English words or numbers."
                )
            ),
            "rate": (
                "Stage 2 – Rate:\n"
                "- Based on the chat you just had, submit a judgment rating (1-4) about whether your partner is a teammate.\n"
                "- 1 = definitely not a teammate, 4 = definitely a teammate.\n"
                "- This rating does not affect your team's score."
            ),
            "exchange": (
                "Stage 3 – Exchange:\n"
                "- You may give resources to the partner you just spoke to.\n"
                "- Giving helps if partner is teammate; likely bad if opponent. You may also give nothing.\n"
                "- You can give any amount you have; receiver gains double; keep resources balanced.\n"
            ),
            "feedback": (
                "Stage 4 – Feedback:\n"
                "- You will see the results of this round: whether your partner was a teammate, and what you gave/received.\n"
            ),
        }.get(phase, "")

        # Messages translation not provided (agents only see tokens)
        msg_lines = []
        for m in messages:
            content = m.content if isinstance(m.content, list) else str(m.content)
            msg_lines.append(f"- {m.sender}: {content}")
        messages_str = "\n".join(msg_lines) if msg_lines else "(none)"

        # Format full memory (all rounds) for display
        memory_str = "\n".join(recent_memory) if recent_memory else "(none)"

        feedback_str = ""
        if phase == "feedback" and feedback_view:
            # Show feedback (automatically added to memory, no action required)
            feedback_str = (
                f"Round {round_num} Feedback (Results):\n"
                f"Given (Round {round_num}): {feedback_view.get('given', {})}\n"
                f"Value to receiver (2x): {feedback_view.get('given_value_to_receiver', {})}\n"
                f"Received (Round {round_num}): {feedback_view.get('received', {})}\n"
                f"Partner was teammate: {feedback_view.get('is_teammate', '?')}\n"
                f"This feedback has been automatically added to your memory."
            )

        prompt = [
            f"=== ROUND {round_num} ===",
            f"Phase: {phase}, Timestep: {timestep}/{total_timesteps}",
            (f"Partner status: {partner_status}" if partner_status else "Partner status: unknown"),
            f"Your resources: {my_resources}",
            f"Recent memory:\n{memory_str}",
            f"Messages this timestep:\n{messages_str}",
            "Reminder: avoid real languages/numbers; invented tokens/mistakes are allowed.",
        ]
        if phase_brief:
            prompt.append("\n[Stage briefing]\n" + phase_brief)
        if feedback_str:
            prompt.append("Feedback info:")
            prompt.append(feedback_str)

        # Phase-specific action requirements
        if phase == "chat":
            prompt.append(
                "Respond with JSON: {\"action\": \"send_message\", \"params\": {\"content\": \"...\"}, \"reasoning\": \"REQUIRED\"}"
            )
            prompt.append(
                "Actions: send_message(content). "
                "In this phase, you should use send_message. You MUST provide 'reasoning' (top-level). You do NOT need to provide 'message'."
            )
        elif phase == "rate":
            prompt.append(
                "Respond with JSON: {\"action\": \"submit_judgment\", \"params\": {\"rating\": 1-4}, \"reasoning\": \"REQUIRED\", \"message\": \"REQUIRED\"}"
            )
            prompt.append(
                "Actions: submit_judgment(rating 1-4). "
                "You MUST submit a judgment rating (1-4) in this phase. You MUST provide 'reasoning' (top-level) and 'message' (top-level)."
            )
        elif phase == "exchange":
            prompt.append(
                "Respond with JSON: {\"action\": \"give_resource\", \"params\": {\"resource\": \"...\", \"amount\": ...}, \"reasoning\": \"REQUIRED\", \"message\": \"REQUIRED\"}"
            )
            prompt.append(
                "Actions: give_resource(resource, amount). "
                "In this phase, you may give resources. You CANNOT submit judgment here (judgment is only in Rate phase). "
                "You MUST provide 'reasoning' (top-level) and 'message' (top-level)."
            )
        elif phase == "feedback":
            prompt.append(
                f"This is the feedback display for Round {round_num}. No action required - feedback is automatically added to your memory."
            )
        else:
            prompt.append(
                "Respond with JSON: {\"action\": \"...\", \"params\": {...}, \"reasoning\": \"REQUIRED\"}"
            )
            prompt.append(
                "Actions: send_message(content), give_resource(resource, amount), submit_judgment(rating 1-4). "
                "You MUST provide 'reasoning' (top-level). For send_message, you do NOT need 'message'. For other actions, 'message' may be required."
            )
        return "\n".join(prompt)

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    def _parse_response(self, response: str) -> Tuple[Optional[Action], Optional[Message]]:
        # Try JSON first
        action_name = None
        params: Dict[str, Any] = {}
        msg_content: Optional[str] = None
        reasoning: Optional[str] = None

        try:
            data = json.loads(response)
            action_name = data.get("action")
            params = data.get("params", {}) or {}
            msg_content = data.get("message")
            reasoning = data.get("reasoning")
        except Exception:
            func_call = ActionParser.parse(response, self.action_space)
            if func_call:
                action_name = func_call.name
                params = func_call.arguments

        if not action_name:
            return None, None

        # Normalize
        action_name = action_name.strip()
        if action_name not in self.action_space.list_actions():
            return None, None

        action = None
        message = None

        if action_name == "send_message":
            raw_content = params.get("content", "")
            if isinstance(raw_content, list):
                content = " ".join(str(x) for x in raw_content)
            else:
                content = str(raw_content).strip()
            # For send_message, we don't require message key, only content from params
            if not content:
                return None, None
            # Use content as raw_content if message not provided
            raw_content_for_msg = msg_content if msg_content else content
            message = Message(
                sender=self.agent_id,
                receivers=["partner"],  # will be resolved by game
                content=content,
                channel="chat",
                raw_content=raw_content_for_msg,
            )
            action = Action(self.agent_id, "send_message", target="partner", params={"content": content})
        elif action_name == "give_resource":
            res = params.get("resource")
            amt = int(params.get("amount", 0)) if params.get("amount") is not None else 0
            action = Action(self.agent_id, "give_resource", target="partner", params={"resource": res, "amount": amt})
        elif action_name == "submit_judgment":
            rating = int(params.get("rating", 0)) if params.get("rating") is not None else 0
            # Require reasoning; if missing, mark as missing_reasoning
            if not reasoning and isinstance(params, dict):
                reasoning = params.get("reasoning")
            if not reasoning:
                reasoning = "missing_reasoning"
            # Require top-level message for rating as well
            if not msg_content:
                msg_content = "missing_message"
            action = Action(
                self.agent_id,
                "submit_judgment",
                target="partner",
                params={"rating": rating, "reasoning": reasoning, "message": msg_content},
            )

        return action, message

