"""
Resource Scramble Simulation Runner

Main simulation orchestrator for pidgin language emergence experiments.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from resource_scramble import (
    ResourceScrambleEnvironment,
    AbstractVocabulary,
    CurriculumManager,
    GameRound,
    EpisodeResult
)
from llm_backend import DeepSeekBackend, PromptBuilder, MessageParser
from language_analysis import LanguageAnalyzer, LanguageMetrics


class ResourceScrambleSimulation:
    """
    Main simulation orchestrator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulation.
        
        Config should contain:
        - api_key: DeepSeek API key
        - vocabulary_size: Size of token vocabulary (default 20)
        - max_message_length: Max tokens per message (default 5)
        - max_rounds: Max rounds per episode (default 10)
        - num_episodes: Total episodes to run (default 100)
        - output_dir: Where to save results
        """
        self.config = config
        
        # Initialize components
        self.llm = DeepSeekBackend(
            api_key=config.get('api_key'),
            model=config.get('model', 'deepseek-chat')
        )
        
        self.vocabulary = AbstractVocabulary(
            size=config.get('vocabulary_size', 20),
            max_length=config.get('max_message_length', 5)
        )
        
        self.curriculum = CurriculumManager()
        
        # Initialize team vocabularies (different initial subsets)
        vocab_tokens = self.vocabulary.tokens
        self.team_a_vocab = vocab_tokens[:15]  # First 15 tokens
        self.team_b_vocab = vocab_tokens[5:]   # Last 15 tokens (overlap of 10)
        
        self.analyzer = LanguageAnalyzer(vocab_tokens)
        
        # Track recent messages to penalize repetition
        self.recent_team_a_messages = []
        self.recent_team_b_messages = []
        self.repetition_window = 5  # Check last 5 messages
        
        # Statistics
        self.episode_results: List[EpisodeResult] = []
        self.total_episodes_run = 0
        
        # Output
        self.output_dir = Path(config.get('output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Run the complete simulation."""
        print("=" * 60)
        print("Resource Scramble Simulation - Language Emergence Study")
        print("=" * 60)
        print(f"\nVocabulary size: {self.vocabulary.size}")
        print(f"Team A vocab: {', '.join(self.team_a_vocab[:5])}... ({len(self.team_a_vocab)} tokens)")
        print(f"Team B vocab: {', '.join(self.team_b_vocab[:5])}... ({len(self.team_b_vocab)} tokens)")
        print(f"Shared tokens: {len(set(self.team_a_vocab) & set(self.team_b_vocab))}")
        print()
        
        max_episodes = self.config.get('num_episodes', 100)
        
        while self.total_episodes_run < max_episodes:
            phase = self.curriculum.get_current_phase()
            print(f"\n--- Curriculum Phase {phase.phase_id} ---")
            print(f"Resources: {phase.num_resources}, "
                  f"Threshold: {phase.success_threshold}, "
                  f"Min episodes: {phase.min_episodes}")
            
            # Run episodes in current phase
            phase_episode_count = 0
            while True:
                if self.total_episodes_run >= max_episodes:
                    break
                
                # Run one episode
                result = self.run_episode()
                self.episode_results.append(result)
                self.curriculum.record_episode(result)
                self.total_episodes_run += 1
                phase_episode_count += 1
                
                # Print progress
                if phase_episode_count % 5 == 0:
                    recent = self.episode_results[-5:]
                    avg_rounds = sum(len(r.rounds) for r in recent) / len(recent)
                    success_rate = sum(1 for r in recent if r.winner != "tie") / len(recent)
                    
                    print(f"  Episode {self.total_episodes_run}: "
                          f"Avg rounds: {avg_rounds:.1f}, "
                          f"Success rate: {success_rate:.2f}")
                
                # Check if should advance
                if self.curriculum.should_advance():
                    print(f"\n✓ Phase {phase.phase_id} completed! Advancing...")
                    if not self.curriculum.advance_phase():
                        print("  Reached final phase.")
                    break
                
                if phase_episode_count >= phase.min_episodes * 2:
                    # Force advance if stuck
                    print(f"\n  Max episodes for phase reached. Forcing advance...")
                    if not self.curriculum.advance_phase():
                        break
                    break
        
        # Final analysis
        print("\n" + "=" * 60)
        print("Simulation Complete - Analyzing Results")
        print("=" * 60)
        
        self.analyze_results()
        self.export_results()
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def run_episode(self) -> EpisodeResult:
        """Run one episode."""
        phase = self.curriculum.get_current_phase()
        env = ResourceScrambleEnvironment(phase, seed=None)  # Random each time
        env.reset()
        
        result = EpisodeResult(episode_id=self.total_episodes_run)
        
        # Get views
        team_a_view = env.get_team_a_view()
        team_b_view = env.get_team_b_view()
        
        conversation_history = []
        max_rounds = self.config.get('max_rounds', 10)
        min_rounds = 3  # Force at least 3 rounds of communication
        
        for round_num in range(max_rounds):
            # Team A's turn
            team_a_response = self.get_agent_response(
                team="A",
                location_data=team_a_view,
                attribute_data=None,
                opponent_message=None,
                conversation_history=conversation_history,
                round_num=round_num + 1
            )
            
            # Team B's turn
            team_b_response = self.get_agent_response(
                team="B",
                location_data=None,
                attribute_data=team_b_view,
                opponent_message=team_a_response['message'],
                conversation_history=conversation_history,
                round_num=round_num + 1
            )
            
            # Evaluate choices
            team_a_reward, team_a_correct = env.evaluate_choice(team_a_response['choice'])
            team_b_reward, team_b_correct = env.evaluate_choice(team_b_response['choice'])
            
            # Penalize repetitive messages
            repetition_penalty_a = self._calculate_repetition_penalty(
                team_a_response['message'], self.recent_team_a_messages
            )
            repetition_penalty_b = self._calculate_repetition_penalty(
                team_b_response['message'], self.recent_team_b_messages
            )
            
            team_a_reward -= repetition_penalty_a
            team_b_reward -= repetition_penalty_b
            
            # Track messages
            self.recent_team_a_messages.append(team_a_response['message'])
            self.recent_team_b_messages.append(team_b_response['message'])
            if len(self.recent_team_a_messages) > self.repetition_window:
                self.recent_team_a_messages.pop(0)
            if len(self.recent_team_b_messages) > self.repetition_window:
                self.recent_team_b_messages.pop(0)
            
            # Exploration bonus for novel messages
            if round_num > 2:  # After initial rounds
                if repetition_penalty_a == 0:  # Novel message
                    team_a_reward += 2
                if repetition_penalty_b == 0:
                    team_b_reward += 2
            
            # Determine winner (zero-sum)
            if team_a_correct and not team_b_correct:
                winner = "A"
                team_a_reward += 10
                team_b_reward -= 10
            elif team_b_correct and not team_a_correct:
                winner = "B"
                team_b_reward += 10
                team_a_reward -= 10
            elif team_a_correct and team_b_correct:
                # Both found good resources, compare values
                if team_a_reward > team_b_reward:
                    winner = "A"
                elif team_b_reward > team_a_reward:
                    winner = "B"
                else:
                    winner = "tie"
            else:
                winner = "tie"
                # Neither correct - penalize both
                team_a_reward -= 3
                team_b_reward -= 3
            
            # Record round
            round_data = GameRound(
                round_id=round_num,
                team_a_message=team_a_response['message'],
                team_b_message=team_b_response['message'],
                team_a_choice=team_a_response['choice'],
                team_b_choice=team_b_response['choice'],
                team_a_reward=team_a_reward,
                team_b_reward=team_b_reward,
                winner=winner
            )
            result.add_round(round_data)
            
            # Update conversation history
            conversation_history.append(
                f"A: {team_a_response['message']} -> chose {team_a_response['choice']}, "
                f"B: {team_b_response['message']} -> chose {team_b_response['choice']}"
            )
            
            # Record for language analysis
            optimal_id = env.get_optimal_choice()
            context = {
                "resource_id": optimal_id,
                "resource_attributes": team_b_view[optimal_id] if optimal_id < len(team_b_view) else {}
            }
            
            self.analyzer.record_communication(
                team_a_msg=team_a_response['message'],
                team_b_msg=team_b_response['message'],
                team_a_success=team_a_correct,
                team_b_success=team_b_correct,
                context=context
            )
            
            # Check for convergence (only after minimum rounds)
            if round_num >= min_rounds - 1:
                if team_a_correct or team_b_correct:
                    if result.converged_round < 0:
                        result.converged_round = round_num
                    break
        
        result.finalize()
        return result
    
    def _calculate_repetition_penalty(self, message: str, recent_messages: List[str]) -> int:
        """Calculate penalty for repetitive messages."""
        if not recent_messages:
            return 0
        
        # Exact match with any recent message
        if message in recent_messages:
            # More severe penalty for more recent repetition
            most_recent_idx = len(recent_messages) - 1 - recent_messages[::-1].index(message)
            recency_factor = (len(recent_messages) - most_recent_idx)
            return 3 * recency_factor
        
        # Partial overlap penalty
        msg_tokens = set(message.split(', '))
        max_overlap = 0
        for recent_msg in recent_messages:
            recent_tokens = set(recent_msg.split(', '))
            overlap = len(msg_tokens & recent_tokens) / max(len(msg_tokens), 1)
            max_overlap = max(max_overlap, overlap)
        
        if max_overlap > 0.8:  # >80% token overlap
            return 2
        
        return 0
    
    def get_agent_response(self,
                          team: str,
                          location_data: Optional[List[Dict[str, Any]]],
                          attribute_data: Optional[List[Dict[str, Any]]],
                          opponent_message: Optional[str],
                          conversation_history: List[str],
                          round_num: int) -> Dict[str, Any]:
        """
        Get agent response via LLM.
        
        Returns:
            Dict with 'message' and 'choice'
        """
        if team == "A":
            vocab = self.team_a_vocab
            messages = PromptBuilder.build_team_a_prompt(
                agent_id="A1",
                vocabulary=vocab,
                location_data=location_data or [],
                team_b_message=opponent_message,
                conversation_history=conversation_history,
                round_num=round_num
            )
        else:
            vocab = self.team_b_vocab
            messages = PromptBuilder.build_team_b_prompt(
                agent_id="B1",
                vocabulary=vocab,
                attribute_data=attribute_data or [],
                team_a_message=opponent_message,
                conversation_history=conversation_history,
                round_num=round_num
            )
        
        # Query LLM with retry
        # Dynamic temperature: increase if agent is stuck in repetitive pattern
        base_temp = 0.7
        recent_messages = self.recent_team_a_messages if team == "A" else self.recent_team_b_messages
        
        # If last 3 messages are very similar, increase temperature to encourage exploration
        if len(recent_messages) >= 3:
            unique_recent = len(set(recent_messages[-3:]))
            if unique_recent <= 1:  # All same
                base_temp = 1.2  # Much higher temperature
            elif unique_recent == 2:  # Mostly same
                base_temp = 0.95
        
        max_attempts = 3
        for attempt in range(max_attempts):
            response = self.llm.query_with_retry(
                messages=messages,
                temperature=base_temp,
                max_tokens=150
            )
            
            # Parse and validate
            parsed = MessageParser.parse_response(response, vocab)
            
            if parsed['valid']:
                return {
                    'message': parsed['message'],
                    'choice': parsed['choice']
                }
            
            # Invalid response, retry with lower temperature
            if attempt < max_attempts - 1:
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                messages.append({
                    "role": "user",
                    "content": f"ERROR: {parsed['error']}. Please follow the format exactly "
                               f"and use ONLY tokens from: {', '.join(vocab)}"
                })
        
        # Failed all attempts, return random
        import random
        return {
            'message': ', '.join(random.sample(vocab, min(3, len(vocab)))),
            'choice': random.randint(0, 3)
        }
    
    def analyze_results(self):
        """Analyze and print results."""
        print("\n--- Language Emergence Analysis ---\n")
        
        # Calculate metrics
        metrics = self.analyzer.calculate_metrics()
        
        print(f"Cross-team Mutual Information: {metrics.cross_team_mutual_information:.4f}")
        print(f"Compositionality Score: {metrics.compositionality_score:.4f}")
        print(f"Symbol Reuse Rate: {metrics.symbol_reuse_rate:.4f}")
        print(f"Shared Vocabulary Size: {metrics.shared_vocabulary_size}")
        print(f"Team A Unique Symbols: {metrics.team_a_unique_symbols}")
        print(f"Team B Unique Symbols: {metrics.team_b_unique_symbols}")
        print(f"Communication Efficiency: {metrics.communication_efficiency:.4f}")
        
        # Pidgin detection
        print("\n--- Pidgin Detection ---\n")
        pidgin = self.analyzer.detect_pidgin()
        
        if pidgin['detected']:
            print("✓ PIDGIN LANGUAGE DETECTED!")
            print(f"  Shared vocabulary size: {pidgin['shared_vocabulary_size']}")
            print(f"  Stable pidgin tokens: {', '.join(pidgin['stable_pidgin_tokens'])}")
            print("\n  Pidgin semantics:")
            for token, semantics in pidgin['pidgin_semantics'].items():
                print(f"    {token}:")
                for concept, prob in semantics:
                    print(f"      {concept}: {prob:.2f}")
        else:
            print("✗ No stable pidgin detected yet")
            print(f"  Reason: {pidgin.get('reason', 'Insufficient convergence')}")
        
        # Performance stats
        print("\n--- Performance Statistics ---\n")
        
        if self.episode_results:
            successes = sum(1 for r in self.episode_results if r.winner != "tie")
            success_rate = successes / len(self.episode_results)
            avg_rounds = sum(len(r.rounds) for r in self.episode_results) / len(self.episode_results)
            
            print(f"Total episodes: {len(self.episode_results)}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Average rounds per episode: {avg_rounds:.2f}")
            
            # Curriculum progress
            print("\n--- Curriculum Progress ---")
            for phase_id, stats in self.curriculum.get_statistics().items():
                print(f"  {phase_id}: {stats['episodes']} episodes, "
                      f"{stats['success_rate']:.2%} success, "
                      f"{stats['avg_rounds']:.1f} avg rounds")
        
        # LLM usage
        print("\n--- LLM Usage ---")
        llm_stats = self.llm.get_statistics()
        print(f"Total API requests: {llm_stats['request_count']}")
        print(f"Total tokens used: {llm_stats['total_tokens']}")
    
    def export_results(self):
        """Export results to files."""
        # Save metrics
        metrics = self.analyzer.calculate_metrics()
        metrics_dict = {
            "cross_team_mi": metrics.cross_team_mutual_information,
            "compositionality": metrics.compositionality_score,
            "symbol_reuse": metrics.symbol_reuse_rate,
            "shared_vocab_size": metrics.shared_vocabulary_size,
            "team_a_unique": metrics.team_a_unique_symbols,
            "team_b_unique": metrics.team_b_unique_symbols,
            "efficiency": metrics.communication_efficiency
        }
        
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save pidgin analysis
        pidgin = self.analyzer.detect_pidgin()
        with open(self.output_dir / "pidgin_analysis.json", 'w') as f:
            json.dump(pidgin, f, indent=2, default=str)
        
        # Save episode results
        episodes_data = []
        for result in self.episode_results:
            episodes_data.append({
                "episode_id": result.episode_id,
                "winner": result.winner,
                "num_rounds": len(result.rounds),
                "team_a_reward": result.total_team_a_reward,
                "team_b_reward": result.total_team_b_reward,
                "converged_round": result.converged_round
            })
        
        with open(self.output_dir / "episodes.json", 'w') as f:
            json.dump(episodes_data, f, indent=2)
        
        # Save communication transcripts
        with open(self.output_dir / "communications.txt", 'w') as f:
            for i, result in enumerate(self.episode_results[-20:]):  # Last 20 episodes
                f.write(f"\n=== Episode {result.episode_id} ===\n")
                for round_data in result.rounds:
                    f.write(f"Round {round_data.round_id + 1}:\n")
                    f.write(f"  A: {round_data.team_a_message} -> {round_data.team_a_choice}\n")
                    f.write(f"  B: {round_data.team_b_message} -> {round_data.team_b_choice}\n")
                    f.write(f"  Winner: {round_data.winner}\n")
        
        # Save semantics
        semantics = self.analyzer.semantics.get_all_semantics()
        semantics_readable = {
            token: [(concept, f"{prob:.3f}") for concept, prob in meanings]
            for token, meanings in semantics.items()
            if meanings
        }
        
        with open(self.output_dir / "token_semantics.json", 'w') as f:
            json.dump(semantics_readable, f, indent=2)
