"""
Language Analysis Metrics for SmallVille

Implements metrics to analyze language emergence:
- Cross-team mutual information
- Pidgin detection
- Compositionality measurement
- Symbol semantics tracking
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import math


@dataclass
class LanguageMetrics:
    """Container for language analysis metrics."""
    cross_team_mutual_information: float = 0.0
    compositionality_score: float = 0.0
    symbol_reuse_rate: float = 0.0
    shared_vocabulary_size: int = 0
    team_a_unique_symbols: int = 0
    team_b_unique_symbols: int = 0
    communication_efficiency: float = 0.0


class SymbolSemantics:
    """
    Tracks symbol-meaning associations over time.
    """
    
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        
        # token -> concept -> count
        self.token_concept_counts: Dict[str, Counter] = {
            token: Counter() for token in vocabulary
        }
        
        # Track contexts where tokens appear
        self.token_contexts: Dict[str, List[Dict[str, Any]]] = {
            token: [] for token in vocabulary
        }
    
    def record_usage(self, token: str, context: Dict[str, Any]):
        """
        Record a token usage with its context.
        
        Context should include:
        - resource_id: Which resource was being discussed
        - resource_attributes: Color, shape, value, etc.
        - outcome: Whether this led to success
        """
        if token not in self.vocabulary:
            return
        
        self.token_contexts[token].append(context)
        
        # Extract concepts from context
        if 'resource_attributes' in context:
            attrs = context['resource_attributes']
            for key, value in attrs.items():
                self.token_concept_counts[token][f"{key}:{value}"] += 1
    
    def get_token_semantics(self, token: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k concept associations for a token.
        
        Returns:
            List of (concept, probability) tuples
        """
        if token not in self.vocabulary:
            return []
        
        counts = self.token_concept_counts[token]
        if not counts:
            return []
        
        total = sum(counts.values())
        
        # Get top concepts
        top_concepts = counts.most_common(top_k)
        return [(concept, count / total) for concept, count in top_concepts]
    
    def get_all_semantics(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get semantics for all tokens."""
        return {
            token: self.get_token_semantics(token)
            for token in self.vocabulary
        }


class LanguageAnalyzer:
    """
    Analyzes language emergence patterns.
    """
    
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.semantics = SymbolSemantics(vocabulary)
        
        # Track all messages
        self.team_a_messages: List[str] = []
        self.team_b_messages: List[str] = []
        
        # Track message -> outcome associations
        self.message_outcomes: List[Tuple[str, str, bool]] = []  # (team, message, success)
        
    def record_communication(self, 
                            team_a_msg: str, 
                            team_b_msg: str,
                            team_a_success: bool,
                            team_b_success: bool,
                            context: Dict[str, Any]):
        """Record a communication exchange."""
        self.team_a_messages.append(team_a_msg)
        self.team_b_messages.append(team_b_msg)
        
        self.message_outcomes.append(("A", team_a_msg, team_a_success))
        self.message_outcomes.append(("B", team_b_msg, team_b_success))
        
        # Record token semantics
        for token in team_a_msg.split(','):
            token = token.strip()
            self.semantics.record_usage(token, context)
        
        for token in team_b_msg.split(','):
            token = token.strip()
            self.semantics.record_usage(token, context)
    
    def calculate_metrics(self) -> LanguageMetrics:
        """Calculate all language emergence metrics."""
        metrics = LanguageMetrics()
        
        metrics.cross_team_mutual_information = self._calculate_mutual_information()
        metrics.compositionality_score = self._calculate_compositionality()
        metrics.symbol_reuse_rate = self._calculate_symbol_reuse()
        
        shared, a_unique, b_unique = self._analyze_vocabulary_usage()
        metrics.shared_vocabulary_size = shared
        metrics.team_a_unique_symbols = a_unique
        metrics.team_b_unique_symbols = b_unique
        
        metrics.communication_efficiency = self._calculate_efficiency()
        
        return metrics
    
    def _calculate_mutual_information(self) -> float:
        """
        Calculate cross-team mutual information: I(Message_A ; Outcome_B)
        
        Measures how much Team A's messages predict Team B's success.
        High MI suggests shared understanding.
        """
        if len(self.message_outcomes) < 10:
            return 0.0
        
        # Simplification: Use message presence as signal
        # Group by message and calculate outcome distribution
        
        message_outcome_counts = defaultdict(lambda: {"success": 0, "fail": 0})
        
        for team, message, success in self.message_outcomes:
            if message:
                key = "success" if success else "fail"
                message_outcome_counts[message][key] += 1
        
        if not message_outcome_counts:
            return 0.0
        
        # Calculate MI (simplified)
        total = sum(
            counts["success"] + counts["fail"] 
            for counts in message_outcome_counts.values()
        )
        
        mi = 0.0
        for message, counts in message_outcome_counts.items():
            n_success = counts["success"]
            n_fail = counts["fail"]
            n_message = n_success + n_fail
            
            if n_message == 0:
                continue
            
            p_message = n_message / total
            p_success = sum(c["success"] for c in message_outcome_counts.values()) / total
            
            if n_success > 0:
                p_success_given_message = n_success / n_message
                if p_success > 0:
                    mi += (n_success / total) * math.log2(
                        p_success_given_message / p_success
                    )
            
            if n_fail > 0:
                p_fail = 1 - p_success
                p_fail_given_message = n_fail / n_message
                if p_fail > 0:
                    mi += (n_fail / total) * math.log2(
                        p_fail_given_message / p_fail
                    )
        
        return max(0.0, mi)
    
    def _calculate_compositionality(self) -> float:
        """
        Measure compositional structure emergence.
        
        Check if tokens combine systematically (e.g., tok3, tok8 = red circle).
        """
        if len(self.team_a_messages) < 20:
            return 0.0
        
        # Analyze multi-token messages
        multi_token_messages = []
        for msg in self.team_a_messages + self.team_b_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            if len(tokens) >= 2:
                multi_token_messages.append(tokens)
        
        if len(multi_token_messages) < 10:
            return 0.0
        
        # Simple heuristic: If same tokens appear in similar positions across messages,
        # suggests compositional structure
        
        position_token_counts = defaultdict(Counter)
        
        for tokens in multi_token_messages:
            for pos, token in enumerate(tokens[:3]):  # First 3 positions
                position_token_counts[pos][token] += 1
        
        # Calculate position consistency
        consistency_scores = []
        for pos, counter in position_token_counts.items():
            if sum(counter.values()) > 0:
                # Entropy: low entropy = consistent token usage at position
                total = sum(counter.values())
                entropy = -sum(
                    (count / total) * math.log2(count / total)
                    for count in counter.values() if count > 0
                )
                # Normalize by max entropy
                max_entropy = math.log2(len(self.vocabulary))
                consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def _calculate_symbol_reuse(self) -> float:
        """
        Calculate rate of symbol reuse across contexts.
        
        Higher reuse suggests efficient, abstract language.
        """
        all_messages = self.team_a_messages + self.team_b_messages
        
        if not all_messages:
            return 0.0
        
        # Count token occurrences
        token_counts = Counter()
        for msg in all_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            token_counts.update(tokens)
        
        # Calculate reuse: tokens used more than once
        reused_tokens = sum(1 for count in token_counts.values() if count > 1)
        total_unique = len(token_counts)
        
        return reused_tokens / total_unique if total_unique > 0 else 0.0
    
    def _analyze_vocabulary_usage(self) -> Tuple[int, int, int]:
        """
        Analyze vocabulary usage patterns.
        
        Returns:
            (shared_size, team_a_unique, team_b_unique)
        """
        # Extract all tokens used
        team_a_tokens = set()
        for msg in self.team_a_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            team_a_tokens.update(tokens)
        
        team_b_tokens = set()
        for msg in self.team_b_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            team_b_tokens.update(tokens)
        
        shared = len(team_a_tokens & team_b_tokens)
        a_unique = len(team_a_tokens - team_b_tokens)
        b_unique = len(team_b_tokens - team_a_tokens)
        
        return shared, a_unique, b_unique
    
    def _calculate_efficiency(self) -> float:
        """
        Calculate communication efficiency: success rate / avg message length.
        """
        if not self.message_outcomes:
            return 0.0
        
        successes = sum(1 for _, _, success in self.message_outcomes if success)
        success_rate = successes / len(self.message_outcomes)
        
        all_messages = self.team_a_messages + self.team_b_messages
        avg_length = np.mean([
            len([t for t in msg.split(',') if t.strip()])
            for msg in all_messages
        ]) if all_messages else 1.0
        
        return float(success_rate / avg_length) if avg_length > 0 else 0.0
    
    def detect_pidgin(self) -> Dict[str, Any]:
        """
        Detect emergence of pidgin language.
        
        Pidgin criteria:
        - Shared symbols used consistently by both teams
        - Different from initial vocabularies
        - Stable usage over time
        """
        if len(self.team_a_messages) < 30:
            return {"detected": False, "reason": "Insufficient data"}
        
        # Get shared vocabulary
        shared, a_unique, b_unique = self._analyze_vocabulary_usage()
        
        # Extract shared tokens
        team_a_tokens = set()
        for msg in self.team_a_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            team_a_tokens.update(tokens)
        
        team_b_tokens = set()
        for msg in self.team_b_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            team_b_tokens.update(tokens)
        
        shared_tokens = team_a_tokens & team_b_tokens
        
        # Check stability: tokens used consistently in recent messages
        recent_messages = (self.team_a_messages[-10:] + self.team_b_messages[-10:])
        recent_token_counts = Counter()
        for msg in recent_messages:
            tokens = [t.strip() for t in msg.split(',') if t.strip()]
            recent_token_counts.update(tokens)
        
        stable_shared = [
            token for token in shared_tokens
            if recent_token_counts[token] >= 5  # Used at least 5 times recently
        ]
        
        detected = len(stable_shared) >= 3 and shared >= 5
        
        return {
            "detected": detected,
            "shared_vocabulary_size": shared,
            "stable_pidgin_tokens": stable_shared,
            "pidgin_semantics": {
                token: self.semantics.get_token_semantics(token, top_k=3)
                for token in stable_shared
            }
        }
