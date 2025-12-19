"""
Test script to verify the implementation without running full experiment.
"""

import os
from resource_scramble import (
    ResourceScrambleEnvironment,
    AbstractVocabulary,
    CurriculumPhase,
    CurriculumManager
)
from language_analysis import LanguageAnalyzer, SymbolSemantics


def test_environment():
    """Test environment setup."""
    print("\n=== Testing Environment ===")
    
    phase = CurriculumPhase(
        phase_id=1,
        num_resources=4,
        feature_dimensions=2
    )
    
    env = ResourceScrambleEnvironment(phase, seed=42)
    state = env.reset()
    
    print(f"✓ Environment created with {state['num_resources']} resources")
    
    # Test views
    team_a_view = env.get_team_a_view()
    team_b_view = env.get_team_b_view()
    
    print(f"✓ Team A view (locations): {len(team_a_view)} items")
    print(f"  Example: Resource {team_a_view[0]['id']} at ({team_a_view[0]['x']:.1f}, {team_a_view[0]['y']:.1f})")
    
    print(f"✓ Team B view (attributes): {len(team_b_view)} items")
    print(f"  Example: Resource {team_b_view[0]['id']}: {team_b_view[0]['color']} {team_b_view[0]['shape']}, value={team_b_view[0]['value']}")
    
    # Test evaluation
    optimal = env.get_optimal_choice()
    reward, correct = env.evaluate_choice(optimal)
    print(f"✓ Optimal choice: Resource {optimal}, reward={reward}, correct={correct}")


def test_vocabulary():
    """Test abstract vocabulary system."""
    print("\n=== Testing Vocabulary ===")
    
    vocab = AbstractVocabulary(size=20, max_length=5)
    
    print(f"✓ Vocabulary created: {', '.join(vocab.tokens[:5])}... ({vocab.size} tokens)")
    
    # Test valid messages
    valid_msg = "tok1, tok5, tok12"
    assert vocab.is_valid_message(valid_msg), "Valid message rejected"
    print(f"✓ Valid message accepted: '{valid_msg}'")
    
    # Test invalid messages
    invalid_msg = "hello, world"
    assert not vocab.is_valid_message(invalid_msg), "Invalid message accepted"
    print(f"✓ Invalid message rejected: '{invalid_msg}'")
    
    # Test parsing
    parsed = vocab.parse_message("tok3, tok7, tok11")
    print(f"✓ Message parsed: {parsed}")


def test_curriculum():
    """Test curriculum manager."""
    print("\n=== Testing Curriculum ===")
    
    manager = CurriculumManager()
    
    phase1 = manager.get_current_phase()
    print(f"✓ Current phase: {phase1.phase_id}")
    print(f"  Resources: {phase1.num_resources}, Threshold: {phase1.success_threshold}")
    
    # Simulate advancement
    print(f"✓ Can advance: {manager.should_advance()}")
    
    print(f"✓ Curriculum has {len(manager.phases)} phases")


def test_language_analyzer():
    """Test language analysis."""
    print("\n=== Testing Language Analyzer ===")
    
    vocab = AbstractVocabulary(size=20)
    analyzer = LanguageAnalyzer(vocab.tokens)
    
    # Simulate communication
    context = {
        "resource_id": 0,
        "resource_attributes": {
            "color": "red",
            "shape": "circle",
            "value": 8,
            "is_trap": False
        }
    }
    
    for i in range(10):
        analyzer.record_communication(
            team_a_msg="tok3, tok7",
            team_b_msg="tok11, tok15",
            team_a_success=True,
            team_b_success=False,
            context=context
        )
    
    print(f"✓ Recorded 10 communication exchanges")
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    print(f"✓ Metrics calculated:")
    print(f"  - Cross-team MI: {metrics.cross_team_mutual_information:.4f}")
    print(f"  - Compositionality: {metrics.compositionality_score:.4f}")
    print(f"  - Symbol reuse: {metrics.symbol_reuse_rate:.4f}")
    
    # Test pidgin detection
    pidgin = analyzer.detect_pidgin()
    print(f"✓ Pidgin detection: {pidgin['detected']}")


def test_api_key():
    """Check if API key is set."""
    print("\n=== Checking API Configuration ===")
    
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ API key is set: {masked_key}")
    else:
        print("⚠ API key not set (required for full experiment)")
        print("  Set with: $env:DEEPSEEK_API_KEY = 'your-key'")


def main():
    """Run all tests."""
    print("=" * 70)
    print("SmallVille Implementation Tests")
    print("=" * 70)
    
    try:
        test_environment()
        test_vocabulary()
        test_curriculum()
        test_language_analyzer()
        test_api_key()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nReady to run experiment with: python run_experiment.py")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
