"""
Run Resource Scramble Experiment

Usage:
    python run_experiment.py
    
Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key (required)
"""

import os
import sys
from datetime import datetime
from simulation import ResourceScrambleSimulation


def main():
    """Run the experiment."""
    
    # Check for API key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set!")
        print("\nPlease set your DeepSeek API key:")
        print("  Windows PowerShell: $env:DEEPSEEK_API_KEY='your-key-here'")
        print("  Windows CMD: set DEEPSEEK_API_KEY=your-key-here")
        print("  Linux/Mac: export DEEPSEEK_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Configuration
    config = {
        'api_key': api_key,
        'model': 'deepseek-chat',
        'vocabulary_size': 20,          # K = 20 abstract tokens
        'max_message_length': 5,        # L = 5 tokens per message
        'max_rounds': 10,                # Max rounds per episode
        'num_episodes': 50,              # Total episodes to run (reduced for testing)
        'output_dir': f'./results/exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("\n" + "=" * 70)
    print("SMALLVILLE: Resource Scramble Experiment")
    print("Studying Pidgin Language Emergence in Zero-Sum Games")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {config['model']}")
    print(f"  Vocabulary: {config['vocabulary_size']} abstract tokens")
    print(f"  Max message length: {config['max_message_length']} tokens")
    print(f"  Episodes: {config['num_episodes']}")
    print(f"  Output: {config['output_dir']}")
    print()
    
    try:
        # Create and run simulation
        sim = ResourceScrambleSimulation(config)
        sim.run()
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
