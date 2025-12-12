"""
Resource Exchange Game Example

Demonstrates how to run the 4-player resource exchange game with different LLM backends.

Usage:
    # OpenAI backend
    python resource_exchange_game.py --backend openai --api-key sk-xxx --model gpt-4o

    # vLLM backend (local server)
    python resource_exchange_game.py --backend vllm --model meta-llama/Llama-3-8B

    # Dummy backend (for testing)
    python resource_exchange_game.py --backend dummy
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to path so that `import TinyVille...` works.
# __file__ = .../TinyVille/examples/resource_exchange_game.py
# pkg_root = .../TinyVille (outer, one level above inner package dir)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../TinyVille/TinyVille
PROJECT_ROOT = os.path.dirname(PKG_DIR)  # .../TinyVille
for p in (PROJECT_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from TinyVille.games.resource_exchange import ResourceExchangeConfig, ResourceExchangeGame
from TinyVille.core.llm_backends import create_llm_backend


def main():
    parser = argparse.ArgumentParser(description="Resource Exchange Game with LLM Agents")
    
    # LLM backend configuration
    parser.add_argument("--backend", type=str, default="dummy",
                       choices=["openai", "vllm", "dummy"],
                       help="LLM backend type")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model name (for OpenAI/vLLM)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                       help="vLLM server API base URL")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens in response")
    
    # Game configuration
    parser.add_argument("--rounds", type=int, default=14,
                       help="Total number of rounds")
    parser.add_argument("--chat-timesteps", type=int, default=3,
                       help="Number of timesteps in chat phase")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory to save game logs")
    
    # Cache configuration
    parser.add_argument("--cache-dir", type=str, default="./cache-new",
                       help="Directory for LLM response caching")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    
    args = parser.parse_args()
    
    # Prepare LLM backend config
    llm_config = {
        "type": args.backend,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    
    if args.backend == "openai":
        if not args.api_key:
            args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            raise ValueError("--api-key or OPENAI_API_KEY environment variable required for OpenAI backend")
        
        llm_config.update({
            "model": args.model,
            "api_key": args.api_key,
            "cache_dir": f"{args.cache_dir}/openai",
            "use_cache": not args.no_cache,
        })
    elif args.backend == "vllm":
        llm_config.update({
            "model": args.model,
            "api_base": args.api_base,
            "cache_dir": f"{args.cache_dir}/vllm",
            "use_cache": not args.no_cache,
        })
    elif args.backend == "dummy":
        # Dummy backend doesn't need extra config
        pass
    
    # Create game configuration
    config = ResourceExchangeConfig(
        total_rounds=args.rounds,
        chat_timesteps=args.chat_timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        llm_backend=llm_config,
    )
    
    print("=" * 70)
    print("Resource Exchange Game")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    if args.backend != "dummy":
        print(f"Model: {args.model}")
    print(f"Rounds: {args.rounds}")
    print(f"Chat timesteps per round: {args.chat_timesteps}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Create and run game
    game = ResourceExchangeGame(config)
    
    print("\n🎮 Starting game...\n")
    result = game.run()
    
    # Display results
    print("\n" + "=" * 70)
    print("GAME RESULTS")
    print("=" * 70)
    
    print("\n📊 Final Scores:")
    for team, score_info in result["final_scores"].items():
        print(f"  {team}:")
        print(f"    Provisional: {score_info['provisional']}")
        print(f"    Penalty: {score_info['penalty']}")
        print(f"    Final: {score_info['final']}")
    
    # Determine winner
    winner = max(result["final_scores"].items(), key=lambda x: x[1]["final"])
    print(f"\n🏆 Winner: {winner[0]} (Final Score: {winner[1]['final']})")
    
    # Save log
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"resource_exchange_{timestamp}.json")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 Game log saved to: {log_path}")
    
    # Display vocabulary
    print("\n📚 Alien Vocabulary:")
    vocab_items = list(result["vocabulary"].items())[:10]  # Show first 10
    for word, meaning in vocab_items:
        print(f"  {word} = {meaning}")
    if len(result["vocabulary"]) > 10:
        print(f"  ... and {len(result['vocabulary']) - 10} more")
    
    # Display pairing summary
    print("\n👥 Pairing Summary (first 3 rounds):")
    for i, pairing in enumerate(result["pairings"][:3], 1):
        print(f"  Round {i}: {pairing}")
    
    # Display chat statistics
    total_messages = sum(len(round_log.get("chat", [])) for round_log in result["rounds"])
    print(f"\n💬 Total messages exchanged: {total_messages}")
    
    # Display exchange statistics
    total_exchanges = sum(len(round_log.get("exchange", [])) for round_log in result["rounds"])
    print(f"🔄 Total exchanges: {total_exchanges}")
    
    print("\n" + "=" * 70)
    
    return result


if __name__ == "__main__":
    main()

