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
    parser.add_argument("--backend", type=str, default="poe",
                       choices=["openai", "vllm", "poe", "dummy"],
                       help="LLM backend type")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                       help="Model name (for OpenAI/vLLM)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--api-base", type=str, default="https://api.poe.com/v1",
                       help="vLLM server API base URL")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens in response")
    parser.add_argument("--invention-hint", type=str, default=None,
                       help="Custom phrasing to soften encouragement for inventing tokens (overrides default)")
    parser.add_argument("--soft-invention", action="store_true",
                       help="Use a soft default invention hint (short phrasing)")
    
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
    elif args.backend == "poe":
        # Poe backend expects POE_API_KEY env var or --api-key
        if not args.api_key:
            args.api_key = os.environ.get("POE_API_KEY")
        if not args.api_key:
            raise ValueError("--api-key or POE_API_KEY environment variable required for Poe backend")
        llm_config.update({
            "model": args.model,
            "api_key": args.api_key,
            "api_base": args.api_base,
            "cache_dir": f"{args.cache_dir}/poe",
            "use_cache": not args.no_cache,
        })
    elif args.backend == "dummy":
        # Dummy backend doesn't need extra config
        pass
    
    # Create base game configuration
    invention_hint = args.invention_hint
    if args.soft_invention and not invention_hint:
        invention_hint = "Don't hesitate to make mistakes as long as it helps you win. Dialects in different groups are allowed."

    base_config = ResourceExchangeConfig(
        total_rounds=args.rounds,
        chat_timesteps=args.chat_timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        llm_backend=llm_config,
        invention_hint=invention_hint,
    )

    print("=" * 70)
    print("Resource Exchange Game — Ablation variants")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    if args.backend != "dummy":
        print(f"Model: {args.model}")
    print(f"Rounds: {args.rounds}")
    print(f"Chat timesteps per round: {args.chat_timesteps}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Run each ablation preset once and save separate logs
    summaries = {}
    for preset_name, preset_vals in base_config.ablation_presets.items():
        print(f"\n--- Running preset: {preset_name} (reward_scale={preset_vals['reward_scale']}, penalty_scale={preset_vals['penalty_scale']}) ---")
        cfg = ResourceExchangeConfig(
            total_rounds=base_config.total_rounds,
            chat_timesteps=base_config.chat_timesteps,
            seed=base_config.seed,
            log_dir=os.path.join(base_config.log_dir, preset_name) if base_config.log_dir else None,
            llm_backend=base_config.llm_backend,
            invention_hint=base_config.invention_hint,
        )
        # Apply preset
        cfg.reward_penalty = preset_vals

        # Ensure per-preset log dir exists
        if cfg.log_dir:
            os.makedirs(cfg.log_dir, exist_ok=True)

        game = ResourceExchangeGame(cfg)
        print("\n🎮 Starting game...\n")
        result = game.run()

        # Save log per preset
        if cfg.log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "-")
            prompt_name = args.soft_invention
            log_path = os.path.join(cfg.log_dir, f"resource_exchange_{model_name}_{prompt_name}_{preset_name}_{timestamp}.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n📝 Game log saved to: {log_path}")

        # Collect summary
        final_scores = result['final_scores']
        winner = max(final_scores.items(), key=lambda x: x[1]['final'])
        summaries[preset_name] = {
            'winner': winner[0],
            'final_score': winner[1]['final'],
            'reward_config': result.get('reward_config', {})
        }

    # Display summary of all presets
    print("\n" + "=" * 70)
    print("SUMMARY — Ablation presets")
    print("=" * 70)
    for name, s in summaries.items():
        print(f"  {name}: winner={s['winner']}, final_score={s['final_score']}, reward_cfg={s['reward_config']}")

    return summaries
    
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

