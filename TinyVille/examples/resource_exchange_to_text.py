"""
Convert resource_exchange JSON log to a human-readable text transcript.

Features:
- Translates chat tokens to meanings using the per-game vocabulary.
- Lists per-round chat, resource exchanges, and feedback.
- Shows final team scores.

Usage:
    python resource_exchange_to_text.py --input TinyVille/logs/resource_exchange_YYYYMMDD_HHMMSS.json \
                                        --output transcript.txt

If --input is omitted, the latest resource_exchange_*.json under TinyVille/logs is used.
If --output is omitted, prints to stdout.
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../TinyVille
LOG_DIR = ROOT / "logs"


def find_latest():
    files = sorted(LOG_DIR.glob("resource_exchange_*.json"))
    return files[-1] if files else None


def format_round(r, translate):
    lines = []
    lines.append(f"\n--- Round {r['round']} ---")
    pairing = r.get("pairing", {})
    if pairing:
        lines.append(f"Pairing: {pairing}")

    chat = r.get("chat", [])
    if chat:
        lines.append("Chat:")
        for m in chat:
            content = m.get("content")
            if content is None:
                tokens = []
            elif isinstance(content, str):
                tokens = content.split()
            else:
                tokens = list(content)
            translated = translate(tokens)
            lines.append(f"  {m['sender']} -> {m['receiver']}: {content}")
            lines.append(f"    meaning: {translated}")
    else:
        lines.append("Chat: (none)")

    exch = r.get("exchange", [])
    if exch:
        lines.append("Exchange:")
        for e in exch:
            lines.append(
                f"  {e['giver']} -> {e['receiver']}: {e['amount']} {e['resource']} "
                f"(giverΔ={e['deltas']['giver_delta']}, recvΔ={e['deltas']['receiver_delta']})"
            )
    else:
        lines.append("Exchange: (none)")

    fb = r.get("feedback", [])
    if fb:
        lines.append("Feedback:")
        for f in fb:
            lines.append(
                f"  {f['player']}: partner_team={'teammate' if f['is_teammate'] else 'opponent'}, "
                f"rating={f.get('rating')}, given={f['feedback_view'].get('given')}, "
                f"received={f['feedback_view'].get('received')}"
            )
    else:
        lines.append("Feedback: (none)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Readable transcript for resource exchange logs.")
    parser.add_argument("--input", type=str, help="Path to resource_exchange_*.json (default: latest in TinyVille/logs)")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    src = pathlib.Path(args.input) if args.input else find_latest()
    if not src or not src.exists():
        print("No log file found.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(src.read_text())
    rounds = data.get("rounds", [])
    final_scores = data.get("final_scores", {})
    vocab = data.get("vocabulary", {})

    # translator
    def translate(tokens):
        return " ".join([vocab.get(t, f"[{t}]") for t in tokens])

    out_lines = []
    out_lines.append("=" * 70)
    out_lines.append(f"Resource Exchange Transcript - {src.name}")
    out_lines.append("=" * 70)
    out_lines.append(f"Total rounds: {len(rounds)}")
    out_lines.append(f"Vocabulary size: {len(vocab)}")
    out_lines.append("\nFinal scores:")
    for team, sc in final_scores.items():
        out_lines.append(
            f"  {team}: provisional={sc.get('provisional')}, penalty={sc.get('penalty')}, final={sc.get('final')}"
        )

    for r in rounds:
        out_lines.append(format_round(r, translate))

    output_text = "\n".join(out_lines)

    if args.output:
        pathlib.Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Transcript written to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

