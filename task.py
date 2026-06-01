"""
Statistical-learning task generator.

Task: sample a word uniformly from a small vocabulary, emit its characters,
repeat. Writes the resulting character stream to `input.txt` so it can be
consumed by `min-char-rnn.py` unchanged.

From: https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning

Usage:
    python task.py                       # default: shared_letters, 50 chars
    python task.py shared_letters
    python task.py disjoint_letters --chars 200000
    python task.py one_word
"""

from __future__ import annotations

import argparse
import random

REGIMES: dict[str, list[str]] = {
    "one_word":         ["cat"],
    "disjoint_letters": ["cat", "mop", "red"],
    "shared_letters":   ["cat", "hat", "map"],
}


def generate_sequence(words: list[str], num_chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < num_chars:
        out.extend(rng.choice(words))
    return "".join(out[:num_chars])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regime", nargs="?", default="shared_letters",
                        choices=list(REGIMES.keys()))
    parser.add_argument("--chars", type=int, default=50,
                        help="total characters to emit (default: 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="input.txt")
    args = parser.parse_args()

    words = REGIMES[args.regime]
    text = generate_sequence(words, args.chars, seed=args.seed)
    with open(args.out, "w") as f:
        f.write(text)

    vocab = sorted(set(text))
    print(f"Regime:  {args.regime}")
    print(f"Words:   {words}")
    print(f"Vocab:   {''.join(vocab)} ({len(vocab)} symbols)")
    print(f"Wrote:   {args.out} ({len(text):,} characters)")
    print(f"Preview: {text[:80]}")


if __name__ == "__main__":
    main()
