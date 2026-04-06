"""Interactive playground for the trained transformer LM."""

from typing import List
import torch


def playground(
    generate_fn,
    model_weights: List[torch.Tensor],
    vocab: dict,
) -> None:
    """Launch an interactive playground after training.

    Asks the user whether to enter the playground. If yes, repeatedly
    prompts for a prefix and generates a continuation.

    Parameters
    ----------
    generate_fn : callable
        The generate(prompt, model_weights, max_new_tokens) function.
    model_weights : List[torch.Tensor]
        Trained model weights.
    vocab : dict
        WORD_TO_IDX mapping (used to validate input words).
    """
    print("\n" + "=" * 50)
    print("  LM Playground")
    print("=" * 50)
    ans = input("Enter playground? (y/n): ").strip().lower()
    if ans not in ("y", "yes"):
        return

    # Show the vocabulary
    known_words = sorted(w for w in vocab if w != "<pad>")
    print(f"\nVocabulary ({len(known_words)} words):")
    cols = 6
    for i in range(0, len(known_words), cols):
        row = known_words[i:i + cols]
        print("  " + "  ".join(f"{w:<12s}" for w in row))

    print("\nOnly these words are recognized by our model.")
    print("Type a prompt using words above, and the model will continue.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            prefix = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prefix or prefix.lower() in ("quit", "exit"):
            break

        unknown = [w for w in prefix.split() if w not in vocab]
        if unknown:
            print(f"  Unknown words: {unknown}")
            print(f"  Please use only words from the vocabulary above.\n")
            continue

        try:
            result = generate_fn(prefix, model_weights, max_new_tokens=10)
            print(f"  {result}\n")
        except Exception as e:
            print(f"  Error: {e}\n")

    print("Goodbye!")
