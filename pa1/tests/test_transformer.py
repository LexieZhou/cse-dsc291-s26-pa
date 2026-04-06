from typing import List

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import transformer


def get_weight_path() -> Path:
    return Path(__file__).parent.parent / "weights.pt"


def load_submitted_weights() -> List[torch.Tensor]:
    weights = transformer.load_weights(str(get_weight_path()))
    assert isinstance(weights, list)
    assert len(weights) == 9
    return weights


def count_training_set_matches(weights: List[torch.Tensor]) -> int:
    num_correct = 0
    for sentence in transformer.SENTENCES:
        prompt = " ".join(sentence.split()[:2])
        generated = transformer.generate(
            prompt, weights, max_new_tokens=transformer.NUM_WORDS
        )

        generated_words = generated.split()[: transformer.NUM_WORDS]
        expected_words = sentence.split()
        if generated_words == expected_words:
            num_correct += 1
    return num_correct


def test_weights_file_exists():
    assert get_weight_path().exists(), "weights.pt not found"


def test_weights_can_be_loaded():
    weights = load_submitted_weights()
    assert isinstance(weights, list)
    assert len(weights) == 9


def test_weights_shapes():
    weights = load_submitted_weights()

    assert weights[0].shape == (transformer.VOCAB_SIZE, transformer.MODEL_DIM)
    assert weights[1].shape == (transformer.SEQ_LEN, transformer.MODEL_DIM)
    assert weights[2].shape == (transformer.MODEL_DIM, transformer.MODEL_DIM)
    assert weights[3].shape == (transformer.MODEL_DIM, transformer.MODEL_DIM)
    assert weights[4].shape == (transformer.MODEL_DIM, transformer.MODEL_DIM)
    assert weights[5].shape == (transformer.MODEL_DIM, transformer.MODEL_DIM)
    assert weights[6].shape == (transformer.MODEL_DIM, transformer.FF_DIM)
    assert weights[7].shape == (transformer.FF_DIM, transformer.MODEL_DIM)
    assert weights[8].shape == (transformer.MODEL_DIM, transformer.VOCAB_SIZE)


def test_generate_returns_string():
    weights = load_submitted_weights()
    generated = transformer.generate("attention is", weights, max_new_tokens=2)
    assert isinstance(generated, str)
    assert len(generated.strip()) > 0


def test_generate_training_set_accuracy():
    weights = load_submitted_weights()
    num_correct = count_training_set_matches(weights)
    assert num_correct >= 8


if __name__ == "__main__":
    test_weights_file_exists()
    test_weights_can_be_loaded()
    test_weights_shapes()
    test_generate_returns_string()
    test_generate_training_set_accuracy()
