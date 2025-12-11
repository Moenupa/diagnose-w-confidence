import os

import pytest
import torch
from vllm import LLM, SamplingParams

# vLLM V1 does not support logits processors.
os.environ["VLLM_USE_V1"] = "0"


__doc__ = """
A simplified test to verify that logits-spying works for the installed vLLM version.
This may go wrong if vLLM version is too high, e.g. 0.11.0+, hence this test.
"""


class LogitsSpy:
    def __init__(self):
        self.processed_logits: list[torch.Tensor] = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.processed_logits.append(logits.detach().cpu().float())
        return logits


def test_if_logits_spying_works():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping vLLM version tests.")

    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", enforce_eager=True)
    logits_spy = LogitsSpy()
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=10, logits_processors=[logits_spy]
    )
    outputs = llm.generate("What is the capital of France?", sampling_params)
    assert 0 < len(logits_spy.processed_logits) <= 10
    for logits in logits_spy.processed_logits:
        assert isinstance(logits, torch.Tensor)
        # vocab size of Qwen2.5
        assert logits.numel() == 151936

    from pprint import pp

    pp(
        {
            "text": outputs[0].outputs[0].text,
            "n_logits": len(logits_spy.processed_logits),
            "each_logits_shape": logits_spy.processed_logits[0].shape,
        }
    )


if __name__ == "__main__":
    test_if_logits_spying_works()
