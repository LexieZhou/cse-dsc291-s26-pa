# Part 3 Report: Speculative Decoding

## Setup and Implementation

I implemented single-sequence speculative decoding with the required default
model pair:

- Target model: `EleutherAI/pythia-1.4b-deduped`
- Draft model: `EleutherAI/pythia-160m-deduped`

Both models use the same tokenizer vocabulary. Generation is greedy so that the target-only baseline and speculative decoder are directly comparable. Each benchmark generates 100 new tokens from the prompt:

> The future of artificial intelligence is

In each iteration, the draft model autoregressively proposes `k` tokens. The
target model verifies those proposals in one vectorized forward pass. Draft
tokens are accepted consecutively until the first target-model mismatch. The
decoder then appends the target model's correction token and begins the next
iteration. If every proposal is accepted, the same target forward pass also
provides one bonus token. The implementation stops exactly at EOS or the
requested token limit.

## Optimizations

The implementation applies the following optimizations:

1. The draft model reuses its KV cache while generating the `k` tokens inside
   each proposal.
2. The target model verifies all draft tokens in a single batched forward pass
   and reuses its verified-prefix KV cache across iterations.
3. The target correction or bonus token is taken from the verification logits,
   avoiding an extra target-model forward pass.
4. Models run in `bfloat16` on supported CUDA devices, falling back to
   `float16` on other CUDA devices and `float32` on CPU.
5. CUDA synchronization is performed around wall-clock measurements. The
   notebook also runs one unmeasured warm-up iteration before collecting each
   benchmark result.

## Results

After one unmeasured warm-up iteration for each configuration, the sweep
produced the following results:

| `k` | Draft-token acceptance rate | Speculative tok/s | Baseline tok/s | Speedup |
|---:|---:|---:|---:|---:|
| 2 | 91.55% | 74.54 | 59.11 | 1.26x |
| 4 | 81.05% | 71.93 | 58.43 | 1.22x |
| 8 | 72.27% | 76.40 | 59.01 | 1.29x |
| 16 | 55.21% | 62.69 | 58.86 | 1.06x |

## Analysis

I select `k=2` as the final configuration. It satisfies both performance
targets: its draft-token acceptance rate is **91.55%**, above the required
75%, and its wall-clock speedup is **1.26x**, above the required 1.0x. Relative
to the target-only baseline, throughput rises from 59.11 to 74.54 tokens per
second and latency falls by 20.70%.

The sweep shows the expected trade-off. With `k=2`, the draft model is usually
correct, but the target model must run verification more often. With `k=8`,
raw throughput is highest, but the acceptance rate falls below the rubric
threshold. With `k=16`, long proposals diverge more frequently, so rejected
draft work offsets most of the benefit. For this model pair and prompt, `k=2`
is the fastest configuration that also satisfies the acceptance-rate
requirement.