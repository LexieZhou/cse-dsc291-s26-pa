# Part 1.3 — Benchmark Analysis

Suggested structure:

1. **Setup.** Hardware (CPU model, core count), MPI library + version, world
   size, dtype.
   
2. **Sweeps.** What you varied (batch size, hidden dim, num experts, topk)
   and the configurations you measured. Include a table of timings (ms / fwd
   pass) for `SimpleMoE`, `MoE_TP`, and `MoE_EP`.

3. **Discussion.** Which variant is faster, and why? Identify whether the
   bottleneck for each variant is computation (matmul) or communication
   (which collective and how many bytes per rank). Tie the explanation back
   to the role of `Allreduce` / `Allgather` in TP and `alltoall` in EP.


Plots are optional but encouraged.

---

## 1. Setup

- **CPU:** Apple M4, 10 physical / 10 logical cores
- **MPI:** Open MPI 5.0.9
- **Python / libs:** Python 3.10 (conda env `cse291pa3`), NumPy 2.2.6, mpi4py 4.1.2
- **dtype:** `float64` throughout
- **World size:** 4 ranks, `num_experts = world_size = 4`

## 2. Sweeps

All numbers are **milliseconds per forward pass**, lower is better. `simple` is the per-token Python-loop reference; `tp` is `MoE_TP`; `ep` is `MoE_EP`.

### 2.1 Batch size — feat=64, hidden=256, out=64, topk=2

| batch | simple |   tp  |   ep  |
|------:|-------:|------:|------:|
|     8 |   0.12 |  0.26 |  0.16 |
|    32 |   0.42 |  0.26 |  0.32 |
|   128 |   1.61 |  0.51 |  0.76 |
|   512 |   6.39 |  1.14 |  2.87 |
|  2048 |  25.44 |  7.05 | 14.99 |

### 2.2 Hidden dim — batch=64, feat=64, out=64, topk=2

| hidden | simple |   tp  |   ep  |
|-------:|-------:|------:|------:|
|     64 |   0.60 |  0.23 |  0.50 |
|    256 |   0.80 |  0.52 |  0.63 |
|   1024 |   2.43 |  1.40 |  1.18 |
|   4096 |  27.61 |  3.08 |  4.73 |

### 2.3 topk — batch=64, feat=64, hidden=256, out=64

| topk | simple |   tp  |   ep  |
|-----:|-------:|------:|------:|
|    1 |   0.42 |  0.43 |  0.30 |
|    2 |   0.81 |  0.56 |  0.64 |
|    3 |   1.17 |  0.80 |  1.01 |
|    4 |   1.62 |  0.62 |  0.95 |

## 3. Discussion

Overall, `MoE_TP` is the fastest variant in most medium and large settings, while `MoE_EP` is competitive mainly for small workloads or `topk = 1`. `SimpleMoE` is consistently slower as the workload grows because it uses a per-token Python-loop reference implementation and does not parallelize expert computation.

In the **batch-size sweep**, `SimpleMoE` increases almost linearly with batch size, from `0.12 ms` at batch 8 to `25.44 ms` at batch 2048. `MoE_TP` scales much better, reaching only `7.05 ms` at batch 2048. `MoE_EP` is faster than `SimpleMoE`, but slower than TP for large batches, reaching `14.99 ms` at batch 2048. This suggests that EP becomes increasingly communication-bound as more tokens must be redistributed across ranks through `alltoall`.

In the **hidden-dimension sweep**, increasing hidden size makes expert computation more expensive. `MoE_TP` benefits most from splitting matrix multiplications across ranks: at hidden size 4096, it takes `3.08 ms`, compared with `27.61 ms` for `SimpleMoE`. `MoE_EP` also improves over the baseline, but takes `4.73 ms` at hidden size 4096, likely because its `alltoall` routing overhead is still significant.

In the **topk sweep**, runtime generally increases as each token is routed to more experts. `SimpleMoE` shows the clearest trend, rising from `0.42 ms` at `topk = 1` to `1.62 ms` at `topk = 4`. `MoE_EP` is fastest at `topk = 1` because each token is sent to only one expert rank. However, as `topk` grows, EP must send more token-expert pairs through `alltoall`, increasing communication cost. `MoE_TP` is more stable because its communication pattern is more regular.

The key difference is the communication pattern. In **TP**, each rank stores a shard of the expert weights and computes part of the matrix multiplication. The partial results then need to be combined using collectives such as `Allreduce` or `Allgather`. This communication is relatively regular and is often outweighed by the compute savings for larger hidden dimensions.

In **EP**, each rank owns full experts, so tokens must be routed to the correct expert ranks using `alltoall`, then the results must be sent back. The communication volume scales roughly with `B * topk * d_feat * sizeof(dtype)`

for sending inputs, plus a similar amount for returning outputs. With `float64`, `feat = 64`, `batch = 2048`, and `topk = 2`, EP moves on the order of several MB of activation data per forward pass, which explains why it slows down for large batches.

In summary, `SimpleMoE` is mostly limited by Python overhead and un-parallelized computation. `MoE_TP` is usually fastest because it parallelizes the expensive matrix multiplications and uses efficient structured collectives. `MoE_EP` can be best for very sparse routing, especially `topk = 1`, but becomes communication-bound as batch size or `topk` increases due to `alltoall`.

