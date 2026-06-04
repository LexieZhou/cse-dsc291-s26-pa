# Part 2.3 — Why MoE?

Compare a dense Llama-3 8B against the activated-parameter footprint of
DeepSeek-V3 (~37B activated out of ~671B total). Discuss at least:

1. Total parameters vs. activated parameters per token, and how that shifts
   training FLOPs and memory.
2. Communication cost trade-offs (TP/EP collectives, all-to-all bandwidth) and
   how they scale with `num_experts` and `topk`.
3. Inference economics: why an MoE serves cheaper requests at low load but can
   become expensive at high load (token bucket imbalance, hot experts).
4. One concrete advantage and one concrete disadvantage relative to the dense
   baseline at the same training budget.

---

## 1. Total Parameters vs. Activated Parameters

Dense models use all parameters for every token. For example, Llama-3 8B activates essentially all of its 8B parameters during each forward pass. In contrast, MoE models such as DeepSeek-V3 have many more total parameters, but only a small subset of experts is activated per token.

This shifts the training cost: FLOPs scale mostly with activated parameters, not total parameters. Therefore, an MoE can have much larger total capacity while keeping per-token compute closer to that of a smaller dense model.

However, memory does not get the same discount. All experts must still be stored across the cluster because routing decisions are made dynamically for each token. As a result, MoE reduces compute per token, but still requires enough device memory and sharding to hold the full parameter set.

## 2. Communication Cost Trade-offs

TP and EP have different communication patterns.

In **TP**, each rank holds a shard of the model weights and computes part of the matrix multiplication. The partial outputs must then be combined using collectives such as `Allreduce` or `Allgather`. This communication is regular and mainly scales with activation size, not with `num_experts` or `topk`.

In **EP**, each rank owns one or more full experts. Tokens must be routed to the ranks that own their selected experts, and the outputs must be returned afterward. This requires `alltoall` communication. The communication volume scales roughly as `B × S × H × topk × sizeof(dtype)` for sending activations, plus a similar amount for returning outputs.

Therefore, EP communication grows approximately linearly with `topk`, because each token is sent to more experts. It is less directly dependent on `num_experts` in bandwidth terms, but more experts can increase routing overhead, smaller token buckets, and load imbalance. This matches our benchmark results that EP is efficient when `topk` is small, but becomes more communication-bound as `topk` or batch size increases.

## 3. Inference Economics

At low queries-per-second, MoE can serve requests more cheaply than a dense model with similar total capacity. Each token activates only a small number of experts, so the compute cost is much lower than using all parameters.

At high load, however, MoE can become harder and more expensive to serve efficiently. Tokens are not always evenly distributed across experts. Some experts may become hot and receive many more tokens than others, causing those ranks to become bottlenecks while other ranks are underutilized.

MoE also reduces GEMM efficiency because each expert only processes the tokens routed to it. Smaller per-expert batches make weight reads harder to amortize, so serving can become memory-bandwidth bounded. This is why production MoE systems need large batching, expert-aware scheduling, load balancing, and strong interconnect bandwidth.

## 4. One concrete advantage and disadvantage

A concrete advantage of MoE is **higher capacity per training dollar**. For the same training FLOP budget, an MoE model can store many more total parameters than a dense model while keeping the activated compute per token relatively low. This allows the model to increase knowledge capacity without increasing training FLOPs proportionally.

A concrete disadvantage is **systems complexity and communication overhead**. Unlike a dense baseline, an MoE model must shard many experts across devices and route tokens dynamically. Each MoE layer introduces extra `alltoall` communication, load-balancing challenges, and sensitivity to hot experts. Therefore, MoE can reduce FLOPs per token, but it requires a more complex and communication-heavy training and serving system.