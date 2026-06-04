"""Mixture-of-Experts: reference, tensor-parallel, and expert-parallel variants.

You will implement `ShardedLinear`, `MoE_TP`, and `MoE_EP` in this file. The
reference `SimpleMoE` and a pre-built `Router` are provided.
"""
import numpy as np

from mpi_wrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b."""

    def __init__(self, in_features, out_features):
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Two-layer MLP expert with ReLU."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Softmax-gated top-k router (replicated across ranks)."""

    def __init__(self, input_dim, num_experts):
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)
        gates = gates / np.sum(gates, axis=1, keepdims=True)
        return indices, gates


# ---------------------------------------------------------------------------
# Reference implementation: not parallel. Use this to verify correctness.
# ---------------------------------------------------------------------------
class SimpleMoE:
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
            ]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.1 — Tensor Parallel MoE.
# ---------------------------------------------------------------------------
class ShardedLinear:
    """Linear layer whose weight is column-sharded across MPI ranks.

    Each rank stores a `(in_features, out_features // world_size)` slice of the
    weight matrix. The forward pass produces the *full* output of shape
    `(batch, out_features)` on every rank, which means a collective is required
    to reassemble the columns each rank computed.

    Requires that `out_features` is evenly divisible by the world size.
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert out_features % self.world_size == 0, (
            f"Output features ({out_features}) must be evenly divisible by "
            f"world size ({self.world_size})"
        )

        self.in_features = in_features
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size
        self.output_offset = self.rank * self.local_out_features

        # Initialize local weights and bias
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global), dtype=np.float32)

        local_out = np.dot(x, self.weight) + self.bias

        padded = np.zeros((x.shape[0], self.out_features_global), dtype=local_out.dtype)
        padded[:, self.output_offset : self.output_offset + self.local_out_features] = local_out

        result = np.empty_like(padded)
        mpi.Allreduce(padded, result)
        return result


class ShardedExpert:
    """Expert whose weights are sharded along the hidden / output dim."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)
        return self.fc2(hidden)


class MoE_TP:
    """Mixture-of-Experts with tensor-parallel experts.

    Every rank holds a slice of every expert. Routing is replicated. After
    each expert's forward pass, ranks need a collective to reassemble the
    full output of that expert before applying the gate.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                ShardedExpert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

        if self.rank == 0:
            print(
                f"[MoE_TP] world_size={self.world_size}, num_experts={num_experts}, topk={self.topk}"
            )

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        indices, gates = self.router(x, self.topk)

        for e in range(self.num_experts):
            for k in range(self.topk):
                mask = indices[:, k] == e
                if not mask.any():
                    continue
                tokens = x[mask]
                expert_out = self.experts[e](tokens)
                outputs[mask] += gates[mask, k : k + 1] * expert_out

        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.2 — Expert Parallel MoE.
# ---------------------------------------------------------------------------
class MoE_EP:
    """Mixture-of-Experts with expert-parallel experts.

    Each rank owns *exactly one* expert. After routing, tokens that have been
    assigned to expert `e` must be sent to the rank that owns expert `e`. The
    expert computes its forward pass on the tokens it received and the results
    are sent back to the originating ranks.

    The natural collective for this pattern is **all-to-all**: each rank
    builds `world_size` buckets (one per destination rank) and exchanges them.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # == world size
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert num_experts == self.world_size, (
            "MoE_EP assumes one expert per rank; got "
            f"num_experts={num_experts}, world_size={self.world_size}"
        )

        with rng_context("router"):
            self.router = Router(input_dim, self.num_experts)

        # Each rank initializes its own expert independently — we want the
        # experts to be different, so this rng is rank-specific.
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        indices, gates = self.router(x, self.topk)

        for k in range(self.topk):
            send_tokens = [[] for _ in range(self.world_size)]
            send_positions = [[] for _ in range(self.world_size)]
            for i in range(batch_size):
                e = int(indices[i, k])
                send_tokens[e].append(x[i])
                send_positions[e].append(i)

            send_arrays = [
                np.stack(buf) if len(buf) > 0 else np.zeros((0, self.input_dim))
                for buf in send_tokens
            ]

            recv_arrays = mpi.alltoall(send_arrays)
            recv_counts = [r.shape[0] for r in recv_arrays]

            if sum(recv_counts) > 0:
                received = np.concatenate(recv_arrays, axis=0)
                expert_out = self.expert(received)
            else:
                expert_out = np.zeros((0, self.output_dim))

            send_back = []
            offset = 0
            for c in recv_counts:
                send_back.append(expert_out[offset : offset + c])
                offset += c

            recv_back = mpi.alltoall(send_back)

            for e in range(self.world_size):
                positions = send_positions[e]
                if not positions:
                    continue
                results = recv_back[e]
                for j, pos in enumerate(positions):
                    outputs[pos] += gates[pos, k] * results[j]

        return outputs

    def __call__(self, x):
        return self.forward(x)
