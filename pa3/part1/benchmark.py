"""Benchmark harness for Part 1.3.

Use this as a starting point. To get full credit on the benchmark you should
sweep at least one of {batch_size, hidden_dim, num_experts, topk} and produce a
table or plot in `analysis.md`.

Run with (from the `pa3/` directory, same convention as `test_moe.py`):
    mpirun --oversubscribe -n 4 python part1/benchmark.py
"""
import time

import numpy as np

from mpi_wrapper import mpi
from rng import get_rng
from moe import SimpleMoE, MoE_EP, MoE_TP


def run_moe(moe_type, batch_size=8, feature_dim=32, hidden_dim=128,
            output_dim=64, num_experts=None, topk=2, n_iters=10):
    if num_experts is None:
        num_experts = mpi.Get_size()

    if mpi.Get_rank() == 0:
        X = get_rng().randn(batch_size, feature_dim)
    else:
        X = None
    X = mpi.bcast(X, root=0)

    model_cls = {"simple": SimpleMoE, "ep": MoE_EP, "tp": MoE_TP}[moe_type]
    moe = model_cls(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk,
    )

    _ = moe(X)  # warm up
    mpi.Barrier()

    t0 = time.time()
    for _ in range(n_iters):
        outputs = moe(X)
    mpi.Barrier()
    avg_ms = 1000 * (time.time() - t0) / n_iters

    return outputs, avg_ms


def _print_header(title):
    if mpi.Get_rank() == 0:
        print(f"\n=== {title} (world_size={mpi.Get_size()}) ===")
        print(f"{'config':44}  {'simple':>10}  {'tp':>10}  {'ep':>10}")


def _run_row(tag, **kwargs):
    _, t_simple = run_moe("simple", **kwargs)
    _, t_tp = run_moe("tp", **kwargs)
    _, t_ep = run_moe("ep", **kwargs)
    if mpi.Get_rank() == 0:
        print(f"{tag:44}  {t_simple:10.2f}  {t_tp:10.2f}  {t_ep:10.2f}")


def sweep_batch_size():
    _print_header("Sweep: batch_size  (feat=64, hidden=256, out=64, topk=2)")
    for batch in (8, 32, 128, 512, 2048):
        _run_row(
            f"batch={batch:<5}",
            batch_size=batch, feature_dim=64, hidden_dim=256,
            output_dim=64, topk=2,
        )


def sweep_hidden_dim():
    _print_header("Sweep: hidden_dim  (batch=64, feat=64, out=64, topk=2)")
    for hidden in (64, 256, 1024, 4096):
        _run_row(
            f"hidden={hidden:<5}",
            batch_size=64, feature_dim=64, hidden_dim=hidden,
            output_dim=64, topk=2,
        )


def sweep_topk():
    _print_header("Sweep: topk  (batch=64, feat=64, hidden=256, out=64)")
    ws = mpi.Get_size()
    for topk in range(1, ws + 1):
        _run_row(
            f"topk={topk:<5}",
            batch_size=64, feature_dim=64, hidden_dim=256,
            output_dim=64, topk=topk,
        )


def benchmark_moe():
    sweep_batch_size()
    sweep_hidden_dim()
    sweep_topk()


if __name__ == "__main__":
    benchmark_moe()
