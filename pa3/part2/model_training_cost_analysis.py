"""Model training cost analysis for Part 2.

You will implement three functions:

  - `model_training_cost_analysis_llama(config_path)`
  - `model_training_cost_analysis_deepseek(config_path)`
  - `get_optimal_N_D_from_cost(cost_budget)`

Run from the command line:

  python model_training_cost_analysis.py --model_config llama3_8b_config.json
  python model_training_cost_analysis.py --model_config deepseek_v3_config.json
  python model_training_cost_analysis.py --training_budget 5000000
"""
import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Analyze training cost of a dense Llama-style model.

    Returns:
        total_params:   total trainable parameter count (int)
        flops_layer_TF: forward FLOPs of a single transformer layer (TFLOPs)
        peak_memory_GB: peak forward memory of a single transformer layer (GB)

    See the Part 2.1 writeup for the sequence-length / batch convention.
    """
    with open(model_config_path) as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    A = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    L = cfg["num_hidden_layers"]
    S = cfg["max_position_embeddings"]
    V = cfg["vocab_size"]
    tie = cfg.get("tie_word_embeddings", False)
    head_dim = H // A

    q_proj = H * (A * head_dim)
    k_proj = H * (KV * head_dim)
    v_proj = H * (KV * head_dim)
    o_proj = (A * head_dim) * H
    attn_params = q_proj + k_proj + v_proj + o_proj

    mlp_params = 3 * H * I
    norm_params = 2 * H
    layer_params = attn_params + mlp_params + norm_params

    embed_params = V * H
    lm_head_params = 0 if tie else V * H
    final_norm = H

    total_params = embed_params + lm_head_params + final_norm + L * layer_params

    flops_qkvo = 2 * S * (q_proj + k_proj + v_proj + o_proj)
    flops_qk = 2 * A * S * S * head_dim
    flops_av = 2 * A * S * S * head_dim
    flops_mlp = 2 * S * (3 * H * I)
    flops_layer = flops_qkvo + flops_qk + flops_av + flops_mlp
    flops_layer_TF = flops_layer / 1e12

    bytes_per = 2
    layer_param_bytes = layer_params * bytes_per
    input_act_bytes = S * H * bytes_per
    attn_score_bytes = A * S * S * bytes_per
    mlp_intermediate_bytes = 2 * S * I * bytes_per
    peak_intermediate = max(attn_score_bytes, mlp_intermediate_bytes)
    peak_memory_GB = (layer_param_bytes + input_act_bytes + peak_intermediate) / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Analyze training cost of a DeepSeek-V3-style MoE model.

    Same return signature as the Llama version. See the Part 2.3 writeup
    for the MLA attention and the dense-vs-MoE layer breakdown.
    """
    with open(model_config_path) as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    I_dense = cfg["intermediate_size"]
    I_moe = cfg["moe_intermediate_size"]
    A = cfg["num_attention_heads"]
    L = cfg["num_hidden_layers"]
    S = cfg["max_position_embeddings"]
    V = cfg["vocab_size"]
    tie = cfg.get("tie_word_embeddings", False)

    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    qk_head = qk_nope + qk_rope

    first_dense = cfg["first_k_dense_replace"]
    n_routed = cfg["n_routed_experts"]
    n_shared = cfg["n_shared_experts"]
    topk = cfg["num_experts_per_tok"]

    q_a = H * q_lora
    q_b = q_lora * (A * qk_head)
    kv_a = H * (kv_lora + qk_rope)
    kv_b = kv_lora * (A * (qk_nope + v_head))
    o_proj = (A * v_head) * H
    attn_norms = q_lora + kv_lora
    attn_params = q_a + q_b + kv_a + kv_b + o_proj + attn_norms

    dense_mlp = 3 * H * I_dense
    per_expert = 3 * H * I_moe
    routed_total = n_routed * per_expert
    shared_total = n_shared * per_expert
    router = H * n_routed
    norms = 2 * H

    dense_layer_params = attn_params + dense_mlp + norms
    moe_layer_params = attn_params + routed_total + shared_total + router + norms

    embed = V * H
    lm_head = 0 if tie else V * H
    final_norm = H

    n_dense = first_dense
    n_moe = L - first_dense
    total_params = (
        embed + lm_head + final_norm
        + n_dense * dense_layer_params
        + n_moe * moe_layer_params
    )

    active_routed = topk * per_expert
    active_moe_layer = attn_params + active_routed + shared_total + router + norms
    activated_params = (
        embed + lm_head + final_norm
        + n_dense * dense_layer_params
        + n_moe * active_moe_layer
    )

    flops_attn_proj = 2 * S * (q_a + q_b + kv_a + kv_b + o_proj)
    flops_qk = 2 * A * S * S * qk_head
    flops_av = 2 * A * S * S * v_head
    flops_attn = flops_attn_proj + flops_qk + flops_av

    flops_router = 2 * S * H * n_routed
    flops_moe_mlp = (topk + n_shared) * 2 * S * (3 * H * I_moe)
    flops_moe_layer = flops_attn + flops_router + flops_moe_mlp
    flops_layer_TF = flops_moe_layer / 1e12

    bytes_per = 2
    layer_param_bytes = moe_layer_params * bytes_per
    input_act_bytes = S * H * bytes_per
    attn_score_bytes = A * S * S * bytes_per
    mlp_intermediate_bytes = (topk + n_shared) * 2 * S * I_moe * bytes_per
    peak_intermediate = max(attn_score_bytes, mlp_intermediate_bytes)
    peak_memory_GB = (layer_param_bytes + input_act_bytes + peak_intermediate) / (1024 ** 3)

    print(f"  [deepseek] total_params      = {total_params:,} ({total_params/1e9:.2f} B)")
    print(f"  [deepseek] activated_per_tok = {activated_params:,} ({activated_params/1e9:.2f} B)")

    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """Pick the GPU and (N, D) that minimize loss under a $ training budget.

    cost_budget: a monetary training budget (in dollars)
    Returns:
        N: optimal model parameter count (absolute number)
        D: optimal training token count (absolute number)
        training_budget_flops: effective total training FLOPs
        best_gpu: name of the selected GPU, one of {'H100', 'H200', 'B200'}

    See the Part 2.2 writeup for the scaling law, the GPU price / TFLOPs
    table, and the MFU assumption.
    """
    mfu = 0.40
    gpus = {
        "H100": (3.0, 989e12),
        "H200": (4.0, 989e12),
        "B200": (6.0, 2250e12),
    }

    best_gpu = None
    best_flops = -1.0
    for name, (price_per_hour, peak_flops) in gpus.items():
        hours = cost_budget / price_per_hour
        effective_flops = hours * 3600 * peak_flops * mfu
        if effective_flops > best_flops:
            best_flops = effective_flops
            best_gpu = name

    training_budget_flops = best_flops

    alpha_n, alpha_d = 0.34, 0.29
    a_n = alpha_n * 406.4
    a_d = alpha_d * 410.7
    ratio = a_n / a_d
    exp_d = alpha_d / alpha_n
    k = ratio ** (1.0 / alpha_n)
    denom = 6.0 * k
    D = (training_budget_flops / denom) ** (1.0 / (1.0 + exp_d))
    N = k * D ** exp_d

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--training_budget", type=float, default=None,
                        help="Training budget in dollars")
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_deepseek(args.model_config)
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_llama(args.model_config)
            )
        else:
            print("Unknown model type — name your config llama*.json or deepseek*.json")
            raise SystemExit(1)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
