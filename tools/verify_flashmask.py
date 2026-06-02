#!/usr/bin/env python3
"""
最小可验证脚本：测试 flashmask_pybind 的 forward 和 backward。

FlashMask V2 tensor 格式:
  - Q, K, V:  [B, L, H, D]  (batch, seqlen, heads, dim)
  - startend: [B, H, L, 4]  (batch, heads, seqlen, 4)
  - block_mask: [B, H, m_blocks, n_blocks]
  - out:       [B, L, H, D]
  - softmax_lse: [B, L, H]

用法:
    python tools/verify_flashmask.py [--L 512] [--dtype bf16]
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

import flashmask_pybind


def reference_attention(q, k, v, is_causal, sm_scale):
    """
    PyTorch 原生 attention（与 FlashMask 结果对比）。

    q, k, v: [B, L, H, D]
    返回: out [B, L, H, D]
    """
    # 转为 [B, H, L, D] 方便用标准 matmul
    q_t = q.transpose(1, 2)  # [B, H, L, D]
    k_t = k.transpose(1, 2)  # [B, H, L, D]
    v_t = v.transpose(1, 2)  # [B, H, L, D]

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale  # [B, H, L, L]

    if is_causal:
        L = q.shape[1]
        causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    out_t = torch.matmul(attn_weights, v_t)  # [B, H, L, D]
    out = out_t.transpose(1, 2)  # [B, L, H, D]
    return out


def build_masks(B, H, L, device, mode, block_size=128):
    """
    构造 FlashMask V2 的 startend 和 block_mask。

    mode == "causal_via_flag":
        startend 全哨兵，is_causal=True 让 kernel 内部处理因果 mask。
    mode == "causal_via_mask":
        startend 编码因果 mask 区间，is_causal=False。
        注意: startend 定义的是被 MASK 的行区间，不是可见区间。
    """
    m_blocks = (L + block_size - 1) // block_size
    n_blocks = (L + block_size - 1) // block_size

    if mode == "causal_via_flag":
        startend = torch.full((B, H, L, 4), L, dtype=torch.int32, device=device)
    else:
        # causal: Key j 对 Query 行 [0, j) 不可见 → mask 区间 [0, j)
        # 可见行是 [j, L)
        startend = torch.full((B, H, L, 4), L, dtype=torch.int32, device=device)
        for col in range(L):
            startend[:, :, col, 0] = 0        # lt_start
            startend[:, :, col, 1] = col      # lt_end: mask [0, col)
        # col=0 → mask [0,0)=empty → 所有行可见（正确：所有 token 都能看到第一个 token）

    block_mask = torch.zeros(B, H, m_blocks, n_blocks, dtype=torch.int32, device=device)
    for i in range(m_blocks):
        for j in range(min(i + 1, n_blocks)):
            block_mask[:, :, i, j] = 1

    return startend, block_mask


def check_close(name, a, b):
    """打印并判断两个 tensor 是否 close"""
    abs_diff = (a - b).abs()
    rel_diff = abs_diff / (b.abs().clamp(min=1e-8))
    print(f"\n  {name}:")
    print(f"    Max  abs diff:  {abs_diff.max().item():.6e}")
    print(f"    Mean abs diff:  {abs_diff.mean().item():.6e}")
    print(f"    Max  rel diff:  {rel_diff.max().item():.4f} ({rel_diff.max().item()*100:.2f}%)")
    print(f"    Mean rel diff:  {rel_diff.mean().item():.4f} ({rel_diff.mean().item()*100:.2f}%)")

    ok = torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    status = f"✅ {name} PASS" if ok else f"❌ {name} FAIL"
    print(f"    {status}")

    if not ok:
        flat_idx = abs_diff.view(-1).argmax().item()
        idx = torch.unravel_index(torch.tensor(flat_idx), a.shape)
        idx_tuple = tuple(i.item() for i in idx)
        print(f"    最差异位置 {idx_tuple}: a={a[idx_tuple].item():.6f}, b={b[idx_tuple].item():.6f}")

    return ok


def run_test(B, H, L, D, dtype, mode):
    """运行单次 fwd+bwd 测试"""

    sm_scale = 1.0 / math.sqrt(D)
    device = torch.device("cuda")
    is_causal = (mode == "causal_via_flag")

    # ---- FlashMask 格式: [B, L, H, D] ----
    q = torch.randn(B, L, H, D, dtype=dtype, device=device)
    k = torch.randn(B, L, H, D, dtype=dtype, device=device)
    v = torch.randn(B, L, H, D, dtype=dtype, device=device)

    q_fm = q.detach().clone().requires_grad_(True)
    k_fm = k.detach().clone().requires_grad_(True)
    v_fm = v.detach().clone().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    startend, block_mask = build_masks(B, H, L, device, mode)

    print(f"  mode:              {mode} (is_causal={is_causal})")
    print(f"  q.shape:           {q_fm.shape} (dtype={q_fm.dtype})")
    print(f"  startend.shape:    {startend.shape}")
    print(f"  block_mask.shape:  {block_mask.shape}")
    print(f"  sm_scale:          {sm_scale}")

    # ---- FlashMask Forward ----
    print("  Running FlashMask V2 forward...")
    out, softmax_lse = flashmask_pybind.flashmask_v2_fwd(
        q_fm, k_fm, v_fm, startend, block_mask, sm_scale, is_causal,
    )
    print(f"  out.shape:         {out.shape}")
    print(f"  softmax_lse.shape: {softmax_lse.shape}")

    # ---- 参考 Forward ----
    print("  Running reference attention...")
    out_ref = reference_attention(q_ref, k_ref, v_ref, is_causal, sm_scale)
    print(f"  out_ref.shape:     {out_ref.shape}")

    # ---- Forward 对比 ----
    fwd_ok = check_close("Forward", out, out_ref)
    if not fwd_ok:
        print("  ⚠ Forward 失败，跳过 backward 测试")
        return False

    # ---- Backward ----
    do = torch.randn(B, L, H, D, dtype=dtype, device=device)

    print("\n  Running FlashMask V2 backward...")
    dq, dk, dv = flashmask_pybind.flashmask_v2_bwd(
        q_fm, k_fm, v_fm, out, softmax_lse,
        startend, block_mask, do, sm_scale, is_causal,
    )
    print(f"  dq.shape: {dq.shape}, dk.shape: {dk.shape}, dv.shape: {dv.shape}")

    # 参考 backward
    print("  Running reference backward...")
    out_ref.backward(do)
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    bwd_ok = True
    for name, a, b in [("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)]:
        if not check_close(name, a, b):
            bwd_ok = False

    return bwd_ok


def main():
    parser = argparse.ArgumentParser(description="Verify FlashMask V2 integration")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=64, help="Number of heads")
    parser.add_argument("--L", type=int, default=4096, help="Sequence length")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"], help="Data type")
    parser.add_argument("--mode", type=str, default="causal_via_flag",
                        choices=["causal_via_flag", "causal_via_mask", "both"],
                        help="Mask mode (causal_via_flag 是 Megatron 集成的主路径)")
    args = parser.parse_args()

    print(f"✓ flashmask_pybind 已导入")
    print(f"  可用函数: {[x for x in dir(flashmask_pybind) if 'flashmask' in x.lower()]}")

    if not torch.cuda.is_available():
        print("✗ CUDA 不可用!")
        sys.exit(1)

    p = torch.cuda.get_device_properties(0)
    print(f"✓ GPU: {p.name} (SM {p.major}.{p.minor})")
    if p.major < 9:
        print("⚠ FlashMask V2 需要 SM90+")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    modes = ["causal_via_flag", "causal_via_mask"] if args.mode == "both" else [args.mode]

    all_ok = True
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"配置: B={args.B}, H={args.H}, L={args.L}, D={args.D}, "
              f"dtype={args.dtype}, mode={mode}")
        print(f"{'='*60}")

        try:
            ok = run_test(args.B, args.H, args.L, args.D, dtype, mode)
        except Exception as e:
            print(f"\n✗ [{mode}] 测试出错: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False
            continue
        if not ok:
            all_ok = False

    print(f"\n{'='*60}")
    print("✅ 所有测试通过!" if all_ok else "❌ 部分测试失败")
    print(f"{'='*60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
