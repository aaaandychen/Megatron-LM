# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import flashmask_pybind

    HAVE_FLASHMASK = True
except ImportError:
    HAVE_FLASHMASK = False


# ---------------------------------------------------------------------------
# Custom autograd Function: 将 FlashMask V2 的 fwd/bwd 接入 PyTorch autograd
# ---------------------------------------------------------------------------

class _FlashMaskAttentionFunc(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for FlashMask V2.

    Forward  → flashmask_pybind.flashmask_v2_fwd()
    Backward → flashmask_pybind.flashmask_v2_bwd()

    所有 tensor 均为 FlashMask 格式 [B, H, L, D].
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        startend: Tensor,
        block_mask: Tensor,
        softmax_scale: float,
        is_causal: bool,
    ) -> Tensor:
        # 前向: 调用 FlashMask V2 CUDA kernel
        out, softmax_lse = flashmask_pybind.flashmask_v2_fwd(
            q, k, v, startend, block_mask, softmax_scale, is_causal,
        )

        # 保存 backward 所需 tensor
        ctx.save_for_backward(q, k, v, out, softmax_lse, startend, block_mask)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        q, k, v, out, softmax_lse, startend, block_mask = ctx.saved_tensors

        dq, dk, dv = flashmask_pybind.flashmask_v2_bwd(
            q, k, v, out, softmax_lse,
            startend, block_mask,
            grad_output,
            ctx.softmax_scale,
            ctx.is_causal,
        )

        # autograd 要求返回与 forward 参数个数一致的 grad（非 tensor 参数返回 None）
        return dq, dk, dv, None, None, None, None


# ---------------------------------------------------------------------------
# FlashMaskAttention — Megatron Module
# ---------------------------------------------------------------------------

class FlashMaskAttention(MegatronModule):
    """
    FlashMask V2 attention backend — 使用自研 flash attention fwd/bwd 算子。

    FlashMask V2 额外支持 block-level 稀疏 attention mask，通过
    startend_row_indices + block_mask 两个张量指定每个 key 位置的
    可见行区间和 block 级别的稀疏模式。

    约定差异:
      - Megatron core:    Q/K/V  shape [S, B, H, D]
      - FlashMask V2:     Q/K/V  shape [B, L, H, D]  (batch, seqlen, heads, dim)

    限制:
      - 仅 SM90+ (H100/H800)
      - 暂不支持 dropout, CP, packed_sequence
      - 暂不支持 learnable softmax offset
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        if not HAVE_FLASHMASK:
            raise ImportError(
                "flashmask_pybind is not installed. "
                "Build it from csrc/flashmask_v2/ following BUILD_GUIDE.md, "
                "then ensure the .so is on PYTHONPATH."
            )

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "FlashMaskAttention does not support context parallelism."

        assert (
            config.attention_dropout == 0.0
            and (attention_dropout is None or attention_dropout == 0.0)
        ), "FlashMaskAttention does not support dropout."

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "FlashMaskAttention pg_collection must have tp process group"
        self.pg_collection = pg_collection

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = projection_size // world_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = max(config.num_attention_heads // world_size, 1)
        self.num_query_groups_per_partition = max(config.num_query_groups // world_size, 1)

        # Softmax scale
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        # GQA: num_query_heads / num_kv_heads
        self.gqa_ratio = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )

        # QK-clip 兼容（FlashMask 暂不支持，设置为 None 表示禁用）
        self.current_max_attn_logits = None

    # ---- shape helpers ----
    # FlashMask V2 期望: [B, L, H, D]  (batch, seqlen, heads, dim)
    # Megatron core:       [S, B, H, D]  (seqlen, batch, heads, dim)

    @staticmethod
    def _to_flashmask(t: Tensor) -> Tensor:
        """[S, B, H, D] → [B, S, H, D]"""
        return t.permute(1, 0, 2, 3).contiguous()

    @staticmethod
    def _from_flashmask(t: Tensor) -> Tensor:
        """[B, S, H, D] → [S, B, H, D]"""
        return t.permute(1, 0, 2, 3).contiguous()

    # ---- mask construction ----

    def _build_flashmask_masks(
        self,
        attention_mask: Optional[Tensor],
        B: int,
        H: int,
        S: int,
        attn_mask_type: AttnMaskType,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """
        将 Megatron attention_mask 转换为 FlashMask V2 所需的参数。

        关键规则:
          - is_causal=True 时，startend 全部设为哨兵值 L，
            由 kernel 内部处理因果 mask，block_mask 仅辅助 tile scheduler。
          - is_causal=False 时，startend + block_mask 完整编码 mask。

        Returns:
            startend: [B, H, S, 4] int32, 哨兵 S 表示区间不活跃
            block_mask: [B, H, m_blocks, n_blocks] int32
        """
        L = S
        block_size = 128
        m_blocks = (L + block_size - 1) // block_size
        n_blocks = (L + block_size - 1) // block_size

        is_causal = attn_mask_type == AttnMaskType.causal

        if attention_mask is None and is_causal:
            # 纯 causal: startend 全哨兵，kernel 自己处理因果
            startend = torch.full((B, H, L, 4), L, dtype=torch.int32, device=device)
            block_mask = torch.zeros(B, H, m_blocks, n_blocks, dtype=torch.int32, device=device)
            for i in range(m_blocks):
                for j in range(min(i + 1, n_blocks)):
                    block_mask[:, :, i, j] = 1
            return startend, block_mask

        if attention_mask is None:
            # no_mask: 全可见
            startend = torch.full((B, H, L, 4), L, dtype=torch.int32, device=device)
            block_mask = torch.ones(B, H, m_blocks, n_blocks, dtype=torch.int32, device=device)
            return startend, block_mask

        # ---- 有自定义 attention_mask ----
        # 统一到 [S, S] bool
        if attention_mask.dim() == 4:
            am = attention_mask[0, 0]
        elif attention_mask.dim() == 3:
            am = attention_mask[0]
        else:
            raise ValueError(f"Unexpected attention_mask ndim: {attention_mask.dim()}")

        # Megatron mask: True = MASKED
        # FlashMask startend: 定义被 MASK 的行区间 [lt_start, lt_end)（不是可见区间）
        startend = torch.full((B, H, L, 4), L, dtype=torch.int32, device=device)
        for col in range(L):
            masked = am[:, col].nonzero(as_tuple=True)[0]
            if len(masked) > 0:
                startend[:, :, col, 0] = masked[0].item()
                startend[:, :, col, 1] = masked[-1].item() + 1

        # block_mask: 标记哪些 tile 有任何被 mask 的行（kernel 用于调度的 hint）
        block_mask = torch.zeros(B, H, m_blocks, n_blocks, dtype=torch.int32, device=device)
        # am_visible = ~am（True = 可见）
        am_visible = ~am
        for i in range(m_blocks):
            rs = i * block_size
            re = min(rs + block_size, L)
            for j in range(n_blocks):
                cs = j * block_size
                ce = min(cs + block_size, L)
                if am_visible[rs:re, cs:ce].any():
                    block_mask[:, :, i, j] = 1

        return startend, block_mask

    # ---- forward ----

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            query:          [S, B, num_q_heads_per_partition, head_dim]
            key:            [S, B, num_kv_heads_per_partition, head_dim]
            value:          [S, B, num_kv_heads_per_partition, head_dim]
            attention_mask: [1, 1, S, S] bool or None
            attn_mask_type: AttnMaskType enum
            attention_bias: 不支持，必须为 None
            packed_seq_params: 不支持，必须为 None

        Returns:
            context: [S, B, hidden_size_per_partition]
        """
        assert packed_seq_params is None, (
            "FlashMaskAttention does not support packed sequences (THD format)."
        )
        assert attention_bias is None, (
            "FlashMaskAttention does not support attention_bias."
        )

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type

        S, B = query.size(0), query.size(1)

        # GQA: replicate key/value to match query heads
        if self.gqa_ratio > 1:
            key = key.repeat_interleave(self.gqa_ratio, dim=2)
            value = value.repeat_interleave(self.gqa_ratio, dim=2)

        q_heads = query.size(2)  # num attn heads per partition after GQA

        # ---- Shape: [S,B,H,D] → [B,H,S,D] ----
        q_fm = self._to_flashmask(query)
        k_fm = self._to_flashmask(key)
        v_fm = self._to_flashmask(value)
        device = q_fm.device

        # ---- 构造 mask 参数 ----
        startend, block_mask = self._build_flashmask_masks(
            attention_mask, B, q_heads, S, attn_mask_type, device,
        )

        # ---- 调用自定义 autograd Function（自动处理 backward） ----
        # is_causal 仅在无显式 attention_mask 且类型为 causal 时才为 True，
        # 让 FlashMask kernel 内部处理因果 mask（与 startend=哨兵 配合）。
        # 当有自定义 attention_mask 时，mask 已经完全编码在 startend + block_mask 中。
        is_causal = (attn_mask_type == AttnMaskType.causal and attention_mask is None)

        out_fm = _FlashMaskAttentionFunc.apply(
            q_fm, k_fm, v_fm, startend, block_mask,
            self.softmax_scale, is_causal,
        )

        # ---- Shape: [B,H,S,D] → [S,B,H,D] → flatten ----
        context = self._from_flashmask(out_fm)  # [S, B, H, D]
        new_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.reshape(*new_shape)

        return context
