from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TritonKernelTopKOutput,
    _mask_topk_ids_padded_region,
)


def _to_triton_kernels_format(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,
    num_token_non_padded: Optional[torch.Tensor],
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo],
) -> TritonKernelTopKOutput:
    """Convert HashTopK's StandardTopKOutput to TritonKernelTopKOutput.

    Re-uses HashTopK's already-computed `topk_ids` to derive the bitmatrix,
    and threads HashTopK's `topk_weights` through `routing_from_bitmatrix`
    so they survive into `RoutingData.gate_scal`. We do NOT call
    `routing(router_logits, expt_indx=topk_ids)` because that path
    re-derives weights from selected-logit softmax, discarding HashTopK's
    scoring (per gpt-5.5 review NOT APPROVED for the naive path).
    """
    assert num_token_non_padded is None, (
        "HashTopK->TritonKernel conversion does not support padded-region "
        "masking; -1 indices in topk_ids would corrupt the bitmatrix"
    )
    assert expert_location_dispatch_info is None, (
        "HashTopK->TritonKernel conversion does not support EPLB "
        "(logical->physical ID remap); router_logits columns are logical, "
        "gathering via physical IDs would index wrong columns"
    )

    from triton_kernels.routing import routing_from_bitmatrix
    from triton_kernels.topk import topk_forward

    # Uniqueness within each token's real top-k is required: routing_from_
    # bitmatrix derives the per-expert histogram from the bitmatrix (binary
    # per (token, expert)) but reads the gate list as N*k entries. Duplicates
    # would make hist undercount vs gate list and corrupt routing offsets.
    # HashTopK's tid2eid table is designed to dispatch each token to distinct
    # experts; assert it for clean failure on violation.
    sorted_ids, _ = topk_ids.sort(dim=1)
    assert not (
        sorted_ids[:, 1:] == sorted_ids[:, :-1]
    ).any(), "HashTopK->TritonKernel conversion requires unique expert IDs per token"

    # Power-of-2 padding for triton-kernels routing kernels.
    # Both `topk.topk_forward` and `routing_details._routing_compute` require
    # N_EXPTS_ACT (and `N_EXPTS_ACT * BLOCK_M`) to be powers of 2. V4-Flash
    # uses num_experts_per_tok=6, which fails both constraints.
    # Pad to next pow-2 using UNIQUE expert IDs not already in each token's
    # real top-k, with zero weight. This preserves the bitmatrix histogram /
    # gate-list invariant (each token gets exactly k_pow2 unique bits set,
    # and the gate list has exactly k_pow2 entries that match), while
    # contributing zero to the MoE matmul output (gate_scal=0). Cost is
    # ~k_pow2/k extra GEMM work.
    n_expts_act_pow2 = 1 << max(0, n_expts_act - 1).bit_length()
    if n_expts_act_pow2 != n_expts_act:
        pad = n_expts_act_pow2 - n_expts_act
        n_tokens = topk_ids.shape[0]
        # Find `pad` unused expert IDs per token. Build a [N, n_expts_tot]
        # mask of used experts, then take the top-`pad` indices of (1 - mask):
        # values 1 (unused) sort above 0 (used), so torch.topk returns
        # `pad` distinct unused IDs per row. Identity of the chosen IDs does
        # not matter — only that they are unused (so bitmatrix has k_pow2
        # bits set per token) and distinct (so the gate list and bitmatrix
        # describe the same routing).
        used = torch.zeros(
            n_tokens, n_expts_tot, dtype=torch.bool, device=topk_ids.device
        )
        used.scatter_(1, topk_ids.long(), True)
        _, pad_ids = torch.topk((~used).to(torch.int32), k=pad, dim=1)
        pad_ids = pad_ids.to(topk_ids.dtype)
        pad_weights = torch.zeros(
            n_tokens, pad, dtype=topk_weights.dtype, device=topk_weights.device
        )
        topk_weights = torch.cat([topk_weights, pad_weights], dim=-1)
        topk_ids = torch.cat([topk_ids, pad_ids], dim=-1)
        n_expts_act = n_expts_act_pow2

    y_indx_i16 = topk_ids.to(torch.int16)
    _, _, bitmatrix = topk_forward(
        router_logits,
        n_expts_act,
        apply_softmax=False,
        y_indx=y_indx_i16,
    )
    routing_data, gather_idx, scatter_idx = routing_from_bitmatrix(
        bitmatrix,
        topk_weights,
        y_indx_i16,
        n_expts_tot,
        n_expts_act,
    )
    return TritonKernelTopKOutput(routing_data, gather_idx, scatter_idx)


class HashTopK(nn.Module):
    def __init__(
        self,
        topk,
        num_experts,
        num_fused_shared_experts,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),
            requires_grad=False,
        )

        if get_bool_env_var("SGLANG_HACK_TID2EID_INIT_ZERO"):
            print("hack: tid2eid init to zero")
            nn.init.constant_(self.tid2eid, 0)

        assert not apply_routed_scaling_factor_on_output, "not implemented"

    def empty_topk_output(self, device: torch.device):
        topk = self.topk - self.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    def _forward_torch(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = torch.nn.functional.softplus(router_logits).sqrt()

        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_fused_shared_experts == 1:
            # Hash MoE: get routed expert IDs and weights
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

            # reference: biased_grouped_topk_impl in topk.py
            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_fused_shared_experts,
                size=(num_token,),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )

            # don't apply routed scaling factor here
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor
            )
        else:
            topk_ids[:, :] = self.tid2eid[input_ids]
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
            if self.score_func != "softmax":
                topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        assert (
            input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0]
        ), f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"

        if envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.jit_kernel.deepseek_v4 import hash_topk

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_fused_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)

        if is_hip():
            topk_weights = topk_weights.to(torch.float32)

        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)

        if get_moe_runner_backend().is_triton_kernels():
            assert self.num_fused_shared_experts == 0, (
                "HashTopK->TritonKernel conversion does not support fused "
                "shared experts (IDs >= n_routed_experts)"
            )
            return _to_triton_kernels_format(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                n_expts_tot=router_logits.shape[-1],
                n_expts_act=topk_weights.shape[-1],
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )

        return StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def biased_topk_jit_kernel_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

    topk_weights, topk_ids = moe_fused_gate(
        gating_output,
        correction_bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids
