import sys
import traceback
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm import _custom_ops as ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_nope_cache_insert() -> None:
    torch.manual_seed(7)
    kv_c = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.empty(4, 0, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(4, device="cuda", dtype=torch.int64)
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, "bfloat16", scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(cache[0, :4], kv_c, rtol=0, atol=0)


def _cache_shape(cache_dtype: str) -> tuple[int, ...]:
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseSM120Backend,
    )

    return FlashInferMLASparseSM120Backend.get_kv_cache_shape(
        num_blocks=72,
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str=cache_dtype,
    )


def test_auto_nope_cache_shape() -> None:
    shape = _cache_shape("auto")
    assert shape == (72, 64, 512), shape


def test_bf16_nope_cache_shape() -> None:
    shape = _cache_shape("bfloat16")
    assert shape == (72, 64, 512), shape


def test_bf16_nope_cache_dtype_is_advertised() -> None:
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseSM120Backend,
    )

    assert "bfloat16" in FlashInferMLASparseSM120Backend.supported_kv_cache_dtypes


def _validate_bf16_backend(
    qk_rope_head_dim: int,
    *,
    has_sparse_api: bool,
    dcp_size: int = 1,
) -> list[str]:
    import vllm.config as config_module
    from vllm.platforms.interface import DeviceCapability
    from vllm.utils import flashinfer as flashinfer_utils
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseSM120Backend,
    )

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                index_topk=2048,
                qk_nope_head_dim=256,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=512,
            ),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
    )
    with (
        mock.patch.object(config_module, "get_current_vllm_config", return_value=config),
        mock.patch.object(
            flashinfer_utils,
            "has_flashinfer_sparse_mla_sm120",
            return_value=has_sparse_api,
        ),
    ):
        return FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="bfloat16",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type=AttentionType.DECODER,
        )


def test_bf16_nope_backend_configuration_is_valid() -> None:
    invalid_reasons = _validate_bf16_backend(0, has_sparse_api=False)
    assert invalid_reasons == []


def test_bf16_unsupported_backend_configurations_are_rejected() -> None:
    invalid_reasons = _validate_bf16_backend(
        0,
        has_sparse_api=False,
        dcp_size=2,
    )
    assert any("decode context parallelism" in reason for reason in invalid_reasons), (
        invalid_reasons
    )

    invalid_reasons = _validate_bf16_backend(64, has_sparse_api=True)
    assert any("native NoPE" in reason for reason in invalid_reasons), invalid_reasons


def test_fp8_nope_cache_shape() -> None:
    shape = _cache_shape("fp8_ds_mla")
    assert shape == (72, 64, 656), shape


def test_bf16_nope_constructor_accepts_only_nope() -> None:
    import vllm.config as config_module
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120

    topk = torch.arange(2048, dtype=torch.int32).repeat(2, 1)
    indexer = SimpleNamespace(topk_indices_buffer=topk)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm53"),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    def construct(qk_rope_head_dim: int):
        return sm120.FlashInferMLASparseSM120Impl(
            num_heads=16,
            head_size=512 + qk_rope_head_dim,
            scale=0.125,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="bfloat16",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            indexer=indexer,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=qk_rope_head_dim,
        )

    original_get_config = config_module.get_current_vllm_config
    try:
        config_module.get_current_vllm_config = lambda: config
        impl = construct(0)
        try:
            construct(64)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("BF16 SM120 compatibility requires native NoPE")
        config.parallel_config.decode_context_parallel_size = 2
        try:
            construct(0)
        except NotImplementedError as error:
            assert "decode context parallelism" in str(error)
        else:
            raise AssertionError("BF16 NoPE fallback must reject DCP")
    finally:
        config_module.get_current_vllm_config = original_get_config

    assert impl.kv_cache_dtype == "bfloat16"
    assert impl.kv_lora_rank == 512
    assert impl.qk_nope_head_dim == 256
    assert impl.qk_rope_head_dim == 0
    assert impl.topk_indices_buffer is topk


def test_bf16_nope_forward_routes_to_fallback() -> None:
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120

    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.kv_cache_dtype = "bfloat16"
    impl.qk_rope_head_dim = 0
    impl.scale = 0.125
    impl.topk_indices_buffer = torch.tensor([[2, 1, 0], [0, 1, 2]])

    query = torch.randn(2, 3, 512, dtype=torch.bfloat16)
    cache = torch.randn(2, 64, 512, dtype=torch.bfloat16)
    req_ids = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        req_id_per_token=req_ids,
        block_table=block_table,
        block_size=64,
    )
    physical_indices = torch.tensor([[2, 1, 0], [64, 65, 66]], dtype=torch.int32)
    valid_counts = torch.tensor([3, 2], dtype=torch.int32)
    fallback_output = torch.full_like(query, 3)
    calls = {}

    def fake_convert(req_ids_arg, block_table_arg, topk_arg, **kwargs):
        assert kwargs.get("return_valid_counts") is True, kwargs
        calls["convert"] = (req_ids_arg, block_table_arg, topk_arg, kwargs)
        return physical_indices, valid_counts

    def fake_fallback(query_arg, cache_arg, indices_arg, counts_arg, scale_arg):
        calls["fallback"] = (
            query_arg,
            cache_arg,
            indices_arg,
            counts_arg,
            scale_arg,
        )
        return fallback_output

    missing = object()
    original_convert = sm120.triton_convert_req_index_to_global_index
    original_fallback = getattr(sm120, "bf16_nope_sparse_attention", missing)
    try:
        sm120.triton_convert_req_index_to_global_index = fake_convert
        sm120.bf16_nope_sparse_attention = fake_fallback
        output, lse = impl.forward_mqa(query, cache, metadata, None)
    finally:
        sm120.triton_convert_req_index_to_global_index = original_convert
        if original_fallback is missing:
            del sm120.bf16_nope_sparse_attention
        else:
            sm120.bf16_nope_sparse_attention = original_fallback

    converted_req_ids, converted_table, converted_topk, kwargs = calls["convert"]
    assert torch.equal(converted_req_ids, req_ids)
    assert converted_table is block_table
    assert torch.equal(converted_topk, impl.topk_indices_buffer)
    assert kwargs == {
        "BLOCK_SIZE": 64,
        "NUM_TOPK_TOKENS": 3,
        "return_valid_counts": True,
    }
    routed_query, routed_cache, routed_indices, routed_counts, routed_scale = calls[
        "fallback"
    ]
    assert routed_query is query
    assert routed_cache is cache
    assert routed_indices is physical_indices
    assert routed_counts is valid_counts
    assert routed_scale == impl.scale
    assert output is fallback_output
    assert lse is None


def test_glm53_kpool_alignment_uses_sm120_fp8_page_size() -> None:
    from vllm.platforms.cuda import CudaPlatform
    from vllm.platforms.interface import DeviceCapability

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_kpool=4),
        ),
    )
    with mock.patch.object(
        CudaPlatform,
        "get_device_capability",
        return_value=DeviceCapability(12, 0),
    ):
        alignment = CudaPlatform._get_indexer_block_alignment(config)
    assert alignment == 256, alignment


def test_virtual_kpool_storage_uses_sm120_supported_page() -> None:
    from vllm.v1.worker import utils as worker_utils

    kpool_spec = SimpleNamespace(storage_block_size=320)
    with (
        mock.patch.object(
            worker_utils.current_platform,
            "is_device_capability_family",
            side_effect=lambda major: major == 120,
        ),
        # The global list may include pages supported by other architectures;
        # SM120 non-FP4 paged MQA specifically requires block_kv=64.
        mock.patch.object(worker_utils, "PAGED_MQA_PAGE_SIZES", (32, 64, 128)),
    ):
        assert worker_utils.compressed_kernel_block_size(kpool_spec) == 64


def _reference_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    physical_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    width = physical_indices.shape[1]
    valid = torch.arange(width, device=query.device)[None, :] < valid_counts[:, None]
    nonempty = valid_counts > 0
    safe_indices = torch.where(valid, physical_indices, 0).long()
    keys = cache.reshape(-1, cache.shape[-1])[safe_indices].float()
    scores = torch.bmm(query.float(), keys.transpose(1, 2)) * scale
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    scores = torch.where(nonempty[:, None, None], scores, torch.zeros_like(scores))
    probs = torch.softmax(scores, dim=-1)
    output = torch.bmm(probs, keys)
    output.masked_fill_(~nonempty[:, None, None], 0.0)
    return output.to(query.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_sparse_attention_matches_reference() -> None:
    from vllm.v1.attention.backends.mla.glm53_nope_fallback import (
        bf16_nope_sparse_attention,
    )

    torch.manual_seed(11)
    query = torch.randn(3, 4, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    physical_indices = torch.tensor(
        [[0, 62, 63, 64, 127], [1, 63, 64, -1, -1], [-1, -1, -1, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    valid_counts = torch.tensor([5, 3, 0], device="cuda", dtype=torch.int32)
    scale = 1.0 / 16.0

    expected = _reference_attention(query, cache, physical_indices, valid_counts, scale)
    actual = bf16_nope_sparse_attention(
        query,
        cache,
        physical_indices,
        valid_counts,
        scale,
        chunk_size=2,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0)
    assert torch.isfinite(actual).all()


if __name__ == "__main__":
    tests = (
        test_bf16_nope_cache_insert,
        test_auto_nope_cache_shape,
        test_bf16_nope_cache_shape,
        test_bf16_nope_cache_dtype_is_advertised,
        test_bf16_nope_backend_configuration_is_valid,
        test_bf16_unsupported_backend_configurations_are_rejected,
        test_fp8_nope_cache_shape,
        test_bf16_nope_constructor_accepts_only_nope,
        test_bf16_nope_forward_routes_to_fallback,
        test_glm53_kpool_alignment_uses_sm120_fp8_page_size,
        test_virtual_kpool_storage_uses_sm120_supported_page,
        test_chunked_sparse_attention_matches_reference,
    )
    failures = 0
    for test in tests:
        try:
            test()
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc(file=sys.stdout)
        else:
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise SystemExit(1)
