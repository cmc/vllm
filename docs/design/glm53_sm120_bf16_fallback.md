# GLM-5.3 SM120 BF16 NoPE fallback

## Scope and status

This change is a correctness-first, temporary compatibility path for serving
GLM-5.3-Flash on NVIDIA SM120. It preserves the model's native NoPE attention
semantics while a compatible fused sparse-MLA implementation is unavailable.
It is not an upstream-ready performance design.

The source baseline is commit
`933876c388fb129ad82590660e6506614559cb86`. The runtime overlay is built from
the official image
`vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703`.
Validation used four RTX PRO 6000 Blackwell Max-Q GPUs with compute capability
SM120.

The patch leaves the checkpoint and global FP8 quantization configuration
unchanged, while selecting an explicit BF16 KV cache. GLM-5.3 is a
mixed-precision runtime: model-specific MLA projection handling can load or
execute tensors in BF16. This patch neither changes that weight-loading
behavior nor converts the whole model to BF16.

## Architecture and ABI mismatch

GLM-5.3 uses native NoPE with:

- `qk_nope_head_dim = 256`
- `qk_rope_head_dim = 0`
- `kv_lora_rank = 512`
- absorbed decode query/cache width = 512 with no RoPE tail

The SM120 `fp8_ds_mla` path has a different fixed ABI. It expects
`pe_dim = 64`, a 576-wide query, and a 656-byte packed cache entry. That packed
entry contains 512 NoPE bytes, 16 bytes of scaling metadata, and 128 bytes for
the 64-element RoPE tail. It is not an unpacked 656-element BF16 tensor.

Consequently, the existing packed kernel cannot correctly consume GLM-5.3's
512-wide absorbed NoPE tensors. The fallback preserves the logical model layout
instead of inventing a new packed ABI.

## Failure chronology

1. The official vLLM image selected the SM120 sparse-MLA path but failed with
   `pe_dim must be 64 for fp8_ds_mla` for GLM-5.3's
   `qk_rope_head_dim=0`. The packed kernel requires `pe_dim=64`, a 576-wide
   query, and a 656-byte packed entry.
2. Selecting BF16 alone did not change cache allocation. `auto` still selected
   the packed FP8 shape, leading to an invalid `[72, 64, 656]` BF16 reshape
   instead of an unpacked `[72, 64, 512]` cache.
3. Correcting the logical BF16 cache shape exposed a DeepGEMM SM120 assertion:
   non-FP4 compressed kpool pages require `block_kv=64`.
4. Disabling the explicit prefix-caching flag was insufficient. Prefix-caching
   policy does not change the indexer block alignment, virtual page selection,
   or physical compressed-kpool page size; those required source changes.
5. The first unified indexer patch did not apply to the official image because
   its context did not match. The deployed overlay therefore used a strict
   build-time replacement script that required the expected source fragments
   before replacing them. This fork expresses the same changes directly in
   native source.
6. SGLang was investigated and rejected after a sequence of independent
   blockers: stale Transformers GLM registration, a Qwen3-ASR registration
   collision after upgrading Transformers, unsupported tensor-parallel handling
   for `mla_kv_a_proj`, and an unhandled zero head dimension for native NoPE.
7. CUDA's expandable-segment allocator emitted transient OOM mapping warnings
   while weights filled the devices. They are not omitted from the record:
   despite the warnings, weight loading and engine profiling completed, the API
   became healthy, and the container remained running without restarts.

The first five items are resolved by this fallback and its native-source
publication. The SGLang work was a dead end, and the expandable-segment
messages were nonfatal warnings rather than a failed engine.

## Compatibility design

### Logical BF16 cache and narrow backend gate

Explicit FP8 cache dtypes retain the existing packed
`(num_blocks, block_size, 656)` shape. `auto` and explicit `bfloat16` cache-shape
queries use the logical
`(num_blocks, block_size, head_size)` shape; for GLM-5.3 this is
`(num_blocks, block_size, 512)`. The deployed compatibility configuration uses
an explicit BF16 KV cache.

The SM120 backend's dtype list advertises BF16, while combination selection
accepts it only for a resolved model configuration with
`qk_rope_head_dim == 0`. That BF16 NoPE combination does not require
FlashInfer's sparse SM120 API because it is implemented in PyTorch. The
implementation constructor repeats the same gate, so BF16 with a nonzero RoPE
dimension is rejected before execution. The existing `fp8_ds_mla` route and
its FlashInfer API requirement remain unchanged.

### Physical sparse indices and reference attention

The indexer produces virtual per-request top-k indices. Before fallback
attention, vLLM converts them through each request's block table into physical
cache indices and also returns a valid count for every query token. Valid
counts distinguish real selections from padded or invalid slots without
discarding any of the model's 2,048 selected sparse tokens.

The reference path flattens the unpacked BF16 cache, gathers keys by physical
index, masks entries beyond each valid count, and performs the two attention
matrix products with `torch.bmm`. It processes query tokens in chunks to bound
temporary gather, score, and probability tensors. Empty selections return
finite zeros. This is intentionally an unfused reference path.

### SM120 kpool pages

DeepGEMM's SM120 non-FP4 paged-MQA implementation requires 64-entry
`block_kv` pages. The indexer alignment is therefore
`index_kpool * 64`; GLM-5.3's `index_kpool=4` produces a 256-token alignment.
Compressed kpool storage that is at least 64 entries and divisible by 64 is
virtually split into 64-entry kernel pages. For the tested 320-entry storage
block, this yields five physical kernel pages.

This is an architecture-specific contract. It should be replaced if DeepGEMM
exposes a queryable page capability.

## Benefits and costs

Benefits:

- Preserves GLM-5.3's exact 512-wide absorbed NoPE latent semantics.
- Retains all 2,048 sparse selections.
- Requires no new compiled CUDA or FlashInfer kernel.
- Uses a simple logical cache layout that can be checked numerically.
- Passed unit, startup, deterministic generation, and long-context retrieval
  validation on four SM120 GPUs.

Costs and limitations:

- BF16 uses 16 bits per element and therefore uses more KV-cache memory than
  packed FP8 or FP4 representations.
- FP4 nominally uses 4 bits per element before scales and other metadata.
  However, no correct GLM-5.3 512+0 NoPE FP4 packing, scaling, and SM120 kernel
  path was available or validated. FP4 cannot be substituted as a dtype flag.
- Indexed gathers, temporary tensors, Python chunking, and `torch.bmm` add
  memory traffic and dispatch overhead relative to a fused sparse-MLA kernel.
- The 64-entry page rule is hard-coded for the current SM120 non-FP4 DeepGEMM
  contract.
- The branch uses the historical GLM integration commit because later vLLM
  revisions have incompatible cache and page APIs.

These KV-cache trade-offs do not change the checkpoint/global FP8 quantization
configuration or the model's existing mixed-precision weight handling.

## Validation evidence

### Runtime validation

- Served context: 262,144 tokens, confirmed by the live model metadata and the
  tokenizer endpoint used by the long-context test.
- Measured GPU KV-cache capacity: 975,077 tokens.
- Startup: engine profiling completed, the service and container became
  healthy, and the container remained at zero restarts despite the recorded
  expandable-segment OOM warnings.
- Determinism: two seeded, temperature-zero generations returned identical
  normalized outputs.
- Short generation: the live 262K service returned the requested
  `PATCH_OK`.
- OpenCode integration: resolved model configuration reported a 262,144-token
  context and default-model generation returned `OPENCODE_OK`.
- Long-context retrieval: a 239,994-token prompt placed an early needle near
  the beginning of the request; generation returned exactly
  `RIVENDELL-262K-PASS`. The service and container remained healthy afterward.

### Focused test behavior

The unpatched official-image baseline ran all 12 tests through the aggregate
runner. It had three expected passing controls and nine patch-dependent
failures:

1. `test_bf16_nope_cache_insert` — baseline control passed: the existing BF16
   cache writer preserved inserted 512-wide values exactly.
2. `test_auto_nope_cache_shape` — baseline failed with
   `(72, 64, 656)` instead of `(72, 64, 512)`.
3. `test_bf16_nope_cache_shape` — baseline control passed: an explicit BF16
   shape query was already 512-wide.
4. `test_bf16_nope_cache_dtype_is_advertised` — baseline failed because the
   SM120 backend did not advertise `bfloat16`.
5. `test_bf16_nope_backend_configuration_is_valid` — baseline failed when the
   FlashInfer SM120 sparse API was unavailable; the fallback must still be
   selectable for explicit BF16 NoPE.
6. `test_bf16_rope_backend_configuration_is_rejected` — baseline failed to
   report the required native-NoPE rejection when that FlashInfer API was
   available; explicit BF16 with `qk_rope_head_dim=64` must be rejected.
7. `test_fp8_nope_cache_shape` — baseline control passed: explicit
   `fp8_ds_mla` remained `(72, 64, 656)`.
8. `test_bf16_nope_constructor_accepts_only_nope` — baseline rejected the BF16
   NoPE constructor; the patched constructor accepts the real
   `qk_nope_head_dim=256`, `kv_lora_rank=512`, zero-RoPE configuration and
   rejects nonzero RoPE.
9. `test_bf16_nope_forward_routes_to_fallback` — baseline did not request
   `return_valid_counts=True` or route physical indices and counts to the
   fallback.
10. `test_glm53_kpool_alignment_uses_sm120_fp8_page_size` — baseline returned
    128 instead of the required 256-token alignment.
11. `test_virtual_kpool_storage_uses_sm120_supported_page` — baseline selected
    32 rather than the required 64-entry page for 320-entry storage.
12. `test_chunked_sparse_attention_matches_reference` — baseline lacked the
    fallback module; the patched test also checks the 63/64 physical page
    boundary, a partially valid row, an empty row, finite outputs, and agreement
    with a float32 reference at `rtol=0.02, atol=0.02`.

The current source overlay passes all 12 of 12 tests.

## Reproduction

Run these commands from the repository root. The image is derived directly from
the pinned official image and replaces only the five runtime source files named
in the Dockerfile.

```bash
docker build \
  --platform linux/amd64 \
  --file docker/Dockerfile.glm53-sm120-bf16-fallback \
  --tag local/vllm-glm53:sm120-nope-bf16-source \
  .
```

Run the focused source-overlay test on one GPU:

```bash
docker run --rm --gpus device=0 \
  -v "$PWD/tests/v1/attention:/tests:ro" \
  --entrypoint /usr/bin/python3 \
  local/vllm-glm53:sm120-nope-bf16-source \
  /tests/test_glm53_nope_sm120.py
```

To reproduce the expected baseline result with the same test:

```bash
docker run --rm --gpus device=0 \
  -v "$PWD/tests/v1/attention:/tests:ro" \
  --entrypoint /usr/bin/python3 \
  vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703 \
  /tests/test_glm53_nope_sm120.py
```

The baseline command is expected to exit nonzero after reporting all three
controls as passed and all nine compatibility assertions as failed.

## Non-goals

- No FP4 cache implementation.
- No fused GLM-5.3 NoPE kernel.
- No multi-token prediction (MTP) validation.
- No claim that this path has upstream-quality performance.

## Rejected alternatives

- **FlashInfer upgrade alone:** an API upgrade did not provide a validated
  GLM-5.3 512+0 SM120 cache ABI and kernel.
- **Delete the `pe_dim == 64` guard:** bypassing the guard would pass
  incompatible shapes to a kernel with a fixed 64-element RoPE ABI; it would
  not make the operation correct.
- **Invent a 528-byte packed format:** 512 NoPE bytes plus 16 scale bytes is a
  plausible size, but no 528-byte ABI, scaling contract, writer, or reader
  exists for this kernel. Treating it as defined would risk incorrect memory
  interpretation.
- **SGLang:** the Transformers registration, Qwen3-ASR collision,
  `mla_kv_a_proj`, and zero-head-dimension failures made it a separate,
  unresolved port rather than a viable fallback.
- **Claim FP4 or MTP support:** neither path was implemented or validated.

## Security and publication

This note and the reproducible overlay contain no credentials, access tokens,
private endpoints, or private runtime configuration. The published artifact is
limited to the compatibility source, focused tests, Dockerfile, and this design
record.
