# NInfer

**English** | [简体中文](README.zh-CN.md)

> Tensor-parallel NInfer on two consumer cards. Qualified on **2× RTX 5060 Ti (16 GiB each)**: one
> 27B model resident across both GPUs, a **253,952-token single-slot context**, five KV-cache tiers
> (`bf16` / `int8` / `k16i8`), MTP3 speculative decoding with prefix reuse that actually
> hits, and a `/health` that reports engine availability.

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints. It runs
text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs, on one
GPU, or -- for the 27B execution package -- tensor-parallel across two: see
[Dual-GPU (TP2)](#dual-gpu-tp2). The paragraphs below describe upstream; what this fork adds and
measures is in the fork note.

> **This is a fork.** Upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer); this tree
> branches from its commit `feaf4dd` via the TP2 line
> ([wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m),
> [giocom/ninfer-3060X2](https://github.com/giocom/ninfer-3060X2)) and, on that base, adds **dual-GPU
> tensor parallelism** (`--tp 2 --devices A,B`) to the 27B execution package: one resident model, one
> process, two devices, no NVLink and no distributed serving, halving per-card weight and KV residency.
> This fork further adds **KV-cache tiers** (`--kv-dtype bf16|int8|k16i8`), a **working MTP prefix
> reuse at `--tp 2`**, and a **truthful `/health`** with supervisor-driven self-heal; those three were
> measured on **2× RTX 5060 Ti (16 GiB each)**. `--tp 1` output is byte-identical to `feaf4dd` on the
> greedy cases in [`tests/data/tp1-golden/`](tests/data/tp1-golden/MANIFEST.md), and single-GPU
> behaviour, supported identities, artifact format, and protocol surfaces are unchanged. The design
> decisions, numerical contracts, and qualification evidence are in
> [Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md) -- that document is
> upstream's, and also covers the YaRN extended-position work this fork does not use.
> See [NOTICE](NOTICE) for attribution.

## The weights

This fork runs **Qwen3.8-27B NVFP4** in two interchangeable forms. Both were measured here at
`--tp 2` on 2× RTX 5060 Ti; neither fits on a single 16 GiB card.

| | official — upstream's | QUASAR W4A4 |
|---|---|---|
| artifact | [`qwen3_8_27b_nvfp4.ninfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3_8_27b_quasar_w4a4.ninfer`, repacked below |
| size | 21,492,695,040 B (20.02 GiB) | 19,782,140,416 B (18.42 GiB) |
| weights per card at `--tp 2` | 10.08 GiB | 9.87 GiB |
| quantization | NVFP4 MLP plus row-scaled FP8 elsewhere | NVFP4 throughout, including 4-bit activations (QUASAR QAT); DFlash2, MTP and Vision already inside |
| decode at `int8` KV | 76–86 tok/s (MTP3) | 107.1 tok/s (MTP3), 126.3 tok/s (dflash2 K7) |
| single slot at `int8` KV | 262144 (that artifact's native ceiling) | 253952 measured (262144 native ceiling) |
| SHA-256 | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` | `78607ff1f9bfb087a5cd85c8bc72d317e6edefc7e79bdde6e43f1ce21336f898` |

The official form is one download with nothing to convert; the QUASAR W4A4 form needs one lossless
repack step and decodes 25–47% faster on the same KV tier. Everything else in this README applies to
both — the only difference is the artifact path on the command line.

Get the official artifact:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

### The QUASAR W4A4 artifact

The W4A4 form is a repack of the published QUASAR artifact — no quantization or format conversion
is involved.

| | |
|---|---|
| Source | [`MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer`](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer) — a native `.ninfer` artifact built from the QUASAR QAT checkpoint, with DFlash2, MTP and Vision already inside |
| Source file | `qwen3_8_27b_nvfp4.ninfer`, 19,782,132,224 B, SHA-256 `da5efb3332e00ed5a9d719aa5cc09a4066fa03ab0d1706f6119f2fba8f2ba338` |
| Repack | [`tools/convert/qwen3_8_27b/repack_quasar.py`](tools/convert/qwen3_8_27b/repack_quasar.py) (Python standard library only) |
| Result | `qwen3_8_27b_quasar_w4a4.ninfer`, 19,782,140,416 B (18.42 GiB), SHA-256 `78607ff1f9bfb087a5cd85c8bc72d317e6edefc7e79bdde6e43f1ce21336f898` |

```bash
# 1. download the QUASAR artifact
hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir src/quasar

# 2. repack it onto this fork's W4A4 contract (lossless, about a minute)
python3 -m tools.convert.qwen3_8_27b.repack_quasar \
  --src src/quasar/qwen3_8_27b_nvfp4.ninfer \
  --out models/qwen3_8_27b_quasar_w4a4.ninfer
```

Why the repack exists: the QUASAR artifact declares `identity.weights_id = nvfp4`, whose registered
contract does not match its endpoint descriptors, and it keeps the 48 GDN projections fused as
`gdn/a_b_projection` where the W4A4 tier binds separate `a_projection` / `b_projection` objects. The
repack therefore rewrites the container header only: each fused GDN parent becomes two directory
views over the same payload halves, and the identity flips to `nvfp4-w4a4`. **No payload byte is
touched** — no copy, no requantization, standard library only. Run from the published source, the
in-tree repack produces a file byte-identical to the SHA-256 above.

`huggingface.co` is not reachable from every network. `huggingface_hub` honours the `HF_ENDPOINT`
environment variable, so `HF_ENDPOINT=https://hf-mirror.com hf download …` fetches the same files
through a mirror.

### Other registered artifacts

NInfer deliberately supports a closed set of artifacts rather than acting as a general model runtime.
Beyond the two forms above, the engine also registers upstream's identities — Qwen3.6-27B in both
weight profiles, Qwen3.8-27B `groupwise-int`, and Qwen3.6-35B-A3B — and accepts them; they are outside
what this fork is built and measured against. One sizing note on the official form: at its full
262,144-token `int8` slot it leaves 842 MiB free per card — the same residual under `ninfer-serve` as
under the CLI, because the media and response buffers are not reserved while vision is off. Current
builds accept only the version-2 container, and all of those are version 2.

Every `.ninfer` file contains the weights and frontend resources NInfer needs. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file. Each artifact is complete, while GPU
residency is fixed at process startup. Speculative decoding is off by default, so MTP/DFlash state and
the optimized proposal head are not uploaded; vision is off by default too, so its weights, the Vision
scratch phase and the frozen request-transient allocation are omitted. Add `--vision` to the CLI or
server process that must accept image or video input, and `--spec mtp --draft-tokens 3` to the one
that must speculate. Disabled capabilities cannot be enabled by a later request.

## Quick start

Clone **this** repository (not upstream, and not the TP2 forks it descends from):

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Serve the QUASAR W4A4 artifact from [The weights](#the-weights) on two GPUs (253,952-token single
slot, DFlash2 K7 speculative decoding with the optimized draft head). To use the official artifact
instead, swap the path — with `int8` KV it fits a full 262,144-token slot:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_quasar_w4a4.ninfer \
  --host 0.0.0.0 --port 8815 --model-id qwen3.8-27b-quasar-w4a4 \
  --tp 2 --devices 0,1 --kv-dtype int8 \
  --max-context 253952 --kv-capacity 253952 --prefill-chunk 1024 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --max-concurrency 1 --cors
```

[Requirements](#requirements), [Build](#build) and
[Dual-GPU (TP2)](#dual-gpu-tp2) cover the prerequisites, the artifact repack and the complete option
set.

## Performance

This README carries this fork's own numbers in
[KV-cache tiers and long-context limits](#kv-cache-tiers-and-long-context-limits): 2× RTX 5060 Ti,
`--tp 2`, one slot, per-tier prefill/decode/acceptance and the long-context ceilings.

Upstream's single-GPU RTX 5090 campaign (Qwen3.6-27B in both weight profiles, Qwen3.6-35B-A3B, and
the Qwen3.8-27B NVFP4 artifact) is documented in [Performance](docs/performance.md) and in the model
cards. Those figures were not measured here and are not comparable to figures taken on two 16 GiB
cards.

## Evaluation

This fork has no budget for a capability evaluation of its own, so no benchmark table is published
here. For scores, see the model repositories: the per-artifact model cards under
[`model-cards/`](model-cards/) and the artifacts' Hugging Face pages carry upstream's results
(AIME 2025/2026, GPQA-Diamond, ERQA, RealWorldQA, EvalScope 1.9.0, single sample). Those were
measured on upstream's artifacts, not on this fork's repack.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- two NVIDIA GeForce RTX 5060 Ti (16 GiB each), which is the platform this fork is built and
  measured on; the engine itself runs on any `sm_120a` device, one or two;
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below;
- Python 3 with `torch` and `safetensors`, for the conversion and verification tools under
  `tools/convert/` — the engine build itself does not need Python.

The build rejects CUDA architectures other than `120a`. There is no install target or packaged
binary distribution; NInfer is run from its source build tree.

## Build

Clone **this** repository — upstream has no `--tp 2`, and neither do the other TP2 forks carry these
KV tiers.

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Dual-GPU (TP2)

`--tp 2` splits one resident model across two GPUs: one process, one resident model, two CUDA
devices, no NVLink and no distributed serving. It is a capacity feature rather than scale-out — it
halves per-card weight and KV residency. It is implemented for the 27B execution package
(`qwen3.6-27b` and `qwen3.8-27b`, either weight profile); `qwen3.6-35b-a3b` has no tensor-parallel
path and rejects `--tp 2` at startup.

Every two-card number this README publishes was taken on **2× RTX 5060 Ti (16 GiB each)** against the
W4A4 artifact described under [The weights](#the-weights).

### Usage

```bash
# both GPUs, K16V8 KV, a 253,952-token single slot, MTP3 with the optimized proposal head
./build/apps/ninfer-serve models/qwen3_8_27b_quasar_w4a4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 253952 --kv-capacity 253952 --prefill-chunk 1024 \
  --kv-dtype k16i8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-concurrency 1
```

The same flags drive the CLI; add `--no-thinking` when a short `--max-new` budget has to reach the
answer channel, because at this checkpoint's default thinking mode a 16-token budget is consumed
entirely inside the reasoning stream:

```bash
./build/apps/ninfer models/qwen3_8_27b_quasar_w4a4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 253952 --kv-capacity 253952 --kv-dtype k16i8 \
  --messages long_prompt.json --max-new 256 --no-thinking
```

- `--tp 2` requires an explicit `--devices A,B` naming two distinct devices of the same compute
  capability. `--tp 1` remains the default and is unchanged: `scripts/tp1-regression.sh` compares
  this build's greedy output against token streams recorded from upstream `ninfer` at base commit
  `feaf4dd`, over a short chat, a 2,191-token instruction and a 28,677-token document, and requires
  the token ids, the generated text and the deterministic summary rows to be byte-equal. Its scope
  is exactly that: greedy text decode, `--tp 1`, `--rope native`, NVFP4 weights, `qwen3.8-27b`, MTP
  off, concurrency 1. See `tests/data/tp1-golden/MANIFEST.md`.
- `--kv-capacity` must be at least `--max-context`. The explicit form is what fits a tier at its
  ceiling; `auto` also keeps a 512 MiB sizing headroom.
- `--max-concurrency 1` is arithmetic on 16 GiB cards: the tier table below is what one slot costs,
  and `k16i8` at 253,952 leaves no room for a second.
- MTP speculative decoding (`--spec mtp --draft-tokens 1..5`, optionally `--lm-head-draft`) works at
  `--tp 2`, including compatible-prefix reuse. `--spec dflash` and `--vision` are rejected at
  `--tp 2`.
- `--rope yarn`, the extended-position path this line inherits from upstream, is available but unused
  here: this fork's ceiling is 253,952 tokens, below the registered native 262,144, so no rope
  scaling is needed and none is measured.

See the [CLI guide](docs/cli.md) and [HTTP serving](docs/serving.md) for the full option contract.

### KV-cache tiers and long-context limits

`--kv-dtype` selects the KV codec per side: `bf16` (no quantization), `int8` (per-64 fp16 scale),
`k16i8` (BF16 keys + INT8 values, V scaled every 64 dims). All tiers compute QK in BF16; the codec
only changes residency and read-back. The earlier FP8 KV tiers (`fp8` = e4m3 both sides, `k16v8` =
BF16 keys + e4m3 values) were removed after measurement: at the same cost per token the INT8 V codec
accepts at least as well and no longer needs the kernel to expand e4m3 codes into BF16 in shared
memory (which measured slower, not faster: 131k-token decode 21.0 tok/s for `fp8` vs 25.6 for `k16i8`
and 36.5 for `int8`).

Measured on 2× RTX 5060 Ti (TP2, one slot, `--prefill-chunk 1024`, MTP3 + `--lm-head-draft`,
greedy; acceptance on three prompts — science / horror prose / JSON):

| `--kv-dtype` | 21k prefill | 131k prefill | 131k decode | MTP acceptance (P1/P2/P3) | 1-slot ceiling |
|---|---|---|---|---|---|
| `int8` | 4376 tok/s | 1991 tok/s | 36.5 tok/s | 2.18 / 1.96 / 4.00 | 262144 |
| `k16i8` | 3946 tok/s | 1489 tok/s | 25.6 tok/s | **2.46 / 1.97 / 4.00** | 253952 (chunk 1024) / 229376 (chunk 4096) |
| `bf16` | ~3990 tok/s | ~1520 tok/s | 26.5 tok/s | 2.43 / 1.92 / 4.00 | — |

`int8` is the fastest everywhere; `k16i8` trades ~10% of that for BF16 keys (it also runs the
`int8`-side V dequant with a 64-dim scale). Both beat the removed e4m3 V tier on acceptance, which is
what the per-round cost turns into tokens: per-round cost is nearly the same across tiers
(28.5–29.4 ms), so the token-rate spread follows the acceptance length. `--kv-capacity` must be
≥ `--max-context`, and the explicit-capacity form is what fits a tier at its ceiling (`auto` keeps a
512 MiB sizing headroom, which `k16i8` cannot afford at 253952).

Long context (`k16i8`, 1 slot, 253952): decode stays flat in the prefill-free regime and degrades
with context only through the KV read (at 131k tokens the KV stream is ~5.7 GB per round against
10.36 GiB of weights per card), which is why the tiers separate at long context.


Prefix reuse hits when the retained turn checkpoint covers the prompt: a repeated request reports
`cache=7872 reuse=restore_turn_checkpoint` and its time-to-first-token drops from 1788 ms to 71 ms.
A reused prefill computes its suffix from the checkpoint frontier rather than from the full-prefill
chunk grid, so a hit and a cold run can differ in the last bits (and occasionally in greedy text) —
the same class of caveat as vLLM/SGLang prefix caching. The hit is reported back in the response
usage: OpenAI responses carry `usage.prompt_tokens_details.cached_tokens`, and Anthropic responses
carry `usage.cache_read_input_tokens` (with `cache_creation_input_tokens` reported as 0).

`/health` reports engine availability (`200 {"status":"ok"}` / `503 {"status":"unavailable"}`), and
`apps/ninfer-serve` exits non-zero when the engine becomes unusable so a supervisor
(`Restart=on-failure`) reloads the model in about 16 s instead of leaving a dead endpoint up.

### Limitations

- **Vision is `--tp 1` only.** The Vision encoder runs on the primary device against replicated
  weights and has no split path, so `--tp 2 --vision` is rejected at startup.
- **DFlash is rejected at `--tp 2`.** It remains a 35B-A3B text-only backend, and that target has no
  tensor-parallel path at all.
- **Peer access depends on the board.** Upstream measured `cudaDeviceCanAccessPeer` as 0 between two
  RTX 5090s and stages the collectives as host-staged copies over PCIe rather than direct peer
  copies; on the two RTX 5060 Ti cards this fork was measured on, it reports 1 in both directions.
  Either way a decode token costs 128 reduces plus one logit all-gather, and under CUDA Graphs the
  whole collective set is small next to the ~28 ms per round the weights themselves cost.
- **MTP prefix reuse works at `--tp 2`, with two known degradations.** This fork implements the
  retained-state resume on the TP2 path, so a request that extends a prefix the engine still holds
  hits (`reuse=restore_turn_checkpoint`) instead of re-prefilling the whole prompt. It still
  degrades to a full prefill when the retained base already covers the entire prompt, so that there
  is no suffix left to compute; and a hit is a *same-prefix continuation*, not a shared prefix --
  a request whose common prefix is not the one the engine retained does not hit. The answer is
  unchanged in every case and no request fails.
- **MTP is output-equivalent up to near-tie argmax flips, not bit-identical.** A verify round
  evaluates the target model over `K+1` columns at once and an ordinary round over one, which
  selects different GEMM shapes; greedy MTP-on and MTP-off streams can therefore diverge on a
  near-tie token. Every committed token is still one the target model's own argmax selected.
- **`--ignore-eos` is a diagnostic flag.** It exists for fixed-length soak and throughput work.
  Generation past the end-of-turn token is off-distribution and is not a product output. It is not
  the only route to a degenerate stream: greedy decoding at long context also loops on its own
  content, with the flag off.
- **A reused prefix can differ from a cold prefill in the last bits.** A cache hit computes its
  suffix from the retained checkpoint frontier instead of from the full-prefill chunk grid, so the
  two runs are not bit-identical and greedy text can occasionally flip. This is the same class of
  caveat as vLLM/SGLang prefix caching, and it is why a cache hit is not a correctness contract.

The design decisions behind the TP2 path -- the collective transport, the shard map, and what each
correctness gate actually proves -- are in
[Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md); that document also
covers the YaRN extended-position work this fork does not use.

## Capabilities and limits

Capabilities. All three registered model IDs (`qwen3.6-27b`, `qwen3.8-27b`, `qwen3.6-35b-a3b`)
support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- KV cache tiers `bf16`, `int8` (group-64) and `k16i8` (BF16 keys + INT8 values);
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse, including MTP at `--tp 2`, with the hit count reported as
  `usage.prompt_tokens_details.cached_tokens` (OpenAI) / `usage.cache_read_input_tokens`
  (Anthropic);
- a `/health` endpoint that reports engine availability and a server that exits non-zero when the
  engine becomes unusable, so a supervisor can restart it;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

Limits.

- Only the five registered `(model_id, weights_id)` artifact identities are accepted product
  identities; this fork is built and measured against the Qwen3.8-27B NVFP4 pair listed above.
- Execution targets `sm_120a` consumer Blackwell, and this fork is built and measured on exactly two
  RTX 5060 Ti (16 GiB each) with `--tp 2 --devices A,B`. One CUDA device is the engine's default, and
  the 27B execution package also runs on exactly two, which is a capacity feature rather than
  scale-out.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  CPU/GPU offload, or distributed serving. Multi-GPU execution is exactly the two-device
  tensor-parallel width described above: one process, one resident model, no NVLink, no more than
  two devices.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit (`--rope yarn` reaches further on the 27B targets, but this fork
  does not use it; see [Dual-GPU (TP2)](#dual-gpu-tp2)). `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Relationship to upstream

This fork descends from `Neroued/ninfer` at commit `feaf4dd` (2026-08-20) through the TP2 line, and
carries its own work on that base. Upstream `master` has since advanced more than 200 commits and
restructured its runtime -- its executor and KV sizing live in different files than here, and its
KV-cache subsystem was rewritten. The two trees are therefore **not** interchangeable: merging
`master` into this line conflicts in hundreds of files, so this fork tracks its own base and
cherry-picks upstream fixes where they apply instead of following `master`. GitHub reports this
branch as ahead of *and* behind `Neroued:master` -- that is the state of this line, not staleness.

| | this fork | upstream `master` |
|---|---|---|
| `--kv-dtype` | `bf16`, `int8`, **`k16i8`** (BF16 keys + INT8 values) | `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4` |
| Tensor parallelism | `--tp 2 --devices A,B`, validated on 2× RTX 5060 Ti | single GPU |
| `/health` | engine availability, plus a supervisor-driven restart when the engine dies | engine availability |

If you want `nvfp4` / `k8v4` KV tiers or upstream's newest single-GPU scheduling work, use upstream.
If you want tensor-parallel serving on two consumer cards with the `k16i8` tier and MTP prefix reuse
that actually hits, this fork is the line to use.

## Getting help

**Ask a coding agent first.** The maintainer does — questions sent to a human here usually get
forwarded to an agent that has this repository open, so pointing your own agent at this README and
the [`docs/`](docs/) tree will get you an answer sooner. Issues are enabled if you want to leave a
record of the problem.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork's modifications are under the
same licence; see [NOTICE](NOTICE) for the attribution required by Apache-2.0 §4(b).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
