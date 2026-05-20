# Block 5 — Inference Infrastructure

**Audience:** Senior DevOps Engineer — running and operating inference workloads, not writing model code.  
**Goal:** Understand what the serving layer looks like, why vLLM is dominant, how NVIDIA's stack (Triton, TensorRT-LLM, NIM) fits, and what the operational considerations are.

---

## The inference problem is different from training

Training: you have one job, fixed parallelism, predictable traffic, all GPUs working on one task. Optimise for throughput.

Inference: you have unpredictable request arrival, variable sequence lengths, latency SLOs (e.g., time-to-first-token < 500ms), and you want to maximise GPU utilisation across heterogeneous concurrent requests. The scheduling and memory management problems are fundamentally different.

The core constraint in LLM inference: **the KV cache**.

---

## The KV cache — the resource everything fights over

When an LLM processes a prompt and generates tokens, it computes key and value tensors for every attention head at every layer for every token. These are cached so that subsequent tokens don't need to recompute them. The cache grows linearly with sequence length.

For a 70B parameter model with 80 attention layers and 8K context length:

$$\text{KV cache} = 2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{bytes}$$

At BF16, this can be 20–40GB for a single long conversation. If you have 100 concurrent requests, you need 2–4TB of KV cache — far more than GPU VRAM. This is the memory management problem that vLLM solved.

---

## vLLM

The dominant open-source LLM inference engine. Built at UC Berkeley, open-sourced 2023, now maintained by a separate company. The first system to solve the KV cache memory problem at scale.

### PagedAttention

vLLM manages the KV cache using **PagedAttention**: instead of allocating contiguous memory for each request's KV cache, it uses a paging scheme analogous to OS virtual memory.

- KV cache is divided into fixed-size **blocks** (pages), e.g., 16 tokens per block
- A **block table** maps each sequence's logical blocks to physical GPU memory blocks
- Blocks are allocated on demand as the sequence grows
- Blocks can be shared across requests (for prefix caching — if 1000 requests share the same system prompt, those KV blocks are computed once and shared)
- Blocks are freed immediately when a request completes

Before PagedAttention: systems pre-allocated contiguous GPU memory per request based on max sequence length. A request with max 4096 tokens would reserve 4096 tokens of KV cache even if it only generated 50. Utilisation was 1–30%.

After PagedAttention: GPU memory utilisation for KV cache routinely reaches 80–95%. This translates directly to higher throughput (more concurrent requests per GPU) and lower cost.

### Continuous batching

Traditional static batching: collect N requests, run them all, wait for all to finish, collect next batch. Fast requests wait for slow requests.

**Continuous batching** (also called "iteration-level scheduling"): after every single decode step (one token generated), add newly arrived requests to the batch and remove completed requests. The batch composition changes every iteration.

This eliminates the "slow request blocks fast requests" problem. A request generating 2000 tokens doesn't hold up a request generating 10 tokens.

Combined with PagedAttention: vLLM achieves 10–20× higher throughput than naive static batching on the same hardware.

### Parallelism modes in vLLM

**Tensor parallelism (`--tensor-parallel-size N`):** Split each model layer across N GPUs. The weight matrices are sharded; each GPU computes its shard and they all-reduce after each layer. Requires NVLink bandwidth (high inter-GPU communication). Use when the model doesn't fit on a single GPU.

```bash
# Run Llama-3-70B across 4 H100s with tensor parallelism
vllm serve meta-llama/Meta-Llama-3-70B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```

**Pipeline parallelism (`--pipeline-parallel-size N`):** Assign different layers to different GPUs. GPU 0 handles layers 0-20, GPU 1 handles layers 21-40, etc. Less bandwidth-intensive than tensor parallelism (only activations cross GPU boundaries, not layer-wide all-reduces). More latency-sensitive (each layer must complete before the next starts).

**Data parallelism:** Run multiple independent vLLM instances on separate GPUs, front-ended by a load balancer. No inter-GPU communication. Maximum throughput, maximum GPU isolation. Preferred when you have enough GPUs to keep each instance busy.

### vLLM in production — key parameters

```bash
vllm serve <model> \
  --tensor-parallel-size 4 \         # split model across 4 GPUs
  --gpu-memory-utilization 0.90 \    # leave 10% for CUDA kernels etc.
  --max-num-seqs 256 \               # max concurrent requests
  --max-model-len 8192 \             # max context length
  --enable-prefix-caching \          # share KV cache for common prefixes
  --dtype bfloat16 \                 # model precision
  --port 8000
```

`--gpu-memory-utilization`: vLLM pre-allocates this fraction of VRAM for the KV cache pool on startup. The remainder holds model weights + CUDA overhead. Set too high: OOM on startup. Set too low: small KV cache pool limits concurrency.

### vLLM metrics (Prometheus)

```
vllm:num_requests_running          # currently being processed
vllm:num_requests_waiting          # queued, waiting for KV cache space
vllm:gpu_cache_usage_perc          # KV cache pool utilisation
vllm:time_to_first_token_seconds   # latency metric — your SLO target
vllm:e2e_request_latency_seconds   # full request latency
vllm:request_throughput            # tokens/second
```

`num_requests_waiting > 0` consistently means: KV cache is full, you need more GPU memory (either more GPUs or higher `gpu-memory-utilization`).

---

## Triton Inference Server (NVIDIA)

NVIDIA's inference server. Not to be confused with Triton (the GPU programming language — completely different project).

Triton Inference Server is an **ensemble and multi-model serving framework**. It wraps multiple backends and exposes a unified HTTP/gRPC API.

### Supported backends

| Backend | Use case |
|---------|---------|
| TensorRT | Compiled NVIDIA models — maximum throughput |
| vLLM | LLM serving with PagedAttention |
| ONNX Runtime | Cross-platform model serving |
| PyTorch (torchscript) | PyTorch models |
| Python | Custom preprocessing/postprocessing |
| OpenVINO | Intel hardware |

### When to use Triton over plain vLLM

- **Multi-model serving:** you're serving 10 different models from the same GPU cluster, some LLMs, some vision models, some embedding models. Triton manages them all with a unified API and shared GPU resource allocation.
- **Ensemble pipelines:** request → preprocessing (Python backend) → LLM inference (vLLM backend) → postprocessing (Python backend) → response. Triton handles the pipeline routing.
- **Model versioning:** A/B testing model versions, canary deployments of model updates.
- **NVIDIA toolchain integration:** if you're using NeMo for training and NIM for deployment, Triton is the serving runtime.

For simple "just serve this one LLM" use cases: vLLM standalone is simpler and performs identically.

---

## TensorRT-LLM

TensorRT-LLM is NVIDIA's optimised LLM compilation and inference library. It takes a model checkpoint (from Hugging Face, NVIDIA NGC, etc.) and compiles it to TensorRT — a highly optimised, GPU-hardware-specific inference engine.

What TensorRT compilation does:
- Kernel fusion: combine multiple operations into a single GPU kernel (fewer kernel launches = less overhead)
- Quantisation: FP8, INT8, INT4 with calibration
- Custom CUDA kernels for attention, layer norm, etc.
- Hardware-specific instruction scheduling for the target GPU (H100 vs A100 have different Tensor Core ISAs)

**Trade-off:** TensorRT-LLM is faster than vLLM for NVIDIA hardware on known model architectures. But:
- Compilation takes minutes to hours depending on model size
- The compiled engine is GPU-specific (can't move an H100 engine to an A100)
- Model architectures not yet supported by TensorRT-LLM fall back to ONNX or aren't supported at all
- Less flexible for cutting-edge models that change rapidly

**When to use TensorRT-LLM:** production inference on a fixed model, fixed GPU hardware, maximum throughput at any cost. Stable models (Llama, Mistral, Falcon, etc.) on H100s in a production deployment where you've profiled and tuned.

---

## NVIDIA NIM (Inference Microservices)

NIM is NVIDIA's productised, opinionated inference packaging. Think: "here is a container that runs Llama 3 70B on H100s optimally, exposes an OpenAI-compatible API, and requires zero tuning."

What's inside a NIM container:
- TensorRT-LLM engine (pre-compiled for target GPU)
- Triton Inference Server as the runtime
- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Health endpoints, Prometheus metrics
- Pre-tuned configuration for the target GPU SKU

```bash
# Pull and run Llama 3 70B NIM on H100
docker run --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama3-70b-instruct:latest
```

After startup (3–10 minutes for engine download/load), you have an OpenAI-compatible endpoint.

**When NIM makes sense:**
- Enterprise with NVIDIA contract — NIM is included
- You want the fastest path from GPU to serving endpoint with no tuning
- You're serving stable, supported model architectures (Llama, Mistral, Nemotron, etc.)
- You want NVIDIA support for the serving stack

**NIM limitations:**
- Only NVIDIA-blessed model architectures
- Less flexibility than vLLM for custom attention patterns or experimental models
- Requires NGC API key (NVIDIA cloud registry)
- New model support lags the open-source ecosystem by weeks to months

---

## Disaggregated Inference — the emerging architecture

Traditional LLM inference has two phases on the same GPU:
1. **Prefill:** process the full prompt, compute all KV cache entries — compute-bound
2. **Decode:** generate one token per step, autoregressive — memory-bandwidth-bound

These have completely different hardware requirements. Prefill wants maximum Tensor Core FLOPS. Decode wants maximum HBM bandwidth.

**Disaggregated prefill/decode** (pioneered by Splitwise, now in vLLM and commercial systems):
- Run prefill on "prefill nodes" (optimised for compute — or even use smaller, older GPUs)
- Run decode on "decode nodes" (optimised for bandwidth — H100 SXM is actually excellent for decode)
- Transfer KV cache from prefill node to decode node over the fabric after prefill completes

This is why RoCE/IB bandwidth matters for inference, not just training. A 128K context prefill generates tens of GB of KV cache that must be transferred to the decode node in milliseconds.

---

## Summary

| Technology | Role | When to use |
|------------|------|-------------|
| vLLM | Open-source LLM inference engine | Default choice; best flexibility; PagedAttention for KV memory |
| PagedAttention | KV cache memory management | Built into vLLM; eliminates memory fragmentation |
| Continuous batching | Iteration-level request scheduling | Built into vLLM; eliminates static batch inefficiency |
| Triton Inference Server | Multi-model serving framework | Multi-model, ensemble pipelines, NVIDIA toolchain |
| TensorRT-LLM | Compiled inference library | Maximum throughput on known models, fixed hardware |
| NIM | Productised inference container | Fastest path to endpoint; enterprise NVIDIA contracts |
| Tensor parallelism | Split model across GPUs | Model too large for one GPU; needs NVLink bandwidth |
| Pipeline parallelism | Split layers across GPUs | Cross-node model sharding; latency-sensitive |
| Disaggregated inference | Separate prefill and decode nodes | High-throughput, latency-critical production inference |

---

**Next:** [Block 6 — Observability](block6-observability.md) — DCGM metrics in depth, alert thresholds, XID error playbooks, and nvidia-smi commands to run fluently.
