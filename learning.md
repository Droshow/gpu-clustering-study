# GPU Clustering — Learning Path
**Audience:** Senior DevOps Engineer. Skip Terraform, Kubernetes basics, cloud networking 101.  
**Goal:** Understand the full stack from silicon to cluster governance, with enough depth to read architecture docs, talk to infra teams, and contribute meaningfully.

---

## Mental model before you start

```
Silicon (GPU die)
    └── Node (NVLink, NVSwitch — GPUs talking within a box)
            └── Cluster (InfiniBand or RoCE — boxes talking across a fabric)
                    └── Orchestration (K8s GPU Operator, Volcano, Ray)
                            └── Inference (vLLM, Triton, NIM)
                                    └── Governance (PerchGuard, policy layer)
```

Every layer has its own failure modes, its own metrics, and its own people fighting over it. Learn bottom-up. The Arista vs. NVIDIA fight lives at layer 3.

---

## Block 1 — GPU Hardware Fundamentals

**What you're learning:** What a GPU actually is, why it's not just a fast CPU, and what the physical hardware looks like at scale.

### Concepts

- **Warps and SIMT:** A GPU executes 32 threads simultaneously in a *warp*. Every thread in the warp runs the same instruction. This is SIMT (Single Instruction Multiple Threads). The implication: GPUs are fast at uniform, parallel work and terrible at branchy, sequential work. This is why transformers (matrix multiplications) map perfectly to GPUs.
- **SM (Streaming Multiprocessor):** The execution unit. An H100 SXM has 132 SMs. Each SM has Tensor Cores (for matrix math), CUDA Cores (general compute), and shared memory.
- **Tensor Cores:** The important ones for AI. They do mixed-precision matrix multiply-accumulate (MMA) in hardware. FP8, BF16, FP16 → FP32 accumulate. This is where the AI FLOPS number comes from.
- **HBM (High Bandwidth Memory):** The GPU's VRAM. Not DDR, not GDDR — it's 3D-stacked DRAM sitting next to the GPU die on the same substrate. H100 SXM has 80GB HBM3 with ~3.35 TB/s bandwidth. This bandwidth number matters more than raw FLOPS for large models.
- **NVLink:** NVIDIA's proprietary GPU-to-GPU interconnect. Within a DGX H100 node, 8 GPUs are fully connected via NVLink 4.0 at 900 GB/s bidirectional per GPU. Compare to PCIe 5.0 at ~128 GB/s. This is why multi-GPU training within a node is fast.
- **NVSwitch:** The chip that enables all-to-all NVLink connectivity at scale. A DGX H100 has 4 NVSwitches. Without NVSwitch you'd have point-to-point NVLink; with it every GPU can talk to every other GPU at full bandwidth simultaneously.

### Hardware families to know

| System | GPUs | NVLink | Use case |
|--------|------|--------|----------|
| DGX H100 | 8× H100 SXM5 | NVLink 4.0, 4× NVSwitch | Single-node training |
| DGX SuperPOD | 32× DGX H100 (256 GPUs) | NVLink + InfiniBand fabric | Large-scale training |
| HGX H100 | 8× H100 (OEM form) | Same as DGX, no peripherals | Cloud providers, OEMs |
| GB200 NVL72 (Blackwell) | 36× B200 + 72× GB200 in rack | NVLink 5.0, NVSwitch Gen5 | Latest generation |

### Resources
- NVIDIA H100 Architecture whitepaper (free, NVIDIA.com) — read the SM and NVLink sections
- `nvidia-smi topo -m` — run this on any GPU node; it prints the NVLink topology. Learn to read it.
- Hopper Architecture Deep Dive (NVIDIA blog, 2022) — still the best explanatory piece

---

## Block 2 — NVIDIA Software Stack

**What you're learning:** The layers between the hardware and your workload.

### Stack from bottom to top

```
CUDA Driver (kernel module: nvidia.ko)
    └── CUDA Runtime (libcudart)
            └── cuDNN (deep learning primitives — convolutions, attention)
            └── cuBLAS (matrix math)
            └── NCCL (collective communications — all-reduce, all-gather)
                    └── PyTorch / JAX / TensorFlow
                            └── vLLM / Triton Inference Server / TensorRT-LLM
```

### What you actually need to understand

**NCCL (NVIDIA Collective Communications Library)** — This is the most important piece for cluster performance. NCCL implements the collective operations that distributed training depends on:
- *All-reduce:* every GPU sends its gradient, every GPU receives the sum. This is the core of data-parallel training. If the network is slow, the GPUs sit idle waiting for all-reduce to finish.
- *All-gather, reduce-scatter:* used in tensor parallelism (splitting a single model layer across GPUs).
- NCCL uses NVLink within a node and the cluster fabric (IB or RoCE) across nodes. The fabric bandwidth directly determines how large a model you can train at what speed.

**DCGM (Data Center GPU Manager)** — NVIDIA's GPU monitoring daemon. Runs on each node. Exposes:
- SM utilization, memory utilization, memory bandwidth
- NVLink bandwidth (per-link)
- PCIe bandwidth
- Power draw, temperature, throttle state
- **XID errors** — these are GPU error codes. XID 79 = GPU has fallen off the bus. XID 63 = row remapping (memory error). Know the common ones; they're your GPU equivalent of OOM kills.
- `dcgmi` is the CLI. `dcgm-exporter` scrapes these as Prometheus metrics.

**MIG (Multi-Instance GPU)** — Introduced on A100, supported on H100. Partitions a single GPU into up to 7 isolated instances, each with its own HBM slice, SM slice, and NVLink bandwidth slice. A 3g.40gb MIG instance is 3/7 of an H100. Used for inference workloads that don't need a full GPU.

---

## Block 3 — NVIDIA on Kubernetes (the CRD layer)

**What you're learning:** How NVIDIA's K8s stack actually works, beyond "just add `nvidia.com/gpu: 1` to your pod spec."

### GPU Operator

The GPU Operator is a meta-operator that installs and manages the entire NVIDIA software stack on K8s nodes. One `ClusterPolicy` CRD drives everything.

```yaml
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: gpu-cluster-policy
spec:
  driver:
    enabled: true
    version: "550.90.07"
  toolkit:
    enabled: true          # container toolkit (nvidia-container-runtime)
  devicePlugin:
    enabled: true          # exposes nvidia.com/gpu resource to scheduler
  dcgmExporter:
    enabled: true          # Prometheus metrics
  mig:
    strategy: single       # or "mixed" — how MIG instances are presented
  migManager:
    enabled: true
  nodeStatusExporter:
    enabled: true
  gfd:
    enabled: true          # GPU Feature Discovery — labels nodes with GPU capabilities
```

**ClusterPolicy** is the one CRD you own as an infra engineer. Everything else is managed by the operator.

### GPU Feature Discovery labels

GFD labels every node with its GPU capabilities. These become node selectors:
```
nvidia.com/gpu.product=NVIDIA-H100-SXM5-80GB
nvidia.com/gpu.memory=81920
nvidia.com/mig.capable=true
nvidia.com/gpu.count=8
feature.node.kubernetes.io/pci-10de.present=true   # NVIDIA PCI device present
```

Use these in `nodeSelector` / `nodeAffinity` to target specific GPU types without hardcoding node names.

### Network Operator (for InfiniBand / RoCE)

Separate from the GPU Operator. Manages the Mellanox/ConnectX NIC drivers and the RDMA stack. Key CRD: `NicClusterPolicy`.

```yaml
apiVersion: mellanox.com/v1alpha1
kind: NicClusterPolicy
spec:
  ofedDriver:
    image: mofed            # Mellanox OFED driver
    version: "23.10-0.5.5.0"
  rdmaSharedDevicePlugin:
    resources:
      - name: rdma_shared_device_a
        vendors: ["15b3"]    # Mellanox vendor ID
  sriovDevicePlugin:
    resources:
      - name: sriov_a
        vendors: ["15b3"]
        devices: ["1017"]    # ConnectX-6 Dx
```

RDMA (Remote Direct Memory Access) is what enables GPUDirect RDMA — GPU memory directly accessible over the network, bypassing the CPU and host memory. Critical for multi-node training performance.

### Time-slicing vs MIG

| | Time-slicing | MIG |
|---|---|---|
| Isolation | None (shared SM, memory) | Hard (SM + HBM partitioned) |
| Overhead | Zero | Small |
| GPU generations | Any | A100+ only |
| Use case | Dev, low-priority inference | Production multi-tenant inference |
| K8s resource name | `nvidia.com/gpu` | `nvidia.com/mig-3g.40gb` (etc.) |

### Volcano (batch scheduler)

The default K8s scheduler doesn't understand GPU jobs well — it doesn't know that a 256-GPU training job needs all 256 GPUs to start simultaneously (gang scheduling). Volcano solves this:

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
spec:
  minAvailable: 256        # all-or-nothing scheduling
  tasks:
    - replicas: 256
      name: worker
      template:
        spec:
          containers:
            - resources:
                limits:
                  nvidia.com/gpu: "1"
```

Without gang scheduling, partial allocations happen, GPUs sit reserved but idle, and the job never starts. This is a real production failure mode.

---

## Block 4 — The Network Layer (Where the Arista Fight Lives)

**What you're learning:** Why the network is the bottleneck in GPU clusters, and what the two camps are offering.

### Why the network matters more than you'd expect

All-reduce across 256 GPUs requires every GPU to send its gradient to every other GPU. At BF16, a 70B parameter model has ~140GB of gradients. Every iteration, every GPU sends and receives ~140GB across the fabric. At 400Gbps per link, that's ~2.8 seconds per all-reduce per iteration. Training a large model takes millions of iterations. The network is not a side concern.

### InfiniBand (NVIDIA's play)

NVIDIA acquired Mellanox in 2020 for $6.9B specifically for this. InfiniBand is:
- A lossless fabric by design (credit-based flow control, no dropped packets)
- Very low latency (~1μs)
- Proprietary — NVIDIA controls the switches (Quantum series), NICs (ConnectX), cables, and drivers
- Expensive — IB switches are 2–4× the cost of equivalent Ethernet
- Dominant in HPC and current GPU clusters: ~70% of the top 500 supercomputers run IB

NVIDIA's Quantum-X800 InfiniBand switch: 64 ports × 800 Gbps = 51.2 Tbps per switch.

### Ethernet (Arista's bet)

Arista's thesis: Ethernet will win because:
- Enterprises already run Ethernet everywhere — same ops skills, same tooling
- Cost: Ethernet switches are cheaper, more competitive supply chain
- Speed parity: 400GbE is here, 800GbE is shipping — closing the gap with IB
- The **Ultra Ethernet Consortium (UEC)** — Arista, AMD, Broadcom, Cisco, Intel, Meta, Microsoft — is standardizing extensions to Ethernet for AI workloads (congestion control, load balancing, RDMA)

The hard problem with Ethernet for AI: it was designed to drop packets and retransmit. RDMA (RoCEv2) requires a *lossless* network. You get this with:
- **PFC (Priority Flow Control)** — pause frames per traffic class to prevent drops
- **ECN (Explicit Congestion Notification)** — mark packets before the queue fills rather than dropping them
- **DCQCN** — the congestion control algorithm that combines PFC + ECN for RoCE

This is the "lossless Ethernet" problem that Arista's arista_manifest.md already covers. The key insight: Ethernet *can* be lossless, it just requires careful configuration that most network teams aren't used to doing.

### Arista's specific play

Arista 7800R3 and 7700R series are their AI spine switches. Key features:
- Deep buffers (important: IB uses credit-based flow control and rarely needs large buffers; Ethernet uses buffers to absorb bursts)
- ECMP with adaptive routing (GPU traffic is elephant flows — large, long-lived; you want per-packet load balancing, not per-flow)
- CloudVision integration for telemetry and configuration at scale
- Supported in the UEC testing consortium

### The current reality (2026)

Hyperscalers (Meta, Microsoft, Google) are building large Ethernet-based GPU clusters because they want to own the full stack. NVIDIA sells to enterprises who want a working solution fast and are willing to pay for IB. Arista is winning the "next wave" of enterprise AI infrastructure buildouts where enterprises already have Arista in the data center and don't want to run two separate fabrics.

---

## Block 5 — Inference Infrastructure

**What you're learning:** Where autonomous agents actually run, and what the serving layer looks like.

### vLLM

The dominant open-source LLM inference engine. Key concepts:
- **PagedAttention:** Manages KV cache (the memory that grows as a conversation gets longer) in pages rather than contiguous blocks. Dramatically improves GPU memory utilization and throughput.
- **Continuous batching:** Don't wait for all requests in a batch to finish — as one request completes, add a new one. Massively improves GPU utilization compared to static batching.
- **Tensor parallelism:** Split a model layer across multiple GPUs. A 70B model that doesn't fit on one GPU can be split across 4. Requires NVLink bandwidth.
- **Pipeline parallelism:** Split different layers across GPUs. Less bandwidth-sensitive than tensor parallelism, more latency-sensitive.

### Triton Inference Server (NVIDIA)

NVIDIA's inference server. Supports multiple backends (TensorRT, vLLM, ONNX, custom). Exposes HTTP, gRPC, and Prometheus metrics endpoints. Used when you need multi-model serving, model versioning, or tight NVIDIA toolchain integration.

### TensorRT-LLM

NVIDIA's optimized LLM inference library. Takes a model and compiles it to TensorRT for maximum throughput on specific GPU hardware. Less flexible than vLLM but faster on NVIDIA hardware for known model architectures.

### NVIDIA NIM (Inference Microservices)

Containerized, production-ready inference for specific models. Think: NVIDIA's opinionated way to serve Llama 3, Mistral, etc. Runs on NeMo infrastructure, exposes OpenAI-compatible API. The fastest path from GPU to serving endpoint for enterprises with NVIDIA contracts.

---

## Block 6 — Observability

**What you're learning:** What to watch, and what breaking looks like.

### Key metrics (via DCGM Exporter → Prometheus)

| Metric | What it tells you |
|--------|-------------------|
| `DCGM_FI_DEV_GPU_UTIL` | SM utilization — should be >80% during training |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth utilization |
| `DCGM_FI_DEV_FB_USED` | VRAM in use — watch for OOM |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | NVLink throughput — low = collective comm bottleneck |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw — H100 TDP is 700W; throttling starts at thermal limits |
| `DCGM_FI_DEV_XID_ERRORS` | GPU error count — any nonzero is a problem |
| `DCGM_FI_DEV_PCIE_TX/RX_THROUGHPUT` | PCIe bandwidth — relevant when not using GPUDirect |

### XID errors worth knowing

| XID | Meaning |
|-----|---------|
| 8 | GPU diagnostic failure |
| 31 | GPU memory page fault |
| 63 | Row remapping (HBM memory error — ECC correction) |
| 74 | NVLink error |
| 79 | GPU fallen off the bus (fatal — node needs reboot) |
| 92 | High single-bit ECC error rate |

### nvidia-smi commands that matter

```bash
nvidia-smi topo -m                     # NVLink topology matrix
nvidia-smi dmon -s u                   # live utilization per GPU
nvidia-smi nvlink --status -i 0        # NVLink link state for GPU 0
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw \
           --format=csv -l 1           # one-liner live dashboard
dcgmi discovery -l                     # DCGM: list detected GPUs
dcgmi health -g 1 -c                   # DCGM: run health check
```

---

## Block 7 — Papers worth reading (not courses — papers)

These are short, well-written, and will advance your conceptual depth faster than any tutorial.

| Paper | Why |
|-------|-----|
| **Megatron-LM** (Shoeybi et al., 2019) | Defines tensor and pipeline parallelism — the vocabulary everyone uses |
| **Efficient Large Scale Language Modeling with Mixtures of Experts** (Artetxe et al., 2021) | MoE architecture — relevant for understanding H100 Tensor Core utilization patterns |
| **AlpaServe** (Li et al., 2023) | Statistical multiplexing of GPU clusters — how to think about inference resource allocation |
| **vLLM: Efficient Memory Management for LLM Serving** (Kwon et al., 2023) | PagedAttention — required reading for anyone running inference |
| **RDMA over Commodity Ethernet (RoCE)** (Zhu et al., Microsoft, 2015) | The original DCQCN paper — defines the congestion control problem Arista is solving |

---

## Learning sequence recommendation

1. **Week 1:** Block 1 (GPU hardware) + read `nvidia-smi topo -m` output on any cloud GPU instance (a spot A10 costs $0.30/hr, spin it for an hour)
2. **Week 2:** Block 3 (K8s CRDs) — deploy GPU Operator on a local kind cluster with a fake GPU device plugin if no real GPU available
3. **Week 3:** Block 4 (networking) — this is where your Arista context lands; read the RoCE paper and then the UEC whitepaper
4. **Week 4:** Block 5 (inference) — run vLLM locally (CPU mode works for learning the API surface), read the PagedAttention paper
5. **Ongoing:** Block 6 (observability) — next time you have GPU access, run the `nvidia-smi` commands until they're fluent

---

## What you can skip (for now)

- CUDA programming (you're governing clusters, not writing kernels)
- cuDNN / cuBLAS internals (framework concern, not infra concern)
- Model architecture details (transformer internals, attention math)
- Slurm (HPC legacy scheduler — relevant if a customer runs HPC, but Kubernetes is winning)
