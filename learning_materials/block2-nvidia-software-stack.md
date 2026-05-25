# Block 2 — NVIDIA Software Stack

**Audience:** Senior DevOps Engineer — infrastructure depth, not kernel development.  
**Goal:** Understand the layers between hardware and workload. Focus on NCCL (cluster performance) and DCGM (observability). These are the two pieces you will touch in production.

---

## The stack

```
CUDA Driver  (kernel module: nvidia.ko)
    └── CUDA Runtime  (libcudart — loaded by the application)
            └── cuDNN   (deep learning primitives: convolutions, attention ops)
            └── cuBLAS  (dense matrix math: GEMM)
            └── NCCL    (collective communications: all-reduce, all-gather)
                    └── PyTorch / JAX / TensorFlow
                            └── vLLM / Triton Inference Server / TensorRT-LLM
```

As an infra engineer, you own everything from CUDA Driver downward, and you care about NCCL because it is the bridge between "hardware is fast" and "training is fast."

---

## CUDA Driver vs CUDA Runtime — why the distinction matters

**CUDA Driver** (`nvidia.ko`) is the kernel module. It is installed on the host OS. It manages the GPU as a PCIe device — memory allocation, context creation, command submission. It has a version number (`nvidia-smi` shows it).

**CUDA Runtime** (`libcudart.so`) is a userspace library that ships *inside* the container. PyTorch, JAX, and TensorFlow all bundle it. The runtime communicates with the driver via the driver API.

**The compatibility rule:** the CUDA Runtime version in the container must be ≤ the maximum CUDA version supported by the driver on the host. The driver is backward compatible. A driver that supports CUDA 12.4 will run containers with CUDA 12.0, 12.1, 12.2, 12.3, or 12.4 runtimes.

This is why you can upgrade the container (runtime) without upgrading the host driver — up to a ceiling. It is also why a container built with CUDA 12.5 will fail on a host running a CUDA 12.2 driver. This is the most common GPU container compatibility error in production.

```bash
# Host driver version and max CUDA version it supports
nvidia-smi

# CUDA runtime version inside a container
python -c "import torch; print(torch.version.cuda)"
```

---

## NCCL — the piece that determines cluster training speed

NCCL (NVIDIA Collective Communications Library) implements the collective operations that distributed training is built on. When people say "the network is the bottleneck," they mean NCCL is waiting for the network.

### The collectives

**All-reduce** — the core of data-parallel training.

Each GPU holds a copy of the model. Each GPU processes a different batch of data and computes gradients. All-reduce takes the gradients from all GPUs, sums them, and distributes the result back to all GPUs. Every GPU ends the step with identical weights. This happens every single training iteration.

> **DevOps perspective — what "processes a batch and computes gradients" means:**
>
> You have a massive dataset — say, 1 million images. Instead of one GPU processing all of them sequentially, the dataset is split:
> GPU0 processes images 1–1000, GPU1 processes 1001–2000, and so on. Each GPU runs a forward pass on its shard and computes **gradients** — a tensor of floats, one per model parameter, that says "adjust this weight by this much."
>
> The gradients from each GPU are *different* because each GPU saw different data. The model can only update usefully if all GPUs agree on a combined gradient — that's what all-reduce delivers.
>
> The sequence as an infra concern:
> ```
> 1. Dataset split → each GPU gets a shard
> 2. Each GPU: forward pass → backward pass → local gradients   (SM util: high)
> 3. NCCL all-reduce: sum all gradients, distribute to every GPU (SM util: near zero — waiting on network)
> 4. Each GPU updates its weights with the combined gradient
> 5. Repeat every iteration
> ```
>
> Step 3 is where your network utilisation spikes and `DCGM_FI_DEV_GPU_UTIL` drops. The GPUs are done computing and sitting idle waiting for ~140GB to move across the fabric. That idle period is what you are measuring and trying to minimise.

> **DevOps perspective — tensor vs gradient:**
>
> A **tensor** is just the data structure — an n-dimensional array of floats. It is the generic container for everything in deep learning: inputs, outputs, weights, and gradients are all tensors. Think of it like "file" — a config file and a log file are both files.
>
> A **gradient** is a tensor with a specific meaning: one float per model parameter, encoding "adjust this weight by this much." It is produced by the backward pass (backpropagation) and answers: if I nudge this weight slightly, how much does the model's error change?
>
> Example for a model with 3 weights:
> ```
> Weights tensor:   [0.5,  -1.2,  0.8]   ← current model parameters
> Gradient tensor:  [0.03, -0.1,  0.07]  ← computed adjustments from this batch
>
> Weight update:  new_weight = old_weight - (learning_rate × gradient)
> ```
>
> For the infra concern: the "140GB of gradients" flowing through NCCL is simply a tensor of 70 billion floats — one per model parameter. NCCL does not know or care that it is a gradient; it just moves the bytes. What makes it a gradient is what computed it and what the training loop does with it after delivery.

> **DevOps perspective — who actually triggers and executes the all-reduce:**
>
> PyTorch is the orchestrator. NCCL is the executor. They have distinct roles:
>
> ```
> PyTorch training loop (running on CPU, coordinating GPU work)
>   └── loss.backward()        ← tells the GPU to run the backward pass, producing gradients
>   └── optimizer.step()       ← before updating weights, PyTorch calls NCCL:
>         └── ncclAllReduce()  ← PyTorch hands the gradient tensor to NCCL
>               └── NCCL       ← takes over: selects transport (NVLink / IB / RoCE),
>                                 builds ring topology, executes the transfer across all GPUs
>               └── returns    ← NCCL gives back the summed gradient tensor
>   └── weight update          ← PyTorch applies the combined gradient to the weights
> ```
>
> NCCL is not passive — it actively runs a **ring algorithm** where every GPU is simultaneously
> sender and receiver. Each GPU passes its chunk to the next GPU in the ring, receives a chunk
> from the previous one, accumulates, and passes it on. After 2(N-1) steps across N GPUs, every
> GPU holds the complete sum. But NCCL only acts when PyTorch calls it. It has no scheduler of its own.
>
> Why SM utilisation drops to near zero at step 3: the GPU SMs finished the backward pass and are
> idle. Control is with the NIC and NVLink fabric now. The SMs have nothing to execute until NCCL
> returns and PyTorch can proceed to the weight update.
>
> The layer responsibilities:
>
> | Layer | Role | Analogy |
> |-------|------|---------|
> | PyTorch / JAX | Decides *when* to all-reduce and *what* tensor | Application calling `send()` |
> | NCCL | Decides *how* to move it, executes the ring | TCP stack handling packet delivery |
> | NVLink / InfiniBand / RoCE | Physical wire carrying the bytes | The network cable |

```
Before:  GPU0[g0]  GPU1[g1]  GPU2[g2]  GPU3[g3]
After:   GPU0[g0+g1+g2+g3]  GPU1[same]  GPU2[same]  GPU3[same]
```

At BF16 with a 70B parameter model: ~140GB of gradients. Every iteration. Across potentially hundreds of nodes. If the network can't sustain this, the GPUs stall waiting for all-reduce to complete. SM utilisation drops to zero during the wait. This is the GPU idle problem.

**All-gather** — used in tensor parallelism (ZeRO-3, FSDP).

Each GPU holds a *shard* of the model weights. Before a forward pass, each GPU needs the full weights for its layer. All-gather collects the shards from all GPUs and gives every GPU the complete tensor.

**Reduce-scatter** — the inverse. Used after the backward pass to distribute gradient shards back.

**Broadcast** — send one value from one GPU to all others. Used for parameter synchronisation at startup.

### How NCCL chooses its transport

NCCL detects the available transports and builds a ring or tree topology automatically:

| Transport | When used |
|-----------|-----------|
| NVLink (via CUDA IPC) | GPUs on the same node — fastest |
| PCIe + shared memory | Same node, no NVLink (rare on H100) |
| InfiniBand (via RDMA verbs) | Across nodes with IB fabric |
| RoCEv2 (via RDMA verbs) | Across nodes with RoCE fabric |
| TCP/IP | Fallback — never want this in production |

**GPUDirect RDMA** is what enables NCCL to move data directly from GPU HBM to the network NIC without staging through host CPU memory. The path is: GPU HBM → NVLink → NIC → wire. Bypassing the CPU reduces latency and frees CPU cycles. Requires: RDMA-capable NICs (Mellanox ConnectX), correct driver stack (MOFED), and the `NicClusterPolicy` CRD on Kubernetes (see Block 3).

### Key NCCL environment variables you will see

```bash
NCCL_DEBUG=INFO          # enables verbose logging — first thing to set when debugging slow collective comms
NCCL_IB_DISABLE=1        # force NCCL to not use InfiniBand (useful for testing TCP fallback)
NCCL_SOCKET_IFNAME=eth0  # tell NCCL which NIC to use for TCP fallback
NCCL_P2P_DISABLE=1       # disable peer-to-peer (NVLink) transfers — forces PCIe path
NCCL_ALGO=Ring           # force ring algorithm (default: NCCL chooses best)
```

In practice: set `NCCL_DEBUG=INFO` first whenever a distributed job is slow or hanging. NCCL will print which transport it selected, which ring/tree topology it built, and where it's stuck.

### The all-reduce math

At 400Gbps per link, all-reduce over 256 GPUs on a 70B BF16 model:

- Gradient size: 70B params × 2 bytes (BF16) = 140 GB
- Ring all-reduce sends 2× data per GPU: 2 × 140 GB = 280 GB per GPU
- At 400Gbps = 50 GB/s: 280 GB / 50 GB/s = **5.6 seconds per iteration**

If your training step (forward + backward pass) takes 2 seconds, you spend 5.6 seconds waiting for all-reduce. Your GPU utilisation is 2 / (2 + 5.6) = **26%**. The rest is network idle. This is why people spend enormous effort on network bandwidth, gradient compression (FP8 gradients), and reducing all-reduce frequency.

---

## DCGM — your GPU observability layer

DCGM (Data Center GPU Manager) is an NVIDIA daemon (`nv-hostengine`) that runs on each node. It polls the GPU via the NVML API and exposes metrics. `dcgm-exporter` scrapes those metrics as Prometheus-compatible format.

### What DCGM exposes (the ones that matter)

| Metric | What it tells you |
|--------|-------------------|
| `DCGM_FI_DEV_GPU_UTIL` | SM utilisation % — should be >80% during training |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | HBM bandwidth utilisation |
| `DCGM_FI_DEV_FB_USED` | VRAM in use (bytes) — watch for OOM |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | Total NVLink throughput — low = collective comm bottleneck |
| `DCGM_FI_DEV_PCIE_TX/RX_THROUGHPUT` | PCIe bandwidth — high = GPUDirect is NOT working (CPU-mediated transfers) |
| `DCGM_FI_DEV_POWER_USAGE` | Watts — H100 TDP is 700W; hitting thermal throttle will drop clocks |
| `DCGM_FI_DEV_SM_CLOCK` | Current SM clock speed — throttling = this drops below base clock |
| `DCGM_FI_DEV_XID_ERRORS` | GPU error count — any nonzero is a problem |

**Reading GPU utilisation correctly:**  
`DCGM_FI_DEV_GPU_UTIL` is SM utilisation averaged over 1 second. If GPUs are doing collective comms over the network, SMs are *idle* — utilisation drops to near zero during the all-reduce phase. A utilisation that oscillates between 90% and 5% tells you: the GPU is computing, then waiting for NCCL, then computing. This is normal but the ratio matters.

### XID errors

XID codes are NVIDIA's hardware error taxonomy. Every XID event is logged to `dmesg` and exposed by DCGM. These are your GPU equivalent of kernel panics.

| XID | Meaning | Action |
|-----|---------|--------|
| 8 | GPU diagnostic failure | Check GPU health, may need replacement |
| 31 | GPU memory page fault | Usually a software bug (bad pointer), check workload |
| 63 | Row remapping (HBM ECC correction) | Single-bit error corrected — monitor frequency. If increasing: schedule replacement |
| 74 | NVLink error | Check NVLink cables, NVSwitch health |
| 79 | GPU fallen off the bus | Fatal. Node needs reboot. GPU may be failing |
| 92 | High single-bit ECC error rate | HBM degrading — plan replacement |

XID 79 is the one you will see at 2am: the GPU has stopped responding on PCIe. The CUDA driver can no longer communicate with it. The workload crashes. The node needs a hard reboot. If it recurs, the GPU is failing.

### DCGM CLI commands

```bash
# List all GPUs detected by DCGM
dcgmi discovery -l

# Run a health check on GPU group 1
dcgmi health -g 1 -c

# Watch live field values (field 203 = SM util, 1005 = XID errors)
dcgmi dmon -e 203,1005

# Show the full field list DCGM can expose
dcgmi field --list
```

---

## MIG — Multi-Instance GPU

MIG partitions a single GPU into up to 7 isolated instances, each with its own HBM slice, SM slice, and NVLink bandwidth slice. Introduced on A100, supported on H100.

The partition names encode the slice sizes:
- `1g.10gb` — 1/7 of SMs, 10GB HBM
- `2g.20gb` — 2/7 of SMs, 20GB HBM  
- `3g.40gb` — 3/7 of SMs, 40GB HBM
- `7g.80gb` — full GPU

Isolation is hardware-enforced. A fault in one MIG instance cannot affect another. No shared SM scheduling, no shared HBM.

**When to use MIG vs time-slicing:**

| | Time-slicing | MIG |
|---|---|---|
| Isolation | None | Hard (hardware-enforced) |
| Overhead | Zero | Minimal |
| GPU generations | Any | A100+ only |
| Use case | Dev/test, batch inference | Production multi-tenant inference |
| K8s resource | `nvidia.com/gpu` | `nvidia.com/mig-3g.40gb` |

In a multi-tenant inference cluster: MIG means tenant A's runaway workload cannot impact tenant B's latency. Time-slicing has zero overhead but zero isolation — one noisy tenant kills everyone.

---

## Summary

| Component | What it does | When you care |
|-----------|--------------|---------------|
| CUDA Driver | Kernel module, manages GPU as PCIe device | Compatibility ceiling for container runtime versions |
| CUDA Runtime | Userspace lib, ships in container | Must be ≤ driver's max supported CUDA version |
| NCCL | Collective communications (all-reduce etc.) | The network bottleneck lives here |
| GPUDirect RDMA | GPU→NIC without CPU | Requires MOFED + ConnectX; if PCIe bandwidth is high in DCGM, it's not working |
| DCGM | GPU metrics daemon | Your Prometheus source for everything GPU |
| XID errors | Hardware error codes | XID 79 = GPU dead; XID 63 = HBM error |
| MIG | Hardware GPU partitioning | Production multi-tenant inference isolation |

---

**Next:** [Block 3 — K8s CRD Layer](block3-k8s-crd-layer.md) — the GPU Operator, Network Operator, and Volcano: how NVIDIA's software stack lands on Kubernetes.
