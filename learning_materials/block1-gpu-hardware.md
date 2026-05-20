# Block 1 — GPU Hardware Fundamentals

**Audience:** Senior DevOps Engineer — no CUDA programming, no kernel dev. Infrastructure depth only.  
**Goal:** Build a mental model of what a GPU is, why it behaves differently from a CPU, and what the physical hardware looks like at scale.

---

## The core mental shift

**A CPU is a sprinter. A GPU is a marching band.**

A CPU core is optimised for *latency* — one task, done as fast as possible. Deep pipeline, branch prediction, out-of-order execution, large L3 cache. Everything exists to minimise the time a single thread spends waiting.

A GPU is optimised for *throughput* — thousands of smaller cores doing the same thing simultaneously. No branch prediction, minimal cache per core, minimal out-of-order cleverness. The GPU trades latency for bandwidth. A single GPU operation is slower than a CPU operation; a million GPU operations happening at once is what makes the machine useful.

This is why transformers (matrix multiplications — perfectly uniform, no branching) map so naturally to GPUs. The architecture fits the workload like a hand in a glove.

---

## Key concepts

### What "branching" means in code (not git)

Before warps make sense, this term needs to land correctly.

**A branch in code is any point where execution can go one of two directions at runtime** — an `if/else`, a `switch`, a loop condition. The CPU or GPU executes instructions one by one; a branch is literally a fork in that instruction road.

```python
# This is a branch. At runtime the program goes one way or the other.
if row_index < num_rows:
    result = compute_attention(q, k)   # path A
else:
    result = 0.0                       # path B
```

Git borrowed the same word for a different thing — a fork in your *commit history*. Same metaphor, completely different domain. Code branching is about runtime execution paths; git branching is about version history.

**On a CPU:** each thread follows its own path. Thread 1 takes path A, Thread 2 takes path B, independently and simultaneously. No wasted work.

**On a GPU warp:** 32 threads must execute the *same instruction* at the *same clock cycle* — they cannot split up. If 16 of those 32 threads need path A and the other 16 need path B, the GPU cannot honour that simultaneously. So it serialises:

1. All 32 threads execute path A. The 16 that didn't need path A have their results **masked off** (discarded).
2. All 32 threads execute path B. The 16 that didn't need path B have their results **masked off**.

You paid for 32 threads. You got 16 units of useful work. That is the cost of a warp divergence.

### Warps and SIMT

A GPU does not schedule individual threads. It schedules **warps** — groups of 32 threads that execute the *same instruction at the same clock cycle*. This model is called **SIMT: Single Instruction Multiple Threads**.

The consequence: if threads within a warp take different code paths (branching — see above), the GPU serialises both paths and masks off threads on the "wrong" branch. You pay for 32 threads and get 1 unit of useful work. This is **warp divergence** — the GPU's primary performance enemy.

For AI inference and training: transformer attention is pure matrix-multiply-accumulate. No branching. Every thread does the same arithmetic on different data. This is maximal GPU efficiency.

### SM — Streaming Multiprocessor

The SM is the execution unit. Think of it as the GPU equivalent of a CPU core, except there are 132 of them on an H100 SXM5.

Each SM contains:
- **Tensor Cores** — hardware accelerators for mixed-precision matrix multiply-accumulate (MMA). FP8, BF16, FP16 → FP32 accumulate. This is the source of the AI FLOPS number (989 TFLOPS at BF16 for H100).
- **CUDA Cores** — general-purpose FP32/INT32 compute.
- **Shared Memory / L1 cache** — 228 KB on H100, shared among threads within the SM. Fast (~5ns), but small. Kernel authors fight over this.
- **Register file** — per-thread, fastest storage (~2ns), limited.

### Tensor Cores

The only thing you need to know: Tensor Cores do one operation — matrix multiply-accumulate — in hardware, in a single clock cycle, for a 4×4 or 8×8 or 16×16 tile of values. They support FP8, BF16, FP16, TF32, and INT8. The precision of the accumulator is always higher than the inputs (e.g., BF16 inputs → FP32 accumulate). This mixed-precision model is how you get speed without losing numerical stability.

When people say "H100 is 3,958 TFLOPS at FP8" — that number is entirely Tensor Core throughput. CUDA Cores alone would give you a fraction of that.

### HBM — High Bandwidth Memory

HBM is the GPU's VRAM. It is **not** DDR, not GDDR. It is 3D-stacked DRAM — multiple layers of memory dies stacked vertically and connected by through-silicon vias (TSVs), mounted on the same substrate as the GPU die.

The result: extremely wide memory buses and correspondingly massive bandwidth.

| Memory Type | Bandwidth | Where |
|-------------|-----------|-------|
| DDR5 (server CPU) | ~100 GB/s | System RAM |
| GDDR6X (gaming GPU) | ~1 TB/s | Consumer GPU |
| HBM3 (H100 SXM) | ~3.35 TB/s | Data centre GPU |
| HBM3e (H200 SXM) | ~4.8 TB/s | Latest gen |

**Why bandwidth matters more than FLOPS for large models:**  
An H100 can execute 3,958 TFLOPS (FP8). But to do that math, it needs to load the operands from HBM into registers. If you can't feed data to the SMs fast enough, they stall. For large language models where the model weights don't fit in the SM caches, you are almost always *memory bandwidth-bound*, not *compute-bound*. This is the single most important insight for understanding GPU performance in AI workloads.

### NVLink

NVLink is NVIDIA's proprietary GPU-to-GPU interconnect. Within a DGX H100 node, the 8 H100 GPUs are fully connected via NVLink 4.0:

- **900 GB/s bidirectional** per GPU
- Compare: PCIe 5.0 (GPU to CPU) = ~128 GB/s

That is a **7× bandwidth advantage** over PCIe. This is why multi-GPU work within a single node is fast. Tensor parallelism (splitting a single model layer across 4 or 8 GPUs) requires constant inter-GPU communication; NVLink is what makes it viable.

### NVSwitch

NVSwitch is the chip that turns point-to-point NVLink connections into a full all-to-all crossbar.

Without NVSwitch: if GPU 0 wants to talk to GPU 3 and GPU 1 wants to talk to GPU 5 simultaneously, those transfers may contend.

With NVSwitch: every GPU can simultaneously talk to every other GPU at full NVLink bandwidth. A DGX H100 has 4 NVSwitches, providing 900 GB/s non-blocking bandwidth between any pair of GPUs.

---

## The bandwidth cliff — the most important number to internalize

| Link | Bandwidth | Context |
|------|-----------|---------|
| HBM3 (H100 VRAM) | 3,350 GB/s | On-package — fastest you get |
| NVLink 4.0 (intra-node GPU↔GPU) | 900 GB/s | Within a DGX node |
| PCIe 5.0 (GPU↔CPU) | 128 GB/s | Host bus |
| InfiniBand NDR 400Gb (inter-node) | ~50 GB/s | Cluster fabric |
| RoCE 400GbE (inter-node) | ~50 GB/s | Cluster fabric |

Once you cross the **node boundary**, you drop from 900 GB/s (NVLink) to ~50 GB/s (fabric). That is an **18× cliff**. Every design decision in distributed training — tensor parallelism, pipeline parallelism, gradient compression, ring-allreduce topology — exists to minimise how much data crosses that cliff, and to hide the latency when it does.

This is the reason cluster networking is not a side concern. It is *the* bottleneck.

---

## Hardware families to know

| System | GPUs | NVLink gen | Inter-node fabric | Use case |
|--------|------|------------|-------------------|----------|
| DGX H100 | 8× H100 SXM5 | NVLink 4.0, 4× NVSwitch | InfiniBand / Ethernet | Single-node training |
| DGX SuperPOD | 256 GPUs (32 nodes) | NVLink within node | InfiniBand NDR | Large-scale training |
| HGX H100 | 8× H100 (OEM form factor) | Same as DGX, no peripherals | Cloud-provider dependent | Cloud, OEMs |
| GB200 NVL72 (Blackwell) | 36× B200 GPUs + 72 GB200 modules per rack | NVLink 5.0, NVSwitch Gen5 | — | Current generation (2025+) |

HGX is DGX without the branding — same silicon, same NVLink, sold to AWS/Azure/GCP for them to package their own way.

---

## What breaks at this layer

| Symptom | Likely cause |
|---------|--------------|
| Low SM utilisation (<50%) during training | Memory bandwidth bottleneck, or collective comm stall |
| Training MFU (Model FLOPS Utilisation) << 50% | Data loading bottleneck, or PCIe congestion |
| NVLink bandwidth low (from DCGM) | Bad cable, NVSwitch failure, or workload not using tensor parallelism |
| XID 74 errors | NVLink link errors — check cables, then NVSwitch health |

---

## Hands-on: commands to run when you have GPU access

```bash
# Print the full NVLink topology — learn to read this matrix
nvidia-smi topo -m

# Check NVLink link state on GPU 0 (12 links on H100)
nvidia-smi nvlink --status -i 0

# Check NVLink throughput counters
nvidia-smi nvlink --getcounters -i 0

# Live bandwidth: how much is going over NVLink right now
nvidia-smi dmon -s u -d 1
```

The `topo -m` output shows a matrix of GPU pairs. Look for `NV4` (NVLink generation 4) between all GPU pairs — that tells you the all-to-all NVSwitch fabric is functioning. `PHB` (PCIe host bridge) between two GPUs means they're not NVLink-connected and will be slow to communicate.

---

## Summary

| Concept | One-liner |
|---------|-----------|
| Code branch | Runtime fork in execution — `if/else`, not git. GPU can't split a warp at a branch. |
| SIMT / warp | 32 threads, same instruction — divergence kills performance |
| SM | Execution unit; H100 has 132; contains Tensor Cores |
| Tensor Cores | Hardware matrix-multiply — source of all AI FLOPS |
| HBM | VRAM; 3.35 TB/s bandwidth; large models are bandwidth-bound, not compute-bound |
| NVLink | Intra-node GPU↔GPU at 900 GB/s; 7× faster than PCIe |
| NVSwitch | All-to-all crossbar for NVLink within a node |
| Bandwidth cliff | 900 GB/s intra-node → 50 GB/s inter-node — this is what the network fight is about |

---

## Why NVIDIA won — and why it's not just the hardware

GPUs were designed for graphics: every pixel needs the same operations on different data. One instruction, 32 pixels, zero divergence. Perfect fit.

Around 2006–2007, researchers noticed matrix multiplication — the backbone of scientific computing and ML — has the same property. Every output element is the same formula on different data. NVIDIA released CUDA in 2007 to let people write general code for the GPU without going through the graphics API. That's the origin of GPGPU (General Purpose GPU).

Transformers arrived later and turned out to be almost pathologically well-suited to this hardware — attention is matrix multiplication all the way down. GPUs weren't designed for AI; AI researchers found a way to express everything as the one operation GPUs were already built to do.

### The real moat: CUDA, not silicon

Many serious attempts at "something different" exist:

| Chip | Idea | Reality |
|------|------|---------|
| Google TPU | Systolic array — designed purely for matrix multiply, no GPU baggage | Technically elegant; locked inside Google Cloud |
| Cerebras | Wafer-scale chip (dinner-plate sized) — eliminate inter-chip bandwidth cliffs entirely | Impressive engineering; very niche |
| Groq | Deterministic execution, no caches — extreme inference throughput | No training story |
| Tenstorrent (Jim Keller) | RISC-V based AI chip — most credible long-term challenger | Still early |

They all hit the same wall: **even if your chip is faster, you're asking researchers to rewrite everything.** PyTorch, JAX, TensorFlow — all CUDA-native. 15 years of optimised libraries (cuDNN, cuBLAS, NCCL), tutorials, tooling, institutional knowledge. Nobody rewrites working code.

Jensen Huang made one deliberate non-lucky bet: he kept funding CUDA through years where nobody was using it for AI and it looked like a money pit. The luck was that the workload materialised. The moat was the decade of software investment before anyone knew it would.

**The broader pattern:** it is not the most elegant solution that wins. It is the solution with the most momentum after the critical window closes. Better hardware loses to entrenched software ecosystems repeatedly throughout computing history (VHS vs Betamax, x86 vs cleaner ISAs, Windows vs everything). NVIDIA is the current example.

---

**Next:** [Block 2 — NVIDIA Software Stack](block2-nvidia-software-stack.md) — how NCCL exploits this hardware, and what DCGM exposes for observability.
