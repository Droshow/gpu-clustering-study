# Block 7 — Papers Worth Reading

**Audience:** Senior DevOps Engineer — reading to build vocabulary and mental models, not to implement algorithms.  
**Goal:** 5 papers that give you the vocabulary everyone in the GPU cluster space uses, and the background to read architecture docs without guessing at terminology.

---

## How to read these papers as an infra engineer

You are not trying to reproduce the experiments or understand every equation. You are trying to:
1. Understand the problem they are solving
2. Learn the vocabulary they introduce
3. Understand the performance characteristics and trade-offs

Skip the proofs. Read the introduction, the system design section, and the evaluation section. The abstract and conclusion are usually enough to get 80% of the value in 5 minutes. Come back for details when you need them.

---

## 1. Megatron-LM (Shoeybi et al., 2019)

**Full title:** "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"  
**Authors:** NVIDIA Research  
**Link:** https://arxiv.org/abs/1909.08053

### Why read it

This paper defines the vocabulary for model parallelism that the entire industry uses. When you hear "tensor parallelism" and "pipeline parallelism" in architecture discussions, this is where those terms were formalised (tensor parallelism was first clearly described here; pipeline parallelism was extended in later Megatron papers).

### What to extract

**Tensor parallelism:** A single transformer layer's weight matrices are too large for one GPU. Megatron splits them column-wise and row-wise across GPUs. Each GPU computes a partial result; a single all-reduce synchronises after each layer. Requires high-bandwidth NVLink (the communication happens after every layer).

**Pipeline parallelism:** Assign different transformer layers to different GPUs. GPU 0 processes layers 1–8, GPU 1 processes layers 9–16, etc. A "pipeline schedule" keeps all GPUs busy by processing different micro-batches simultaneously (like an assembly line). Fewer all-reduces than tensor parallelism, but adds pipeline bubble overhead.

**The vocabulary:**
- **TP (Tensor Parallel) degree** — how many GPUs a single layer is split across
- **PP (Pipeline Parallel) degree** — how many pipeline stages
- **DP (Data Parallel) degree** — how many replicas of the model run in parallel on different data
- **Total GPU count** = TP × PP × DP

When you see a training job spec like "TP=4, PP=4, DP=16" — that's 256 GPUs total, with 4 GPUs per tensor-parallel group (needing NVLink), 4 pipeline stages (needing fast inter-node links), and 16 data-parallel replicas (needing all-reduce at the DP level).

### Key insight for infra

TP requires very high bandwidth between GPUs (all-reduce after every layer) → only works well within a node over NVLink.  
PP requires moderate bandwidth between pipeline stages (activation tensors, smaller than gradients) → can span nodes.  
DP requires all-reduce of gradients once per iteration → this is the bulk of inter-node traffic.

---

## 2. RDMA over Commodity Ethernet at Scale (Zhu et al., 2015)

**Full title:** "RDMA over Commodity Ethernet at Scale"  
**Authors:** Microsoft Research (Yibo Zhu, Haggai Eran, Daniel Firestone, Chuanxiong Guo, Marina Lipshteyn, Yehonatan Livshin, Ronen Saadon, Shachar Schmid, Liron Tzachanny, Hanoch Weatherspoon)  
**Link:** https://dl.acm.org/doi/10.1145/2829988.2787484  
**Also known as:** The DCQCN paper

### Why read it

This is the foundational paper for the entire "lossless Ethernet" problem that Block 4 describes. Arista's congestion control claims, the UEC work, and every RoCE deployment in existence is built on the problem definition and solution in this paper. Reading it makes Block 4 click.

### What to extract

**The problem:** RoCE (RDMA over Converged Ethernet) requires a lossless fabric. Ethernet drops packets. How do you make Ethernet lossless at datacenter scale?

**PFC alone is insufficient:** PFC pause frames cascade backwards through the fabric, creating head-of-line blocking and pause storms. A single congested link can pause traffic on unrelated links across the datacenter.

**DCQCN:** The paper introduces the DCQCN algorithm — the combination of ECN-based rate control (QCN) applied to RDMA (DCB = Data Center Bridging). The NIC (ConnectX) rate-limits injection based on received CNP (Congestion Notification Packets). This keeps buffers well below the PFC trigger threshold in steady state, using PFC only as a last-resort safety net.

**The key model:** DCQCN is a control loop. The switch signals congestion early via ECN. The NIC receiver generates CNPs. The NIC sender reduces its rate multiplicatively. The rate slowly recovers additively. The parameters (alpha, beta, timer values) require careful tuning for GPU cluster traffic patterns.

### Key vocabulary

- **CNP (Congestion Notification Packet):** sent by the receiver to the sender when it sees ECN marks
- **Rate reduction:** sender cuts injection rate multiplicatively on CNP receipt
- **Rate recovery:** sender increases rate additively when no CNP is received for a timer period
- **Alpha (α):** the aggressiveness parameter for ECN-based congestion detection

---

## 3. vLLM: Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)

**Full title:** "Efficient Memory Management for Large Language Model Serving with PagedAttention"  
**Authors:** UC Berkeley (Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica)  
**Link:** https://arxiv.org/abs/2309.06180

### Why read it

This is the paper that made vLLM. Understanding PagedAttention is required reading for anyone running LLM inference — it explains why vLLM achieves 2–4× higher throughput than naive implementations.

### What to extract

**The problem:** LLM inference has a KV cache that grows dynamically with sequence length. Traditional systems pre-allocate contiguous memory for each request based on max sequence length. Most of that memory is wasted (requests don't hit max length). Additionally, you can't share KV blocks between requests even when they share prefixes.

**PagedAttention:** Map the KV cache through a block table (like OS virtual memory). Physical GPU memory is divided into fixed-size blocks (pages). Each request has a logical sequence of blocks. Blocks are allocated on demand, freed immediately on completion, and can be shared across requests with identical prefixes.

**The results:** Memory utilisation: 20–30% (pre-PagedAttention) → 80–95% (PagedAttention). Throughput: up to 24× higher than FasterTransformer on some workloads.

**Continuous batching:** The paper also formalises continuous batching (iteration-level scheduling) — adding and removing requests from the batch at every decode step, not at fixed intervals.

### Key vocabulary

- **KV cache:** key-value tensors from attention computation; cached to avoid recomputing for earlier tokens
- **Block table:** maps logical KV cache pages to physical memory blocks
- **Prefix caching:** sharing physical blocks for identical prompt prefixes across requests
- **Continuous batching / iteration-level scheduling:** per-step batch management

---

## 4. Megatron-LM: Efficient Large-Scale Language Model Training on GPU Clusters (Narayanan et al., 2021)

**Full title:** "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"  
**Authors:** NVIDIA (Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al.)  
**Link:** https://arxiv.org/abs/2104.04473

### Why read it

The 2021 follow-up to the 2019 Megatron paper. This one combines TP + PP + DP (the "3D parallelism" that all large model training now uses) and provides the first clear analysis of how to optimally allocate GPU counts across the three parallelism dimensions for different model sizes and cluster configurations.

### What to extract

**3D parallelism:** TP × PP × DP. The paper shows how to choose the optimal split for a given model size and GPU cluster configuration. The key insight: TP should be limited to within a node (NVLink bandwidth), PP can span nodes (only activations cross pipeline boundaries, not full gradients), DP is the remaining parallelism and drives inter-node all-reduce traffic.

**The interleaved pipeline schedule:** Previous pipeline schedules had a "pipeline bubble" — some GPUs idle while waiting for the pipeline to fill. The interleaved schedule interleaves multiple model chunks per pipeline stage, reducing the bubble fraction significantly.

**Practical cluster sizing:** The paper gives worked examples for 1T parameter model training. Understanding how they arrived at TP=8, PP=8, DP=N helps you read modern training job specs and understand why clusters are sized the way they are.

---

## 5. AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving (Li et al., 2023)

**Full title:** "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving"  
**Authors:** UC Berkeley / CMU  
**Link:** https://arxiv.org/abs/2302.11665

### Why read it

This paper frames the GPU cluster inference resource allocation problem as statistical multiplexing — a framing that maps directly to how infra teams should think about GPU quota, oversubscription, and multi-model serving. It answers the question: given variable arrival rates and multiple models, how do you allocate GPU resources?

### What to extract

**Statistical multiplexing:** Just like a network multiplexes many flows over one link by taking advantage of the fact that not all flows are busy simultaneously, you can multiplex many models over a GPU cluster because not all models have peak demand simultaneously.

**The model parallelism trade-off:** A larger-than-GPU model deployed with model parallelism (TP/PP) can be served on fewer GPUs if utilisation is low — but this creates latency overhead. The paper models when statistical multiplexing justifies accepting that overhead.

**Practical implications:**
- High-utilisation models: dedicate GPUs, no sharing
- Low-utilisation models: time-share, or use MIG
- Bursty models: plan for peak capacity or use autoscaling with queue depth alerting

### Key vocabulary

- **Statistical multiplexing:** sharing resources among users whose peak demands don't coincide
- **Model placement:** which GPUs serve which model
- **SLO attainment:** fraction of requests meeting the latency service-level objective

---

## Reading order

| Order | Paper | Time investment |
|-------|-------|----------------|
| 1 | RDMA over Commodity Ethernet (DCQCN) | 45 min — read for Block 4 context |
| 2 | vLLM PagedAttention | 30 min — read introduction + system design |
| 3 | Megatron-LM 2019 | 30 min — read introduction + model parallelism section |
| 4 | Megatron-LM 2021 | 20 min — read the 3D parallelism section only |
| 5 | AlpaServe | 20 min — read introduction + the statistical multiplexing framing |

Total: ~2.5 hours. After this, you can read architecture review documents and research blog posts without getting lost on vocabulary.

---

## Bonus: non-paper resources worth your time

| Resource | What you get |
|----------|-------------|
| NVIDIA H100 Architecture Whitepaper (free, NVIDIA.com) | SM architecture, NVLink topology, Transformer Engine — primary hardware reference |
| "Making Deep Learning Go Brrrr From First Principles" (horace.io) | Best practical explanation of roofline model and GPU performance analysis |
| UEC (Ultra Ethernet Consortium) whitepaper | Arista's reference for "why Ethernet can win" — read after Block 4 |
| Lilian Weng's blog (lilianweng.github.io) | Excellent survey posts on attention mechanisms, LLM training techniques — the best quick reference |
