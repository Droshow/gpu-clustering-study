# Block 4 — The Network Layer (Where the Arista Fight Lives)

**Audience:** Senior DevOps Engineer with Arista context.  
**Goal:** Understand why the network is the GPU cluster bottleneck, what InfiniBand and RoCE actually are, how lossless Ethernet is achieved, and where Arista's product play fits.

---

## Why the network is the bottleneck — the math

From Block 2: all-reduce across a 256-GPU cluster on a 70B BF16 model requires every GPU to exchange ~140GB of gradients, every training iteration.

Let's be precise about the ring all-reduce algorithm:

In a ring topology of N GPUs, all-reduce requires each GPU to send a total of `2 × (N-1)/N × data_size` bytes. For N=256 and 140GB:

$$\text{bytes per GPU} = 2 \times \frac{255}{256} \times 140\text{GB} \approx 279\text{GB}$$

At 400Gbps per link = 50 GB/s:

$$\text{all-reduce time} = \frac{279\text{GB}}{50\text{GB/s}} = 5.6\text{ seconds}$$

If the forward + backward compute takes 2 seconds, GPU utilisation is:

$$\text{MFU} = \frac{2}{2 + 5.6} \approx 26\%$$

74% of training time is GPUs sitting idle waiting for the network. This is why every basis point of network bandwidth improvement has direct dollar value at scale.

---

## InfiniBand — NVIDIA's solution

InfiniBand (IB) was designed for high-performance computing in the 1990s. NVIDIA acquired Mellanox (the dominant IB vendor) in 2020 for $6.9B specifically to own this stack end-to-end.

### Why InfiniBand is fast

**Lossless by design.** IB uses credit-based flow control. Before a sender transmits data, it must have "credits" from the receiver indicating buffer availability. If the receiver's buffer is full, the sender simply waits — no packet is sent, no packet is dropped. The network never drops a packet under normal operation.

This matters enormously for RDMA. RDMA does not have a TCP-style retransmit mechanism. If a packet is dropped, the RDMA operation fails and the entire collective communication must restart. IB's credit-based flow control makes this a non-issue.

**Very low latency.** ~1μs end-to-end. This is relevant for small-message collectives (all-reduce on small gradient tensors, or barrier synchronisation between pipeline stages).

**High bandwidth.** Current generation:
- HDR InfiniBand: 200 Gbps per port
- NDR InfiniBand: 400 Gbps per port  
- XDR (next): 800 Gbps per port

NVIDIA's Quantum-X800 switch: 64 ports × 800 Gbps = 51.2 Tbps per switch.

### The lock-in problem

NVIDIA controls the entire IB stack: switches (Quantum series), NICs (ConnectX), cables (QSFP), drivers (MOFED), and the subnet manager (UFM). If you run IB, you are running NVIDIA hardware end-to-end. This is a vendor lock-in story that NVIDIA's sales team considers a feature and everyone else considers a risk.

Cost: IB switches are 2–4× the cost of equivalent-speed Ethernet switches.

---

## RDMA over Converged Ethernet (RoCE) — Arista's foundation

RoCE (pronounced "rocky") is the technology that lets Ethernet carry RDMA traffic. Specifically: take the InfiniBand transport layer and run it over Ethernet frames instead of IB links.

**RoCEv1:** RDMA over Ethernet frames. Layer 2 only — cannot cross routers. Mostly obsolete.

**RoCEv2:** RDMA over UDP/IP. Can route across a layer 3 fabric. This is what production GPU clusters use.

### The fundamental problem with Ethernet for RDMA

Ethernet was designed to drop packets. When a switch buffer fills up, it drops packets. TCP handles this gracefully — the sender detects the drop via timeout or SACK and retransmits. Application barely notices.

RDMA does not work this way. An RDMA operation is a single unacknowledged transfer. If a packet is dropped mid-transfer, the entire RDMA operation fails. At the NCCL level: the all-reduce fails, the training job fails or hangs.

**The lossless Ethernet challenge:** you need to prevent packet drops in the switches, not just recover from them.

---

## How lossless Ethernet works — PFC, ECN, DCQCN

This is the core technical concept behind RoCE deployments. Three mechanisms work together:

### PFC — Priority Flow Control (IEEE 802.1Qbb)

PFC is a pause mechanism. When a switch egress port buffer reaches a threshold, it sends a PAUSE frame to the upstream sender, per-traffic-class. The sender stops transmitting that traffic class until it receives an UNPAUSE.

```
GPU Node A  →  Switch Port 1  →  Switch Port 2  →  GPU Node B
                                       ↑ buffer filling
                    ← PAUSE frame (traffic class 3) ←
GPU Node A pauses                 
```

Traffic class segmentation: PFC operates per-class. You mark your RDMA traffic (RoCE) as a specific DSCP/PCP value, map it to its own priority queue, and only that queue gets paused. Storage, management, and regular Ethernet traffic continue unaffected.

**The PFC problem: head-of-line blocking and pause storms.**  
A PAUSE frame from one congested port propagates backwards through the fabric. In a misconfigured network, pauses cascade from switch to switch, eventually pausing links that are not congested — causing a "pause storm" that freezes traffic cluster-wide. This is the most common failure mode in new RoCE deployments.

### ECN — Explicit Congestion Notification (RFC 3168)

ECN is a proactive signal. When a switch buffer exceeds a lower threshold (before it's full enough to trigger PFC), it marks packets with the CE (Congestion Experienced) bit instead of dropping them. The receiver reflects this mark back to the sender in its ACK. The sender reduces its transmission rate before the buffer overflows.

ECN prevents the situation where PFC is the only signal — it's an earlier warning.

### DCQCN — Datacenter QCN

DCQCN (Datacenter Quantized Congestion Notification) is the congestion control algorithm that ties PFC and ECN together for RoCE. Developed by Mellanox and Microsoft, described in the 2015 paper "RDMA over Commodity Ethernet at Scale."

The algorithm:
1. Switch marks packets with ECN when buffer crosses first threshold
2. RoCE receiver reflects CNP (Congestion Notification Packet) back to sender
3. Sender reduces its injection rate (rate limiting in the ConnectX NIC)
4. Rate slowly recovers as congestion clears
5. PFC acts as the safety backstop if ECN+DCQCN wasn't enough

**The key insight:** ECN + DCQCN keep the buffers well below the PFC trigger threshold in steady state. PFC becomes an infrequent emergency brake rather than the primary congestion mechanism. A well-tuned RoCE network rarely triggers PFC.

---

## The lossless Ethernet configuration

This is what network teams aren't used to configuring. An Arista switch for GPU cluster fabric needs:

```eos
! Traffic class definition — RDMA gets its own class
traffic-class map RDMA-TRAFFIC
   cos 3 to tc 3
   dscp 24 to tc 3     ! CS3 DSCP marking for RoCE

! PFC — enable per-class pause on traffic class 3
qos profile GPU-FABRIC
   pfc pause traffic-class 3

! ECN thresholds — mark early, pause late
qos profile GPU-FABRIC
   ecn threshold minimum 150 kilobytes maximum 1500 kilobytes traffic-class 3

! Deep buffer configuration — absorb microbursts without triggering PFC
platform sand qos input traffic-class 3 minimum-threshold 500 kilobytes
```

The buffer sizing matters. GPU all-reduce creates **elephant flows** — large, long-lived, high-bandwidth flows that fill switch buffers. Deep buffer switches (like Arista 7800R3) are designed to absorb these bursts while ECN and DCQCN reduce the sender rate. IB uses credit-based flow control and rarely needs large buffers; Ethernet with PFC/ECN needs deep buffers as the absorption layer.

---

## The routing problem — per-packet ECMP

Traditional ECMP (Equal-Cost Multi-Path) does per-flow load balancing: it hashes a 5-tuple (src IP, dst IP, src port, dst port, protocol) and sends the entire flow down one path.

GPU all-reduce creates **elephant flows**: a single NCCL all-reduce generates a few very large flows between pairs of GPUs, not thousands of small flows. Per-flow ECMP fails here because:
- One flow saturates one path
- Other paths sit idle
- Utilisation is uneven — some paths are hot, others are cold

What you want: **per-packet (adaptive) load balancing** — distribute individual packets across all available equal-cost paths, balancing utilisation regardless of flow count.

Arista's 7800R3 and 7700R series support:
- **Adaptive routing** — monitors link utilisation and re-routes flows towards less loaded paths dynamically
- **Per-packet ECMP** — enabled on specific queues for RDMA traffic

The tradeoff: per-packet load balancing can cause **out-of-order packet delivery**. For TCP, this is handled by the resequencing buffer. For RDMA, out-of-order packets can trigger NACK storms and degrade performance. Arista's adaptive routing is designed to minimise out-of-order delivery while still achieving load balancing.

---

## The Arista play — specific products and positioning

### Products

**Arista 7800R3** — AI spine switch
- 460.8 Tbps switching capacity
- Deep buffers (important for burst absorption)
- Adaptive routing
- CloudVision integration

**Arista 7700R series** — AI leaf/spine  
- High-density 400GbE / 800GbE ports
- Designed specifically for GPU cluster leaf-spine topologies

**Arista EOS** — the differentiator at the software layer
- Consistent CLI/API across all platforms (unlike Cisco/Juniper multi-OS complexity)
- CloudVision: centralised telemetry, streaming gRPC telemetry, configuration management
- Programmable forwarding pipeline for custom congestion control tuning

### Ultra Ethernet Consortium (UEC)

Arista is a founding member alongside AMD, Broadcom, Cisco, Intel, Meta, and Microsoft. The UEC is standardising Ethernet extensions for AI:
- Standardised congestion control (superseding DCQCN with a more robust algorithm)
- Multipathing at the transport layer
- Better RDMA semantics over Ethernet

This is Arista's moat: they helped write the standard and they're validating switches against UEC test suites. When enterprises want "RDMA over Ethernet that works," Arista's switches are the reference implementation.

---

## The current landscape (2026)

| Player | Bet | Reality |
|--------|-----|---------|
| NVIDIA | InfiniBand — full stack control | Dominant in HPC and enterprise GPU clusters buying complete NVIDIA solutions |
| Meta | Ethernet (custom) — own the whole stack | Running massive RoCE clusters; don't need IB because they have network engineers to tune Ethernet |
| Microsoft (Azure) | Ethernet — InfiniBand is too expensive at scale | Building Ethernet-based GPU clusters; DCQCN paper came from Microsoft Research |
| Google | Custom (TPU fabric + custom Ethernet) | Largely orthogonal to this fight |
| Arista | Ethernet — win the enterprise refresh cycle | Winning deals where enterprises already have Arista and don't want dual fabric operations |
| Cisco | Ethernet (late) | Playing catch-up; joined UEC; less relevant in pure AI infra |

**The enterprise sales motion:** An enterprise already has Arista 7050 or 7280 switches running their data centre Ethernet. They want to add a GPU cluster. Arista's pitch: don't deploy a separate IB fabric that your network team doesn't know how to operate — extend your existing Ethernet fabric with 7800R3 spines, configure PFC/ECN/DCQCN (Arista can automate this via EOS/CVP), and you have lossless RoCE without retraining your team or adding a parallel management plane.

---

## What can go wrong — failure modes in RoCE deployments

| Failure | Symptom | Root cause |
|---------|---------|------------|
| Pause storm | Training jobs hang, cluster-wide throughput collapse | PFC misconfigured; one congested link propagates pauses across fabric |
| Per-flow ECMP imbalance | Some links at 90%, others at 10%; slow training | ECMP hashing on 5-tuple; elephant flows not distributed |
| MTU mismatch | RDMA operations fail; XID errors in NCCL | RoCE requires jumbo frames (9000 MTU) end-to-end; one hop with 1500 MTU causes fragmentation/drops |
| DSCP remarking | RDMA traffic not getting PFC protection | Intermediate switch remarking DSCP bits, traffic falls into wrong queue |
| `nvidia-peermem` not loaded | NCCL falls back to TCP | `lsmod | grep nvidia_peermem`; fix: load module, check Network Operator config |

**The MTU issue is the most common first-deployment problem.** RoCE relies on jumbo frames (9000 byte MTU) to achieve efficient large-message transfers. If any switch or host NIC in the path has MTU set to 1500, the path's effective MTU is 1500. RDMA transfers will be fragmented or fail depending on the implementation. Check `ip link show` on every GPU node and `show interfaces | grep mtu` on every switch in the path.

---

## Summary

| Concept | One-liner |
|---------|-----------|
| All-reduce bottleneck | 256 GPUs × 140GB gradients / 50GB/s = 5.6s idle per iteration |
| InfiniBand | Lossless by design (credit-based); NVIDIA owns the full stack; 2-4× cost premium |
| RoCEv2 | RDMA over UDP/IP; requires lossless Ethernet config to work reliably |
| PFC | Per-class pause frames — prevents drops but causes pause storms if misconfigured |
| ECN | Early congestion signal — marks packets before buffer is full |
| DCQCN | The algorithm that combines ECN + PFC for RoCE; sender self-limits on CNP receipt |
| Adaptive routing | Per-packet/adaptive ECMP — distributes elephant flows evenly across paths |
| Arista's play | UEC member, deep buffer switches, EOS/CVP automation; winning enterprise refresh cycle |
| MTU | Jumbo frames (9000) required end-to-end for RoCE — check every hop |

---

**Next:** [Block 5 — Inference Infrastructure](block5-inference-infrastructure.md) — where workloads actually run: vLLM, Triton, TensorRT-LLM, and NIM.
