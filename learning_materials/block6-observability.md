# Block 6 — Observability

**Audience:** Senior DevOps Engineer — you own the monitoring stack.  
**Goal:** Know what to watch, what thresholds mean something, what XID errors require what action, and which commands to run first when something is wrong.

---

## The observability stack

```
GPU Hardware
    └── NVML (NVIDIA Management Library — C API)
            └── DCGM (Data Center GPU Manager — daemon, aggregates NVML)
                    └── dcgm-exporter (Prometheus /metrics endpoint per node)
                            └── Prometheus (scrape + store)
                                    └── Grafana (dashboard) / Alertmanager (alerts)
```

You configure `dcgm-exporter` via the GPU Operator's `ClusterPolicy`. It runs as a DaemonSet on every GPU node. Prometheus scrapes it. You alert on the metrics.

---

## The metrics that matter

### GPU Compute

| Metric | Normal | Investigate if |
|--------|--------|----------------|
| `DCGM_FI_DEV_GPU_UTIL` | >80% during training | <50% sustained — GPU is starved or waiting on network |
| `DCGM_FI_DEV_SM_CLOCK` | At or near base clock | Significantly below base — thermal or power throttle |
| `DCGM_FI_DEV_POWER_USAGE` | 400–700W (H100 TDP: 700W) | >700W impossible (hardware limited); sustained low (<200W during training) = throttling |
| `DCGM_FI_DEV_GPU_TEMP` | 50–80°C | >85°C — thermal throttle will begin |

**Reading GPU utilisation correctly:** During distributed training, SM utilisation is **not** steady at 90%. It oscillates:
- High (compute): 85–95% — forward + backward pass running
- Low (network wait): 0–10% — NCCL all-reduce in progress

The ratio between high and low phases tells you how network-bound you are. If you see 40% average utilisation and the oscillation is pronounced, the bottleneck is the network, not the compute.

### Memory

| Metric | What it means |
|--------|---------------|
| `DCGM_FI_DEV_FB_USED` | VRAM in use (bytes). H100 has 80GB. |
| `DCGM_FI_DEV_FB_FREE` | VRAM available |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | HBM bandwidth utilisation |

**VRAM OOM pattern:** `FB_USED` steadily increases → job crashes with CUDA OOM. Usually means: KV cache grew beyond available VRAM (inference), or gradient accumulation with too large a batch (training). The OOM itself doesn't produce an XID error — it's a CUDA error in the application.

**Memory bandwidth utilisation:** `MEM_COPY_UTIL` at 100% = memory-bound (expected for large models during decode). At 20% during training = you have a compute-bound workload (actually good). Low memory bandwidth during training often means the GPU is stalled on something else (network wait, PCIe transfers).

### NVLink

| Metric | What it means |
|--------|---------------|
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | Total NVLink throughput across all links. Low during tensor-parallel training = collective comm bottleneck. |
| `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` | CRC errors on NVLink. Should be 0. Any nonzero: cable issue or NVSwitch problem. |
| `DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESS_COUNT_TOTAL` | NVLink link recovery events. Occasional OK; frequent = hardware degrading. |

### PCIe

| Metric | Normal | Red flag |
|--------|--------|----------|
| `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | Low during GPU-GPU comms | High during training = GPUDirect RDMA not working; data routing through host RAM |
| `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` | Same | Same |

If `PCIE_TX/RX_THROUGHPUT` is high while `NVLINK_BANDWIDTH` is also high during distributed training, data is going PCIe → host RAM → NIC instead of GPU → NIC directly. Check: `nvidia-peermem` module loaded, `NicClusterPolicy` deployed, `NCCL_DEBUG=INFO` output.

### Errors

| Metric | What to do |
|--------|-----------|
| `DCGM_FI_DEV_XID_ERRORS` | Any nonzero — check `dmesg` for XID code, consult table below |
| `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | Single-bit ECC errors (corrected). Monitor rate — increasing rate means HBM degrading |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | Double-bit ECC errors (uncorrected). Any occurrence is serious — data corruption possible |

---

## XID error playbook

XID errors are surfaced in three places:
1. `dmesg` on the node: `NVRM: Xid (PCI:...): 79, ...`
2. DCGM metric: `DCGM_FI_DEV_XID_ERRORS` increments
3. GPU Operator logs (if configured to forward)

| XID | Name | Severity | Action |
|-----|------|----------|--------|
| 8 | GPU diagnostic failure | High | Run `dcgmi diag -r 3` full diagnostic; if it fails, file RMA |
| 13 | Graphics Engine Exception | Medium | Usually software bug (bad pointer access by CUDA kernel); check application code first |
| 31 | GPU memory page fault | Medium | CUDA kernel accessing invalid memory address; check application; if recurring on clean workload = HBM fault |
| 43 | GPU stopped processing | High | NVIDIA driver crashed; `sudo systemctl restart nvidia-dcgm`; if recurring = GPU failing |
| 48 | DBE (double-bit ECC error) | Critical | Data corruption occurred; drain node immediately; schedule GPU replacement |
| 63 | Row remapping (SBE ECC) | Low–Medium | HBM memory cell degraded; ECC corrected it. Monitor `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`. Accelerating rate = schedule replacement |
| 74 | NVLink error | Medium | Check NVLink cable, NVSwitch health; `nvidia-smi nvlink --status -i <gpu_id>` |
| 79 | GPU fallen off PCIe bus | Critical | Fatal hardware error. Node needs reboot. If XID 79 recurs after reboot = GPU failing. File RMA. |
| 92 | High SBE ECC rate | High | HBM degrading faster than normal; drain node, schedule replacement |
| 94 | Contained Channel Error | High | GPU self-contained the error; workload may have been terminated. Check if reboot clears it. |

**XID 79 procedure:**
```bash
# 1. Cordon node immediately (workload scheduler)
kubectl cordon <node>

# 2. Check dmesg for context
dmesg | grep -i "xid\|nvrm" | tail -50

# 3. Reboot
sudo reboot

# 4. After reboot: run full diagnostic
dcgmi diag -r 3 -i <gpu_id>

# 5. If XID 79 recurs: file NVIDIA RMA
```

---

## nvidia-smi commands — build fluency

```bash
# Topology — which GPUs are connected by NVLink, which by PCIe
# NV4 = NVLink gen 4; PHB = PCIe host bridge (slow); SYS = across NUMA nodes
nvidia-smi topo -m

# Live per-GPU utilisation, memory, power — refreshes every 1s
nvidia-smi dmon -s u -d 1

# More detail: SM util, memory util, memory used, power, temperature
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu \
           --format=csv -l 1

# NVLink link status for GPU 0 — shows each of the 18 links on H100
nvidia-smi nvlink --status -i 0

# NVLink throughput counters (useful during training to see if links are active)
nvidia-smi nvlink --getcounters -i 0

# Process list — which PIDs are using which GPU and how much VRAM
nvidia-smi

# MIG instance list (if MIG is enabled)
nvidia-smi mig -lgip      # list GPU instance profiles
nvidia-smi mig -lgi       # list active GPU instances
nvidia-smi mig -lci       # list active compute instances
```

---

## DCGM CLI commands

```bash
# List all GPUs detected
dcgmi discovery -l

# Run health check (quick)
dcgmi health -g 1 -c

# Run full diagnostic (takes minutes, tests memory, compute, bandwidth)
dcgmi diag -r 3

# Watch live field values
# Field IDs: 203=SM util, 204=mem util, 155=NVLink BW, 1005=XID errors
dcgmi dmon -e 203,204,155,1005 -d 1000

# Show all available fields
dcgmi field --list | grep -i "nvlink\|xid\|ecc\|util"
```

---

## Alerting thresholds — reasonable starting points

These are starting points. Tune based on your workload patterns.

| Metric | Alert condition | Severity |
|--------|----------------|----------|
| `DCGM_FI_DEV_XID_ERRORS` | `> 0` (any XID in last 5 minutes) | Warning (check XID code) |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | `> 0` | Critical (drain node) |
| `DCGM_FI_DEV_GPU_TEMP` | `> 85°C` for 5 minutes | Warning |
| `DCGM_FI_DEV_SM_CLOCK` | `< 80% of base clock` for 10 minutes during active workload | Warning (thermal/power throttle) |
| `DCGM_FI_DEV_GPU_UTIL` | `< 30%` for 30 minutes during scheduled training window | Warning (job stalled?) |
| `DCGM_FI_DEV_FB_USED / FB_TOTAL` | `> 95%` | Warning (OOM risk) |
| `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` | `rate > 0 over 5m` | Warning (NVLink degrading) |

---

## Network observability (Arista side)

On the fabric side, the metrics to watch for RoCE health:

```eos
! Arista EOS — check interface counters for congestion indicators
show interfaces counters rates

! PFC pause frame counters — should be near zero in a healthy cluster
show qos interface <intf> counters

! ECN marking counters — non-zero is normal; very high = congestion present
show qos interface <intf> ecn counters

! Buffer utilisation
show platform sand qos queue counters
```

A healthy GPU cluster Ethernet fabric shows:
- PFC pause frames: near zero (ECN is doing its job before buffers fill)
- ECN marks: non-zero and proportional to traffic load
- Interface error counters: exactly zero

If PFC pause frames are increasing: either ECN thresholds are tuned too high (ECN isn't catching congestion early enough), or there's a genuine congestion event (traffic burst exceeding link capacity).

---

## Correlating GPU metrics with network metrics

The failure modes where you need both sides:

| Symptom | GPU metric | Network metric | Likely cause |
|---------|-----------|----------------|--------------|
| Low training throughput | Low `GPU_UTIL` during "network wait" phase | High PFC pause counts | Network congestion causing NCCL to stall |
| NCCL timeout / job hang | `GPU_UTIL` drops to 0 and stays | Interface errors or link down | Physical link failure |
| Slow but not hung | `GPU_UTIL` oscillates widely | ECN marks elevated | Network congestion; buffers filling; DCQCN reducing rates |
| High PCIe bandwidth | High `PCIE_TX/RX` | Normal network stats | GPUDirect RDMA not working; CPU mediating transfers |

---

## Summary — first response checklist

When a GPU job is slow or hung:

```bash
# 1. Is the GPU computing at all?
nvidia-smi dmon -s u -d 1 | head -20

# 2. Any hardware errors?
dmesg | grep -i "xid\|nvrm" | tail -20

# 3. What is NCCL doing?
# (set NCCL_DEBUG=INFO before job start — check job logs)

# 4. Is GPUDirect working?
dcgmi dmon -e 203,1009,1010 -d 1000  # 1009/1010 = PCIe TX/RX

# 5. NVLink health
nvidia-smi nvlink --status -i 0
nvidia-smi nvlink --getcounters -i 0

# 6. DCGM health check
dcgmi health -g 1 -c
```

---

**Next:** [Block 7 — Papers](block7-papers.md) — the 5 papers that give you the vocabulary to read architecture docs and talk to researchers.
