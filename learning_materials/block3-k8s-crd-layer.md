# Block 3 — NVIDIA on Kubernetes (the CRD Layer)

**Audience:** Senior DevOps Engineer — you know Kubernetes. This is specifically what NVIDIA adds on top.  
**Goal:** Understand the CRDs you own, what the GPU Operator manages, how nodes get GPU labels, how InfiniBand/RoCE gets exposed to pods, and why gang scheduling matters.

---

## The problem NVIDIA's K8s stack solves

Kubernetes out of the box knows nothing about GPUs. A pod spec with `nvidia.com/gpu: 1` would fail without the driver stack installed, the device plugin exposing that resource, and the container runtime configured to mount the GPU into the container. NVIDIA's Kubernetes stack solves all of that — but you need to understand each piece to operate it.

---

## GPU Operator

The GPU Operator is a **meta-operator**: a single operator that manages all the other components of the NVIDIA software stack on K8s nodes. One CRD (`ClusterPolicy`) drives everything.

### What it installs and manages

| Component | What it does |
|-----------|--------------|
| NVIDIA Driver | Installs and manages the `nvidia.ko` kernel module on each node |
| Container Toolkit | Configures `nvidia-container-runtime` so containers can access GPUs |
| Device Plugin | Exposes `nvidia.com/gpu` as a K8s schedulable resource |
| DCGM Exporter | Runs DCGM daemon + Prometheus exporter on each GPU node |
| GPU Feature Discovery (GFD) | Labels nodes with GPU capabilities |
| MIG Manager | Manages MIG partition configuration |
| Node Status Exporter | Reports GPU node health to K8s |

### ClusterPolicy — the one CRD you own

```yaml
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: gpu-cluster-policy
spec:
  driver:
    enabled: true
    version: "550.90.07"       # pin this; uncontrolled driver upgrades break things
  toolkit:
    enabled: true              # nvidia-container-runtime — required for GPU containers
  devicePlugin:
    enabled: true              # exposes nvidia.com/gpu to the scheduler
  dcgmExporter:
    enabled: true              # Prometheus metrics endpoint on each node
  mig:
    strategy: single           # 'single': all GPUs on a node in same MIG mode
                               # 'mixed': each GPU can have different MIG config
  migManager:
    enabled: true
  nodeStatusExporter:
    enabled: true
  gfd:
    enabled: true              # GPU Feature Discovery
```

**`mig.strategy: single` vs `mixed`:**
- `single`: every GPU on the node presents the same MIG profile. Simpler scheduling — all pods on that node see `nvidia.com/mig-3g.40gb`.
- `mixed`: each GPU can have a different partition layout. More flexible, harder to schedule correctly.

### What happens when the GPU Operator is healthy

```bash
# Check operator pods
kubectl get pods -n gpu-operator

# Expected: one pod per component per node
# nvidia-driver-daemonset-*
# nvidia-container-toolkit-daemonset-*
# nvidia-device-plugin-daemonset-*
# nvidia-dcgm-exporter-*
# gpu-feature-discovery-*
```

If `nvidia-device-plugin-daemonset` is not running, `nvidia.com/gpu` will not appear as a node resource and all GPU pod scheduling will fail silently (pods stuck in `Pending`).

---

## GPU Feature Discovery (GFD) labels

GFD queries the GPU on each node and writes the results as Kubernetes node labels. These are your node selectors — use them instead of hardcoding node names.

```
nvidia.com/gpu.product=NVIDIA-H100-SXM5-80GB
nvidia.com/gpu.memory=81920
nvidia.com/gpu.count=8
nvidia.com/mig.capable=true
nvidia.com/mig.strategy=single
feature.node.kubernetes.io/pci-10de.present=true    # NVIDIA PCI device present
```

**Usage in pod specs:**

```yaml
# Target H100 nodes only
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-H100-SXM5-80GB

# Or use nodeAffinity for more complex rules
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.memory
              operator: Gt
              values: ["40960"]    # GPU with >40GB VRAM
```

---

## Network Operator — for InfiniBand and RoCE

**Separate from the GPU Operator.** The Network Operator manages the Mellanox/NVIDIA ConnectX NIC driver stack and the RDMA layer. Without it, NCCL cannot use GPUDirect RDMA and falls back to CPU-mediated transfers (PCIe bandwidth in DCGM goes up; training speed goes down).

### Key CRD: NicClusterPolicy

```yaml
apiVersion: mellanox.com/v1alpha1
kind: NicClusterPolicy
metadata:
  name: nic-cluster-policy
spec:
  ofedDriver:
    image: mofed                        # Mellanox OFED — the userspace RDMA driver
    repository: nvcr.io/nvidia/mellanox
    version: "23.10-0.5.5.0"
  rdmaSharedDevicePlugin:
    image: k8s-rdma-shared-dev-plugin
    repository: ghcr.io/mellanox
    version: v1.3.2
    resources:
      - name: rdma_shared_device_a
        vendors: ["15b3"]               # 15b3 = Mellanox/NVIDIA PCI vendor ID
  sriovDevicePlugin:
    image: sriov-network-device-plugin
    repository: ghcr.io/k8snetworkplumbingwg
    version: v3.6.2
    resources:
      - name: sriov_a
        vendors: ["15b3"]
        devices: ["1017"]               # 1017 = ConnectX-6 Dx device ID
```

**What MOFED does:** OFED (OpenFabrics Enterprise Distribution) is the userspace RDMA driver stack. It provides the `ibverbs` API that NCCL uses to do RDMA. Without MOFED inside the container (or on the host, depending on deployment model), NCCL cannot use InfiniBand or RoCE — it falls back to TCP.

**What the RDMA Shared Device Plugin does:** Exposes the RDMA device (e.g., `rdma_shared_device_a`) as a K8s resource, so pods can request RDMA access in their resource spec.

**What SR-IOV Device Plugin does:** For RoCE over SR-IOV (creating virtual functions on the ConnectX NIC), this exposes VFs as schedulable resources. Used when you want hardware isolation per pod at the NIC level.

### Pod spec requesting RDMA

```yaml
resources:
  limits:
    nvidia.com/gpu: "8"
    rdma/rdma_shared_device_a: "1"    # request RDMA access
```

If a pod requests GPUs but not RDMA, it will still run — but NCCL will not use RDMA transport. Check `NCCL_DEBUG=INFO` output to confirm which transport was selected.

---

## RDMA — the concept that ties it together

**RDMA (Remote Direct Memory Access)** means: a NIC on Machine A can read or write memory on Machine B, without involving Machine B's CPU. No kernel involvement, no system call, no memory copy on the receiving side.

**GPUDirect RDMA** extends this: the NIC can read/write GPU HBM directly. No staging through host DRAM. No CPU involvement.

The data path without GPUDirect RDMA:
```
GPU HBM → PCIe → Host DRAM → CPU copy → Host DRAM → PCIe → NIC → wire
```

The data path with GPUDirect RDMA:
```
GPU HBM → PCIe → NIC → wire
```

The performance difference is significant for large tensor transfers (all-reduce gradients, KV cache transfers for disaggregated inference). GPUDirect RDMA halves the effective latency and removes a major CPU bottleneck.

> **DevOps perspective — what RDMA actually is:**
>
> Normal network transfer path:
> ```
> Sender CPU → copies data to NIC buffer → wire → Receiver NIC → interrupt → Receiver CPU → copies to app memory
> ```
>
> RDMA path:
> ```
> Sender NIC → wire → Receiver NIC → directly into app memory
>                                     ↑ receiver CPU never involved
> ```
>
> No system call on the receiver. No kernel interrupt. No memory copy. The NIC handles it entirely
> in hardware using a protocol called **RDMA verbs** (`ibverbs` API) — which is exactly what NCCL
> calls when `NicClusterPolicy` is deployed correctly.
>
> GPUDirect RDMA takes it one step further: the NIC skips host DRAM entirely and reads/writes GPU
> HBM directly over PCIe. The CPU is completely uninvolved in moving the gradient tensor.
>
> The infra signal that GPUDirect is NOT working: `DCGM_FI_DEV_PCIE_TX/RX_THROUGHPUT` will be high.
> That means transfers are staging through host DRAM via the CPU — the slow path. When GPUDirect is
> working correctly, PCIe bandwidth to the CPU is low and NVLink bandwidth is high.

**Requirements for GPUDirect RDMA:**
1. NVIDIA GPU (A100/H100 or newer recommended)
2. Mellanox ConnectX-5 or newer NIC
3. MOFED driver installed
4. `nvidia-peermem` kernel module loaded (bridges GPUDirect with RDMA verbs)
5. `NicClusterPolicy` deployed with correct configuration

> **DevOps perspective — what a NIC is in a GPU cluster:**
>
> A NIC (Network Interface Card) is the physical hardware that connects a machine to a network.
> In a standard server it handles regular Ethernet (`eth0`). In a GPU cluster the NIC is a
> **Mellanox/NVIDIA ConnectX** card — it handles both Ethernet (for RoCE) and InfiniBand, and
> critically, supports RDMA in hardware.
>
> ```
> Standard NIC (e.g. Intel e810):
>   - Ethernet only
>   - All transfers go through CPU/kernel
>   - Cannot do RDMA
>
> ConnectX-6/7 NIC (Mellanox/NVIDIA):
>   - Ethernet + InfiniBand
>   - RDMA in hardware (ibverbs API)
>   - GPUDirect RDMA capable — DMA engine talks directly to GPU HBM
> ```
>
> Full hardware picture on a GPU node:
> ```
> GPU (H100) ←—NVLink—→ GPU (H100)
>      ↕ PCIe                ↕ PCIe
> ConnectX NIC ←—IB/RoCE—→ ConnectX NIC (on another node)
>      ↕
> Arista switch (Block 4)
> ```
>
> In the `NicClusterPolicy`, `vendors: ["15b3"]` is the PCI vendor ID for Mellanox — the device
> plugin uses it to discover ConnectX NICs on each node. `devices: ["1017"]` narrows it to
> ConnectX-6 Dx specifically. This is how K8s learns which hardware is present and exposes it
> as the `rdma/rdma_shared_device_a` schedulable resource.

---

## Time-slicing vs MIG — K8s scheduling implications

| | Time-slicing | MIG |
|---|---|---|
| K8s resource name | `nvidia.com/gpu` | `nvidia.com/mig-3g.40gb` (etc.) |
| Isolation | None | Hard (hardware-enforced) |
| Noisy neighbour risk | High | None |
| Scheduling granularity | Per GPU (shared) | Per MIG instance |
| Typical use | Dev environments, low-priority batch | Production multi-tenant inference |
| Memory accounting | Each consumer sees full GPU VRAM | Capped to slice size |

One practical gotcha with time-slicing: if you configure a node to expose 4× time-slices per GPU, K8s sees 4 `nvidia.com/gpu` resources per GPU. A pod requesting 4 `nvidia.com/gpu` can be scheduled to a single physical GPU with all 4 slices — which may or may not be what you want. Resource requests don't mean isolation.

---

## Volcano — gang scheduling

The default Kubernetes scheduler is designed for microservices. It does not understand that a 256-GPU training job needs all 256 pods to start *simultaneously* to be useful. Without gang scheduling:

1. 200 GPU pods start, acquire their GPUs, and block waiting for the other 56.
2. The other 56 GPUs are allocated to different jobs — or not available yet.
3. 200 GPUs sit idle, reserved but doing no work.
4. The training job never starts. Other jobs can't use the reserved GPUs.

This is called **partial allocation deadlock**. It is a real production failure mode in GPU clusters.

**Volcano** solves this with gang scheduling:

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-training-job
spec:
  minAvailable: 256          # all-or-nothing: schedule only if 256 pods can start
  schedulerName: volcano
  tasks:
    - replicas: 256
      name: worker
      template:
        spec:
          schedulerName: volcano
          containers:
            - name: trainer
              image: nvcr.io/nvidia/pytorch:24.01-py3
              resources:
                limits:
                  nvidia.com/gpu: "1"
          restartPolicy: OnFailure
```

`minAvailable: 256` means Volcano will not schedule any of the 256 pods until it can schedule all 256 simultaneously. If only 200 GPUs are free, the job waits. When 256 GPUs are available, all 256 pods start at the same time.

**Volcano also handles:**
- **Queue management:** multiple teams share the cluster with quota enforcement
- **Preemption:** lower-priority jobs yield to higher-priority jobs (with gang awareness — it preempts whole jobs, not individual pods)
- **Fair-share scheduling:** time-based GPU allocation across teams

**Ray** is the alternative for Python-native distributed workloads (primarily inference autoscaling). Ray handles its own distributed runtime and doesn't need Volcano — it manages workers internally.

---

## Cluster topology — what you actually deploy

```
┌─────────────────────────────────────────────────────┐
│  GPU Node                                           │
│  ┌──────────────────────────────────────────────┐  │
│  │  nvidia-driver-daemonset  (nvidia.ko loaded) │  │
│  │  nvidia-container-toolkit (runtime config)   │  │
│  │  nvidia-device-plugin     (exposes GPU res.) │  │
│  │  dcgm-exporter            (Prometheus scrape)│  │
│  │  gpu-feature-discovery    (node labels)      │  │
│  │  network-operator pods    (MOFED + RDMA)     │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ↑
    ClusterPolicy (GPU Operator) + NicClusterPolicy (Network Operator)
    both deployed once, reconcile forever
```

---

## Common failure modes

| Symptom | Likely cause | Check |
|---------|--------------|-------|
| Pods stuck in `Pending`, `nvidia.com/gpu: 0/0` | Device plugin not running | `kubectl get ds -n gpu-operator` |
| Pod starts but can't find GPU | Container toolkit misconfigured | Check `nvidia-container-runtime` config on node |
| NCCL using TCP fallback | MOFED not loaded, or `nvidia-peermem` missing | `NCCL_DEBUG=INFO`, check `lsmod | grep nvidia_peermem` |
| Gang-scheduled job never starts | Not enough GPUs free for `minAvailable` | Check Volcano queue status: `vcctl queue list` |
| MIG pods see wrong resource | MIG manager not reconciled | Check MIG manager pod logs, `nvidia-smi mig -lgip` |

---

## Summary

| CRD / Component | Who manages it | What you configure |
|-----------------|----------------|--------------------|
| `ClusterPolicy` | GPU Operator | Driver version, MIG strategy, which components are enabled |
| `NicClusterPolicy` | Network Operator | MOFED version, RDMA device plugin resources |
| GFD labels | GPU Operator (GFD) | Read-only — used in `nodeSelector`/`nodeAffinity` |
| Volcano `Job` | You | `minAvailable` for gang scheduling |
| `nvidia.com/gpu` resource | Device Plugin | Requested in pod `resources.limits` |
| `nvidia.com/mig-Xg.Ygb` resource | MIG Manager + Device Plugin | Requested in pod `resources.limits` for MIG workloads |

---

**Next:** [Block 4 — The Network Layer](block4-network-layer.md) — why the network is the actual bottleneck, InfiniBand vs RoCE, PFC/ECN/DCQCN, and where Arista fits.
