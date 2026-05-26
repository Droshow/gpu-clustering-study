# Block 3 — Hands-On Demo: NVIDIA Stack on Local Kubernetes

**Goal:** Deploy a simulated NVIDIA GPU Operator stack on a local cluster, observe how the CRDs work, simulate NCCL collective communication behaviour, and tie every observable back to what Block 2 and Block 3 taught.

**Environment:** kind (Kubernetes in Docker) — no real GPU required. We mock the device plugin and simulate NCCL behaviour with CPU processes.

---

## Prerequisites

```bash
# Install kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
chmod +x kind && sudo mv kind /usr/local/bin/kind

# Install kubectl (if not already present)
# Verify
kubectl version --client
kind version
```

---

## Part 1 — Spin up a local cluster

```bash
# Create a 4-node cluster (1 control-plane + 3 "GPU" workers)
cat <<EOF | kind create cluster --name gpu-lab --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
    labels:
      nvidia.com/gpu.present: "true"
      nvidia.com/gpu.product: "NVIDIA-H100-SXM5-80GB-SIMULATED"
      nvidia.com/gpu.count: "8"
      nvidia.com/gpu.memory: "81920"
      nvidia.com/mig.capable: "true"
  - role: worker
    labels:
      nvidia.com/gpu.present: "true"
      nvidia.com/gpu.product: "NVIDIA-H100-SXM5-80GB-SIMULATED"
      nvidia.com/gpu.count: "8"
      nvidia.com/gpu.memory: "81920"
      nvidia.com/mig.capable: "true"
  - role: worker
    labels:
      nvidia.com/gpu.present: "true"
      nvidia.com/gpu.product: "NVIDIA-H100-SXM5-80GB-SIMULATED"
      nvidia.com/gpu.count: "8"
      nvidia.com/gpu.memory: "81920"
      nvidia.com/mig.capable: "true"
EOF
```

**What this simulates:** In a real cluster, GFD writes these labels automatically by querying the GPU hardware. Here we write them manually to mimic what GFD produces. This is exactly what you would use in `nodeSelector` to target H100 nodes.

```bash
# Verify labels are present
kubectl get nodes --show-labels | grep nvidia
```

---

## Part 2 — Simulate the GPU Device Plugin (expose nvidia.com/gpu resource)

In production, the Device Plugin DaemonSet exposes `nvidia.com/gpu` as a schedulable resource. We simulate it by patching node capacity directly.

```bash
# Patch each worker node to advertise 8 simulated GPUs
for node in $(kubectl get nodes --selector=nvidia.com/gpu.present=true -o name); do
  kubectl proxy --port=8002 &
  PROXY_PID=$!
  sleep 1

  NODE_NAME=$(echo $node | cut -d/ -f2)

  curl -s -X PATCH \
    http://localhost:8002/api/v1/nodes/${NODE_NAME}/status \
    -H "Content-Type: application/json-patch+json" \
    -d '[
      {"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"8"},
      {"op":"add","path":"/status/allocatable/nvidia.com~1gpu","value":"8"}
    ]'

  kill $PROXY_PID 2>/dev/null
  sleep 1
done

# Verify
kubectl describe nodes | grep -A5 "Allocatable:"
```

**What you are observing:** `nvidia.com/gpu` is now a first-class K8s resource. The scheduler can place pods against it exactly like CPU or memory. Without the device plugin running, this resource is absent — pods requesting it stay `Pending`.

---

## Part 3 — Deploy a ClusterPolicy (simulated GPU Operator)

We deploy the CRD schema and a mock ClusterPolicy to understand the structure. This mirrors what you would apply against a real GPU Operator installation.

```bash
# Install the GPU Operator CRD schema (no controller — just the type)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/main/deployments/gpu-operator/crds/nvidia.com_clusterpolicies.yaml

# Apply a ClusterPolicy that reflects a production H100 configuration
cat <<EOF | kubectl apply -f -
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: gpu-cluster-policy
spec:
  driver:
    enabled: true
    version: "550.90.07"        # pin driver version — uncontrolled upgrades break running jobs
  toolkit:
    enabled: true               # nvidia-container-runtime — required for GPU containers
  devicePlugin:
    enabled: true               # exposes nvidia.com/gpu to the scheduler
  dcgmExporter:
    enabled: true               # Prometheus metrics on each node — your observability source
  mig:
    strategy: single            # all GPUs on a node use the same MIG profile
  migManager:
    enabled: true
  gfd:
    enabled: true               # writes node labels like nvidia.com/gpu.product
EOF

kubectl get clusterpolicy gpu-cluster-policy -o yaml
```

**Key field to understand — `mig.strategy: single`:** means every GPU on a node must be in the same MIG partition mode. If you want `3g.40gb` slices, all 8 GPUs on that node become `3g.40gb`. The MIG Manager DaemonSet reconciles this on the host using `nvidia-smi mig` commands.

---

## Part 4 — Schedule a GPU workload using GFD labels

```bash
# Pod that uses nodeSelector to target H100 nodes — exactly as GFD enables in production
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload-demo
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-SXM5-80GB-SIMULATED
  containers:
    - name: workload
      image: busybox
      command: ["sh", "-c", "echo GPU workload running on H100 node && sleep 3600"]
      resources:
        limits:
          nvidia.com/gpu: "1"
  restartPolicy: Never
EOF

kubectl get pod gpu-workload-demo -o wide
```

**Observe:** The pod lands on a worker node that carries the `nvidia.com/gpu.product` label. In production, change the label value to `NVIDIA-A100-SXM4-80GB` to target A100 nodes instead — no YAML change except the label value.

---

## Part 5 — Simulate NCCL all-reduce and observe SM utilisation oscillation

This is the core Block 2 concept made tangible. We run 3 processes that simulate the compute → wait → compute cycle of a distributed training iteration.

```bash
# Deploy a 3-worker "training simulation" job
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: nccl-allreduce-sim
spec:
  completions: 3
  parallelism: 3
  template:
    spec:
      containers:
        - name: worker
          image: python:3.11-slim
          command:
            - python3
            - -c
            - |
              import time, os, random

              worker_id = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
              print(f"[Worker {worker_id}] Starting distributed training simulation")

              for iteration in range(5):
                  # Phase 1: forward + backward pass (SM util: high)
                  compute_time = random.uniform(1.5, 2.5)
                  print(f"[Worker {worker_id}] Iter {iteration} — COMPUTE (simulated SM util: ~85%) for {compute_time:.1f}s")
                  time.sleep(compute_time)

                  # Phase 2: NCCL all-reduce (SM util: near zero — waiting on network)
                  # In a real job: PyTorch calls ncclAllReduce() here
                  # Worker 1 simulates a slow node (bad NIC / congested link)
                  nccl_time = random.uniform(4.0, 6.0) if worker_id == 1 else random.uniform(4.5, 5.5)
                  print(f"[Worker {worker_id}] Iter {iteration} — NCCL ALL-REDUCE (simulated SM util: ~2%) for {nccl_time:.1f}s")
                  time.sleep(nccl_time)

                  # Phase 3: weight update
                  print(f"[Worker {worker_id}] Iter {iteration} — WEIGHT UPDATE complete. All GPUs now have identical weights.")

              print(f"[Worker {worker_id}] Training complete.")
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
      restartPolicy: Never
EOF
```

**Watch the output in real time:**

```bash
kubectl logs -l job-name=nccl-allreduce-sim --prefix -f
```

**What you are seeing:**
- Each worker cycles: COMPUTE (2s, SM util high) → ALL-REDUCE (5s, SM util near zero) → repeat
- Worker 1 simulates a slow node — in production this would be a congested RoCE link or a flapping NVLink
- **All workers are bottlenecked by the slowest one** — the all-reduce barrier means no worker can proceed to the next iteration until the last one finishes its NCCL step
- This is exactly why `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` dropping on one node stalls the entire job

**MFU calculation from these numbers:**
$$\text{MFU} = \frac{\text{compute time}}{\text{compute time} + \text{all-reduce time}} = \frac{2}{2 + 5.6} \approx 26\%$$

74% of GPU time is wasted waiting for the network. This is the GPU idle problem from Block 2.

---

## Part 6 — Demonstrate partial allocation deadlock (why Volcano exists)

```bash
# First: exhaust most GPU resources with a low-priority job
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: resource-hog
spec:
  completions: 20
  parallelism: 20
  template:
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
        - name: hog
          image: busybox
          command: ["sleep", "120"]
          resources:
            limits:
              nvidia.com/gpu: "1"
      restartPolicy: Never
EOF

# Now try to start a 24-pod training job (needs all GPUs simultaneously)
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job-deadlocked
spec:
  completions: 24
  parallelism: 24
  template:
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
        - name: trainer
          image: busybox
          command: ["sh", "-c", "echo All 24 workers must start together && sleep 60"]
          resources:
            limits:
              nvidia.com/gpu: "1"
      restartPolicy: Never
EOF

# Observe: some pods start, some stay Pending — deadlock
kubectl get pods -l job-name=training-job-deadlocked
```

**What you are observing:** The native K8s scheduler places pods greedily. Some training pods start and hold GPUs. The remaining pods can't start. The started pods can't do useful work without the full gang. GPUs are reserved but idle — exactly the partial allocation deadlock described in Block 3.

```bash
# Clean up
kubectl delete job resource-hog training-job-deadlocked
```

**Resolution in production:** Install Volcano and use `minAvailable: 24` in the `batch.volcano.sh/v1alpha1/Job` spec. Volcano holds all 24 pods until it can schedule all 24 simultaneously.

---

## Part 7 — Inspect what DCGM would expose (metric mapping)

No real DCGM in this lab, but map the simulation output to the metrics you would read in production:

| What you saw in the simulation | DCGM metric in production | Healthy value |
|-------------------------------|--------------------------|---------------|
| COMPUTE phase running | `DCGM_FI_DEV_GPU_UTIL` | >80% |
| ALL-REDUCE phase running | `DCGM_FI_DEV_GPU_UTIL` drops | ~2-5% |
| Worker 1 slow (bad link sim) | `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | Low on that node |
| All workers bottlenecked by Worker 1 | Entire job stalls — all SM utils drop | N/A |
| GPUDirect not working | `DCGM_FI_DEV_PCIE_TX/RX_THROUGHPUT` | High (CPU-mediated) |

```bash
# In a real cluster with dcgm-exporter running, you would query:
# curl http://<node-ip>:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
# or via Prometheus:
# DCGM_FI_DEV_GPU_UTIL{gpu="0",node="gpu-node-01"}
```

---

## Cleanup

```bash
kubectl delete job nccl-allreduce-sim
kubectl delete pod gpu-workload-demo
kubectl delete clusterpolicy gpu-cluster-policy
kind delete cluster --name gpu-lab
```

---

## What this demo covered

| Block 2 concept | Where it appeared in the demo |
|-----------------|------------------------------|
| SM utilisation oscillation | Part 5 — COMPUTE vs ALL-REDUCE phases |
| NCCL as the network bottleneck | Part 5 — slowest worker stalls all others |
| GPU idle problem | Part 5 — MFU calculation |
| DCGM metric mapping | Part 7 — metric table |

| Block 3 concept | Where it appeared in the demo |
|-----------------|------------------------------|
| GFD node labels | Part 1 — manually applied, used in Part 4 nodeSelector |
| Device plugin exposing nvidia.com/gpu | Part 2 — patched node capacity |
| ClusterPolicy structure | Part 3 — applied and inspected |
| Partial allocation deadlock | Part 6 — demonstrated with native scheduler |
| Why Volcano / gang scheduling exists | Part 6 — deadlock shown, Volcano solution explained |

---

**Next:** [Block 4 — Network Layer Demo](block4-demo.md) — RoCE vs InfiniBand, PFC/ECN congestion control, and where Arista switches fit in the GPU fabric.
