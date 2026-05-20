# 🚀 PROJECT MANIFESTO: The AI-Ready Networking Lab

**User Role:** DevOps Engineer (AWS/Cloud Native focus)  
**Objective:** Build a hands-on demo and high-impact LinkedIn content showcasing the convergence of **AI Infrastructure**, **Infrastructure as Code (Terraform)**, and **Arista Networks**.  
**Budget:** ~$0–5 total. No real GPU cluster needed — the narrative *"I simulated a $500k AI spine on my laptop"* is the hook.

---

## 1. Context & Background (May 2026)

- **The Company:** Arista Networks (ANET) is currently the leader in high-speed cloud networking, hitting **$11.5B+** annual revenue with **35%+ YoY growth**.
- **The Visionary:** CEO **Jayshree Ullal** is pushing a "Cloud-First/AI-First" agenda. Her core thesis is that **Ethernet is the "Eventual Winner"** over proprietary systems like NVIDIA's InfiniBand for large-scale AI clusters.
- **The Tech Shift:** Transitioning from "Front-end" networking (client-server) to **"Back-end" AI Spines** (GPU-to-GPU). This requires a "Lossless" network to support **RoCE** (RDMA over Converged Ethernet).

---

## 2. Technical Core Concepts

- **EOS (Extensible OS):** A single-image, Linux-based network OS that runs identically across every Arista device (Hardware, Virtual, or Cloud).
- **CloudEOS:** The cloud-native version of EOS available on AWS/Azure, manageable via API.
- **NetDevOps:** The methodology of managing networks using the same CI/CD pipelines and IaC tools used for apps.
- **The AI Bottleneck:** AI training jobs are "Elephant Flows." If the network has jitter or packet loss, billion-dollar GPU clusters sit idle. Arista solves this with deep buffers and intelligent congestion control.

---

## 3. The Tech Stack for the Demo

| Layer | Tool / Service | Cost |
|---|---|---|
| Local Lab | **cEOS** (containerized EOS) in Docker | $0 |
| Infrastructure | AWS (VPC, EC2) — optional, for screenshots | ~$0.04/hr, destroy after |
| Network Layer | **CloudEOS Free Trial** (AWS Marketplace) | $0 software license |
| Orchestrator | **Terraform** (`aristanetworks/cloudeos` provider) | $0 |
| Configuration Intent | **Arista AVD** — YAML-based framework for defining network state | $0 |
| Visibility | **Arista CloudVision** — optional, screenshots only | Free trial |

---

## 4. Agent Execution Roadmap (To-Do List)

### Phase 0: Local Lab (Free — Start Here)

- [ ] **Install Docker Desktop** on Windows → Settings → WSL Integration → enable Ubuntu.
- [ ] **Install containerlab:** `bash -c "$(curl -sL https://get.containerlab.dev)"`
- [ ] **Get cEOS image:** Register free at arista.com → Software Downloads → EOS → cEOS-lab → download `.tar.xz`.
- [ ] **Import image:** `docker import cEOS64-lab-<version>.tar.xz ceos:latest`
- [ ] **Deploy topology:** `cd lab && sudo clab deploy -t topology.clab.yml`
- [ ] **Verify lab:** `pip install requests && python3 lab/verify_lab.py` — confirms BGP is up and routes are in the table.

> 💡 This phase alone produces 80% of the article content at $0 cost.  
> Files ready: `lab/topology.clab.yml`, `lab/configs/spine1.cfg`, `lab/configs/leaf1.cfg`, `lab/verify_lab.py`

### Phase 1: AWS Deployment (Optional — ~$2–5 total)

- [ ] **Provision Infra:** Use Terraform to create a Transit VPC hosting the Arista CloudEOS Free Trial instance.
- [ ] **API Access:** Enable Arista **eAPI** (JSON-RPC) on the virtual instance.
- [ ] **Connectivity:** Peer one Spoke VPC to the Transit Hub — one peer is enough for the screenshot.
- [ ] **💰 Destroy immediately after screenshots:** `terraform destroy` — don't leave it running.

### Phase 2: "AI-Ready" Configuration (The Secret Sauce)

- [ ] **Define Lossless Fabric:** Code the configuration for **Priority Flow Control (PFC)** and **ECN (Explicit Congestion Notification)**.
- [ ] **BGP Automation:** Set up automated BGP peering between the CloudEOS router and AWS VPC virtual gateways.
- [ ] **Validation:** Write a simple Python script or use a Terraform test to verify that the routing table has updated across the fabric.

### Phase 3: Content Creation (LinkedIn Strategy)

- [ ] **The "Visual":** Generate a diagram showing the "AI Spine" architecture vs. a traditional "Tree" network.
- [ ] **The "Hook":** Draft a post explaining how $30,000 GPUs are useless if the network isn't "Lossless."
- [ ] **The Technical Showcase:** Share a snippet of the Terraform code managing the Arista EOS state, emphasizing the **"Infrastructure as Code"** for networking.

---

## 5. Key Keywords & Buzzwords for 2026

| Term | Description |
|---|---|
| **UEC (Ultra Ethernet Consortium)** | The standard Arista is helping lead |
| **RoCEv2** | The protocol that lets GPUs talk memory-to-memory |
| **Universal AI Spine** | The architectural role Arista plays in the data center |
| **Digital Twin** | Using vEOS to simulate a network before pushing to production |

---

## Next Immediate Steps for Agent

1. **Provide a `docker-compose.yml`** to spin up a 2-node cEOS topology locally (Phase 0).
2. **Provide a boilerplate `main.tf`** using `aristanetworks/cloudeos` to deploy a single CloudEOS Free Trial instance on AWS (Phase 1, optional).
3. **Draft the LinkedIn article outline** based on the cEOS local demo narrative.
