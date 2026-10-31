# Production Cloud Deployment Guide (AWS & GCP Scale-to-Zero MLOps)

AegisVision AI is architected to run in **three deployment topologies** depending on latency SLAs and cloud budget constraints:

1. **Zero-Idle-Cost Serverless GPU on Google Cloud Run (`nvidia-l4`, `min-instances=0`)**
2. **AWS ECS Fargate + Amazon SageMaker Serverless Inference (Terraform IaC)**
3. **Edge Industrial IPC / Jetson Orin (`TensorRT 10 FP16/INT8`)**

---

## 1. Google Cloud Run with Scale-to-Zero NVIDIA L4 GPU

Google Cloud Run supports attaching an **NVIDIA L4 (24GB VRAM)** accelerator directly to a stateless container while preserving `minScale: 0`. When no factory line or reviewer is invoking the endpoint, compute scales to **0 instances ($0.00/hr)**.

### Step-by-Step Deployment Commands
```bash
# 1. Authenticate and configure Artifact Registry
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud artifacts repositories create ai-portfolio \
  --repository-format=docker \
  --location=us-central1

# 2. Submit Cloud Build pipeline (builds image & deploys Cloud Run L4 GPU service)
gcloud builds submit --config deploy/gcp/cloudbuild.yaml .

# Or deploy directly via declarative Knative service manifest:
gcloud run services replace deploy/gcp/service.yaml --region=us-central1
```

---

## 2. AWS ECS Fargate + Amazon SageMaker Serverless Inference (Terraform)

Keeping a dedicated `g5.xlarge` GPU EC2 instance running 24/7 costs ~$730/month. For bursty or multi-tenant inspection workloads, AegisVision decouples:
- **Control Plane & Hybrid RAG Copilot (`FastAPI` + `LangGraph` + `Qdrant`)**: Hosted on **AWS ECS Fargate Spot**.
- **Quantized CV Model Execution (`YOLO26` + `Detectron2` ONNX INT8)**: Hosted on **Amazon SageMaker Serverless Inference** (`4096 MB`, `max_concurrency = 10`), which scales down to zero automatically when idle.

### Step-by-Step Terraform Provisioning
```bash
cd deploy/aws
terraform init
terraform plan -var="aws_region=us-east-1" -var="environment=production"
terraform apply -auto-approve
```

---

## 3. Exporting PyTorch Weights to Quantized ONNX & TensorRT

```python
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Export YOLO26 NMS-Free Detection / Segmentation head to ONNX FP16
model = YOLO("yolo26s-seg.pt")
model.export(format="onnx", opset=18, dynamic=True, half=True)

# 2. Apply QDQ INT8 Dynamic Quantization for CPU / Serverless Edge
quantize_dynamic(
    model_input="yolo26s-seg.onnx",
    model_output="yolo26s-seg.int8.onnx",
    weight_type=QuantType.QInt8,
)
```
