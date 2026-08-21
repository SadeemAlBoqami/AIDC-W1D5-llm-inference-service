# 🚀 AIDC LLM Inference Service

A production-ready, containerized Large Language Model (LLM) inference microservice designed for AI Data Center environments. Built with **FastAPI**, **PyTorch (CUDA)**, and **Docker**, utilizing **Qwen2.5-0.5B-Instruct** for accelerated local inference.

---

## 📌 Architecture Overview

- **Inference Engine:** PyTorch & Hugging Face Transformers with CUDA FP16 precision.
- **API Framework:** FastAPI with asynchronous request handling and Pydantic validation.
- **Containerization:** NVIDIA Container Runtime over Docker (Ubuntu 22.04 base).
- **Hardware Acceleration:** Native GPU passthrough leveraging NVIDIA Tensor/CUDA cores.

---

## ⚙️ Prerequisites & Environment

- **OS:** Linux / Windows 11 with WSL2 (Ubuntu)
- **GPU:** NVIDIA RTX GPU (CUDA 12.4+ supported)
- **Drivers:** NVIDIA Game Ready / Studio Driver + NVIDIA Container Toolkit
- **Docker:** Docker Desktop with WSL2 integration enabled

---

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SadeemAlBoqami/AIDC-W1D5-llm-inference-service.git
cd AIDC-W1D5-llm-inference-service
```

### 2. Build the Docker Image

```bash
docker build -t aidc-llm-service .
```

### 3. Run with GPU Passthrough

```bash
docker run -d \
  --name llm-gpu-service \
  --gpus all \
  -p 8000:8000 \
  aidc-llm-service
```

### 4. Monitor Initialization Logs

```bash
docker logs -f llm-gpu-service
```

Once initialized, access the interactive Swagger documentation at **http://localhost:8000/docs**.

---

## 📊 API Reference

### Health Check

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "cuda",
  "gpu_available": true
}
```

### Generate Inference

**Endpoint:** `POST /generate`

**Payload:**

```json
{
  "prompt": "What are the core engineering components of an AI Data Center?",
  "max_tokens": 80,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Sample Response:**

```json
{
  "generated_text": "The key components for building an effective AI data center include:\n\n1. **Data Storage**: These servers store all types of structured and unstructured data that is used to train or analyze artificial intelligence models.\n\n2. **AI Application Servers (AAS)**: These handle applications running on top of the underlying infrastructure...",
  "metrics": {
    "latency_sec": 3.11,
    "tokens": 80,
    "throughput_tok_per_sec": 25.69
  }
}
```

---

## 📈 Benchmarks & Performance Metrics

| Metric | Measured Value | Target SLA |
|---|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct | Parameter-efficient Instruct LLM |
| Compute Device | NVIDIA GPU (CUDA FP16) | Accelerated Compute |
| Inference Latency | 3.11 s (for 80 tokens) | < 5.0 s |
| Throughput | 25.69 tokens/sec | > 20 tokens/sec |
| Time to First Token (TTFT) | < 150 ms | Real-time SLA |

---

## 📂 Project Structure

```
aidc-llm-inference-service/
├── Dockerfile              # NVIDIA CUDA runtime build
├── requirements.txt        # FastAPI, Uvicorn, PyTorch, Transformers
├── README.md               # Engineering documentation & benchmarks
└── src/
    ├── benchmark.py        # Automated benchmarking & latency measurement
    └── serve_api.py        # Microservice logic, GPU mapping & metrics

```

---

## 📄 License

This project is distributed under the **MIT License**.
