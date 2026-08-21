# 🚀 AIDC LLM Inference Service

A production-ready, containerized Large Language Model (LLM) inference microservice designed for AI Data Center environments. Built with **FastAPI**, **PyTorch (CUDA)**, and **Docker**, utilizing **Qwen2.5-0.5B-Instruct** for accelerated local inference.

---

## 📌 Architecture Overview

* **Inference Engine:** PyTorch & Hugging Face Transformers with CUDA FP16 precision.
* **API Framework:** FastAPI with asynchronous request handling and Pydantic validation.
* **Containerization:** NVIDIA Container Runtime over Docker (Ubuntu 22.04 base).
* **Hardware Acceleration:** Native GPU passthrough leveraging NVIDIA Tensor/CUDA cores.

---

## ⚙️ Prerequisites & Environment

* **OS:** Linux / Windows 11 with WSL2 (Ubuntu)
* **GPU:** NVIDIA RTX GPU (CUDA 12.4+ supported)
* **Drivers:** NVIDIA Game Ready / Studio Driver + NVIDIA Container Toolkit
* **Docker:** Docker Desktop with WSL2 integration enabled

---

## 🛠️ Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/aidc-llm-inference-service.git
cd aidc-llm-inference-service
