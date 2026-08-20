# AIDC LLM Inference & Serving Pipeline

A production-grade pipeline for open-weight Large Language Model (LLM) serving, hardware capacity sizing, and latency/throughput benchmarking.

## 📊 Hardware Sizing & Capacity Planning

Target Accelerator: **NVIDIA Tesla T4 (16GB VRAM)**  
Deployed Model: `ibm-granite/granite-4.1-3b` (3.4B Parameters)

| Component                     | Allocation (GB) | Description                                            |
| :---------------------------- | :-------------- | :----------------------------------------------------- |
| **Model Weights (FP16)**      | ~6.8 GB         | $3.4 \times 10^9 \text{ params} \times 2\text{ bytes}$ |
| **CUDA & Runtime Overhead**   | ~3.0 GB         | PyTorch context & resident execution space             |
| **KV Cache & Context Buffer** | ~6.2 GB         | Dynamic space for batching and context length          |
| **Total Target Memory**       | **16.0 GB**     | Fits cleanly within T4 capacity without OOM            |

## 🚀 Performance Benchmarking

- **Throughput:** ~17.9 tok/s
- **Latency:** ~6.7s for 120 generated tokens
- **Precision:** FP16 (`torch.float16`)

## 🛠️ Quick Start

### 1. Local / Virtual Environment Setup

```bash
pip install -r requirements.txt
python src/benchmark.py
```
