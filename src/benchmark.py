import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_benchmark(
    model_id: str = "ibm-granite/granite-4.1-3b",
    prompt: str = "Explain the role of an AI Data Center in modern infrastructure.",
    max_new_tokens: int = 120,
):
    print(f"==================================================")
    print(f" Running AIDC Inference Benchmark: {model_id}")
    print(f"==================================================")

    # 1. تحديد العتاد
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device.upper()}")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"[*] Initial Allocated VRAM: {initial_vram:.2f} GB")

    # 2. تحميل التوكنايزر والنموذج
    print("[*] Loading Model Weights into Memory...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cuda":
        loaded_vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"[*] Model Weights Footprint: {loaded_vram:.2f} GB")

    # 3. إعداد المدخلات
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 4. عملية التوليد وقياس الأداء
    print("[*] Executing Forward Pass / Generation...")
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    elapsed_time = time.time() - start_time

    # 5. حساب المقاييس
    input_tokens = inputs["input_ids"].shape[-1]
    total_tokens = output.shape[-1]
    generated_tokens = total_tokens - input_tokens
    throughput = generated_tokens / elapsed_time if elapsed_time > 0 else 0

    response = tokenizer.decode(
        output[0][input_tokens:], skip_special_tokens=True
    ).strip()

    # 6. تقرير القياس النهائي
    print("\n---------------- GENERATION RESULT ----------------")
    print(response)
    print("---------------------------------------------------")
    print(f"[*] Latency:            {elapsed_time:.2f} s")
    print(f"[*] Tokens Generated:   {generated_tokens} tokens")
    print(f"[*] Throughput:         {throughput:.2f} tok/s")

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[*] Peak VRAM Utilized: {peak_vram:.2f} GB")
        print(f"[*] Total VRAM Capacity: 16.00 GB (Tesla T4)")
    print(f"==================================================\n")


if __name__ == "__main__":
    run_benchmark()
