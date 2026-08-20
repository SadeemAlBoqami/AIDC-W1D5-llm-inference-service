import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_benchmark(
    model_id: str = "ibm-granite/granite-4.1-3b",
    prompt: str = "Explain what a data center is in two clear sentences.",
    max_new_tokens: int = 100,
):
    print(f"--- Starting Benchmark for: {model_id} ---")

    # 1. فحص جهاز التشغيل المتاح
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device.upper()}")

    # 2. تحميل التوكنايزر والنموذج بدقة float16
    print("Loading tokenizer and model weights...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # 3. إعداد المدخلات
    inputs = tok(prompt, return_tensors="pt").to(device)

    # 4. عملية التوليد وقياس الزمن
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    dt = time.time() - t0

    # 5. استخراج المخرجات وحساب المقاييس
    generated_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    tok_per_sec = generated_tokens / dt if dt > 0 else 0
    response = tok.decode(
        out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()

    # 6. طباعة التقرير
    print("\n--- Generation Output ---")
    print(response)
    print("\n--- Performance Metrics ---")
    print(f"Latency: {dt:.2f} seconds")
    print(f"Tokens Generated: {generated_tokens}")
    print(f"Throughput: {tok_per_sec:.2f} tok/s")


if __name__ == "__main__":
    run_benchmark()