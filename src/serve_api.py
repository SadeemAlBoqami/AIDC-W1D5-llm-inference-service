import time
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(
    title="AIDC LLM Inference Service",
    description="Production-ready inference endpoint for open-weight LLMs",
)

MODEL_ID = "ibm-granite/granite-4.1-3b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing model {MODEL_ID} on {device.upper()}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 120


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device, "model": MODEL_ID}


@app.post("/generate")
def generate(req: InferenceRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=req.max_tokens, do_sample=False
        )
    elapsed_time = time.time() - start_time

    generated_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    throughput = generated_tokens / elapsed_time if elapsed_time > 0 else 0

    response_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()

    return {
        "generated_text": response_text,
        "metrics": {
            "latency_sec": round(elapsed_time, 2),
            "tokens": generated_tokens,
            "throughput_tok_per_sec": round(throughput, 2),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)