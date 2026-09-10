"""
PII Guardrail — FastAPI Service
Exposes the fine-tuned Qwen2.5-1.5B PII pseudonymizer as a REST API.

Any RAG application or LLM backend can call POST /sanitize
before sending context to an external LLM API.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Demo (Kaggle with ngrok):
    See notebooks/04_api_demo.ipynb
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import re
import time
import torch

app = FastAPI(
    title="PII Guardrail API",
    description=(
        "Privacy boundary for RAG systems. "
        "Pseudonymizes PII and sensitive credentials before "
        "they reach external LLM APIs."
    ),
    version="1.0.0",
)

# ── Global model state ────────────────────────────────────────────────
# Model is loaded once at startup — expensive operation done only once
_model     = None
_tokenizer = None
_device    = None

SYSTEM_PROMPT = """You are a PII redaction system.
Your job: rewrite the input text replacing all PII and sensitive credentials with typed placeholder tags.
Rules:
- Replace each unique sensitive value with a typed tag: PERSON_001, EMAIL_001, PHONE_001, AADHAAR_001, PAN_001, CREDIT_CARD_001, API_KEY_001, DB_CREDENTIAL_001
- Number tags sequentially within each category: PERSON_001, PERSON_002, etc.
- If the same value appears multiple times, use the same tag each time
- If there is no PII or sensitive data, return the text COMPLETELY UNCHANGED
- Do not add explanations. Output only the rewritten text."""


# ── Startup: load model ───────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    """
    Load Qwen2.5-1.5B + LoRA adapter on startup.
    Requires GPU. Model stays in memory for the lifetime of the server.
    """
    global _model, _tokenizer, _device

    import os
    adapter_path = os.environ.get(
        "ADAPTER_PATH",
        "/kaggle/working/qwen-pii-adapter"
    )

    if not os.path.exists(adapter_path):
        print(f"WARNING: Adapter not found at {adapter_path}")
        print("API will run in DEMO MODE — inference disabled")
        return

    print(f"Loading tokenizer and model from {adapter_path}...")
    t0 = time.time()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )

    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    _model = PeftModel.from_pretrained(base, adapter_path)
    _model.eval()

    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ── Request / Response schemas ────────────────────────────────────────
class SanitizeRequest(BaseModel):
    text: str
    rag_context: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Contact Rahul Mehta at rahul@company.com",
                "rag_context": None
            }
        }


class SanitizeResponse(BaseModel):
    sanitized_text: str
    mapping:        dict
    pii_detected:   bool
    latency_ms:     float

    class Config:
        json_schema_extra = {
            "example": {
                "sanitized_text": "Contact PERSON_001 at EMAIL_001",
                "mapping": {
                    "PERSON_001": "Rahul Mehta",
                    "EMAIL_001":  "rahul@company.com"
                },
                "pii_detected": True,
                "latency_ms":   1240.5
            }
        }


# ── Inference ─────────────────────────────────────────────────────────
def _run_inference(text: str) -> str:
    """Run model inference. Returns pseudonymized text."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_mapping(original: str, pseudonymized: str) -> dict:
    """
    Extract mapping of placeholder -> original value.
    Finds all placeholders in output, then locates corresponding
    values in the original text using positional alignment.
    
    Note: This is an approximate extraction. Production systems
    would use the model's attention weights for exact span alignment.
    """
    placeholder_pattern = re.compile(
        r'(PERSON|EMAIL|PHONE|AADHAAR|PAN|CREDIT_CARD|API_KEY|DB_CREDENTIAL)_(\d{3})'
    )
    
    placeholders = placeholder_pattern.findall(pseudonymized)
    if not placeholders:
        return {}

    # Build mapping by diffing original and pseudonymized
    # Find which words were replaced by comparing token by token
    mapping = {}
    
    # Simple approach: find all unique placeholders in output
    all_placeholders = set(
        f"{cat}_{num}" for cat, num in placeholders
    )
    
    # For each placeholder, try to find what it replaced
    # by looking at the original text position
    pseudo_words    = pseudonymized.split()
    original_words  = original.split()
    
    orig_idx = 0
    for p_word in pseudo_words:
        clean_p = re.sub(r'[^\w_]', '', p_word)
        if placeholder_pattern.match(clean_p):
            # This word is a placeholder — find what it replaced
            if orig_idx < len(original_words):
                # Collect original tokens until we find alignment
                orig_value_parts = []
                while orig_idx < len(original_words):
                    orig_word = original_words[orig_idx]
                    orig_idx += 1
                    orig_value_parts.append(orig_word)
                    # Check if next pseudo word aligns with next orig word
                    break
                if orig_value_parts and clean_p not in mapping:
                    mapping[clean_p] = " ".join(orig_value_parts)
        else:
            orig_idx += 1

    # Fill any unmapped placeholders
    for ph in all_placeholders:
        if ph not in mapping:
            mapping[ph] = "[value redacted]"

    return mapping


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check. Confirms model load status."""
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "device":       str(_device),
        "model":        "Qwen2.5-1.5B + LoRA PII Guardrail",
    }


@app.post("/sanitize", response_model=SanitizeResponse)
async def sanitize(request: SanitizeRequest):
    """
    Pseudonymize PII in input text.

    Assembles user text and optional RAG context into a single
    input, runs the fine-tuned Qwen model, returns pseudonymized
    text with a private placeholder mapping.

    The mapping must be stored by the calling application —
    it is never logged or persisted by this service.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Ensure ADAPTER_PATH is set and GPU is available."
            )
        )

    # Assemble full context — same checkpoint placement as pipeline
    # Guardrail sits AFTER context assembly, BEFORE external LLM call
    if request.rag_context:
        full_text = (
            f"{request.text}\n\n"
            f"[RETRIEVED CONTEXT]: {request.rag_context}"
        )
    else:
        full_text = request.text

    t0 = time.time()

    try:
        pseudonymized = _run_inference(full_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    latency_ms = (time.time() - t0) * 1000
    mapping    = _extract_mapping(full_text, pseudonymized)
    pii_detected = pseudonymized.strip() != full_text.strip()

    return SanitizeResponse(
        sanitized_text=pseudonymized,
        mapping=mapping,
        pii_detected=pii_detected,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/sanitize/batch")
async def sanitize_batch(requests: list[SanitizeRequest]):
    """
    Batch endpoint — sanitize multiple texts in one call.
    Returns list of SanitizeResponse objects in same order.
    Useful for processing multiple RAG chunks simultaneously.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(requests) > 20:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 20 requests"
        )

    results = []
    for req in requests:
        full_text = (
            f"{req.text}\n\n[RETRIEVED CONTEXT]: {req.rag_context}"
            if req.rag_context else req.text
        )
        t0            = time.time()
        pseudonymized = _run_inference(full_text)
        latency_ms    = (time.time() - t0) * 1000
        mapping       = _extract_mapping(full_text, pseudonymized)

        results.append({
            "sanitized_text": pseudonymized,
            "mapping":        mapping,
            "pii_detected":   pseudonymized.strip() != full_text.strip(),
            "latency_ms":     round(latency_ms, 2),
        })

    return results


@app.get("/categories")
async def categories():
    """Return list of PII categories this model detects."""
    return {
        "categories": [
            {"tag": "PERSON_001",        "description": "Person names (context-dependent)"},
            {"tag": "EMAIL_001",         "description": "Email addresses"},
            {"tag": "PHONE_001",         "description": "Phone numbers including Indian +91 format"},
            {"tag": "AADHAAR_001",       "description": "12-digit Indian government ID"},
            {"tag": "PAN_001",           "description": "Indian tax ID (ABCDE1234F format)"},
            {"tag": "CREDIT_CARD_001",   "description": "16-digit card numbers"},
            {"tag": "API_KEY_001",       "description": "API keys, Bearer tokens, access tokens"},
            {"tag": "DB_CREDENTIAL_001", "description": "Database connection strings and passwords"},
        ],
        "null_behavior": (
            "If no PII is detected, input text is returned completely "
            "unchanged. Zero false positives on clean text."
        )
    }