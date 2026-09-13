"""
PII Guardrail — FastAPI Service on HuggingFace Space
Uses Groq cloud inference instead of loading model locally.
No GPU required. Runs on CPU Basic tier.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import os
import re
import json
import time

app = FastAPI(
    title="PII Guardrail API",
    description="Privacy boundary for RAG systems. Pseudonymizes PII before external LLM calls.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load router at startup (small pkl file, fast) ─────────────────────
router = None

SYSTEM_PROMPT = """You are a PII redaction system.
Your job: rewrite the input text replacing all PII and sensitive credentials with typed placeholder tags.
Rules:
- Replace each unique sensitive value with a typed tag: PERSON_001, EMAIL_001, PHONE_001, AADHAAR_001, PAN_001, CREDIT_CARD_001, API_KEY_001, DB_CREDENTIAL_001
- Number tags sequentially within each category: PERSON_001, PERSON_002, etc.
- If the same value appears multiple times, use the same tag each time
- If there is no PII or sensitive data, return the text COMPLETELY UNCHANGED
- Do not add explanations. Output only the rewritten text."""


@app.on_event("startup")
async def startup():
    global router
    try:
        with open("router_pipeline.pkl", "rb") as f:
            router = pickle.load(f)
        print("Router loaded successfully")
    except Exception as e:
        print(f"Router load failed: {e}")


# ── Schemas ───────────────────────────────────────────────────────────
class SanitizeRequest(BaseModel):
    text: str
    rag_context: Optional[str] = None

class SanitizeResponse(BaseModel):
    sanitized_text: str
    mapping: dict
    pii_detected: bool
    complexity: str
    latency_ms: float


# ── Inference via Groq ────────────────────────────────────────────────
def call_groq(text: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "router_loaded": router is not None,
        "inference": "groq-llama-3.1-8b",
    }


@app.post("/sanitize", response_model=SanitizeResponse)
async def sanitize(request: SanitizeRequest):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    full_text = (
        f"{request.text}\n\n[RETRIEVED CONTEXT]: {request.rag_context}"
        if request.rag_context else request.text
    )

    # Route query
    complexity = "simple"
    if router:
        try:
            complexity = router.predict([request.text])[0]
        except Exception:
            complexity = "simple"

    t0 = time.time()
    try:
        result = call_groq(full_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    latency_ms = (time.time() - t0) * 1000

    pattern = re.compile(
        r'(PERSON|EMAIL|PHONE|AADHAAR|PAN|CREDIT_CARD|API_KEY|DB_CREDENTIAL)_(\d{3})'
    )
    placeholders = set(f"{c}_{n}" for c, n in pattern.findall(result))
    mapping = {p: "[original value — stored privately]" for p in sorted(placeholders)}

    return SanitizeResponse(
        sanitized_text=result,
        mapping=mapping,
        pii_detected=result.strip() != full_text.strip(),
        complexity=complexity,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/categories")
async def categories():
    return {
        "categories": [
            "PERSON", "EMAIL", "PHONE", "AADHAAR",
            "PAN", "CREDIT_CARD", "API_KEY", "DB_CREDENTIAL"
        ]
    }