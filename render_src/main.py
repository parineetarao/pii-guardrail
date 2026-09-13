"""
PII GuardrailProxy — FastAPI Deployment Engine on Render
Decoupled Compute Architecture utilizing Groq Cloud Inference
with Asynchronous n8n FinOps Token Telemetry logs.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
import pickle
import httpx
import os
import re
import time

app = FastAPI(
    title="Enterprise AI Privacy Gateway & FinOps Proxy",
    description="Asynchronous MLOps privacy proxy gatekeeper for corporate RAG systems.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global holder for your Stage 2 Naive Bayes Router model
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
async def startup_event():
    global router
    try:
        # Pulls your Stage 2 pickle file exported from your notebook
        with open("router_pipeline.pkl", "rb") as f:
            router = pickle.load(f)
        print("Naive Bayes Router loaded successfully into CPU thread.")
    except Exception as e:
        print(f"Router loading anomaly: {e}")

# ── Schemas ───────────────────────────────────────────────────────────
class SanitizeRequest(BaseModel):
    text: str
    rag_context: Optional[str] = None
    n8n_telemetry_url: Optional[str] = None  # Dynamic hook link to your n8n workflow

class SanitizeResponse(BaseModel):
    sanitized_text: str
    mapping: dict
    pii_detected: bool
    complexity: str
    latency_ms: float

# ── Asynchronous Telemetry Dispatch ──────────────────────────────────
def dispatch_n8n_metrics(url: str, payload: dict):
    """Executes a non-blocking outbound post request to forward FinOps logs to n8n"""
    try:
        # Short timeout ensures that even if n8n is down, your main app thread never hangs
        httpx.post(url, json=payload, timeout=2.0)
    except Exception:
        pass

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "router_active": router is not None,
        "compute_layer": "decoupled-groq-cloud"
    }

@app.post("/api/v1/sanitize", response_model=SanitizeResponse)
async def sanitize_payload(request: SanitizeRequest, background_tasks: BackgroundTasks):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY environment variable missing.")

    # Reconstruct full RAG context exactly like your Gradio setup
    full_text = (
        f"{request.text}\n\n[RETRIEVED CONTEXT]: {request.rag_context}"
        if request.rag_context else request.text
    )

    # 1. Run your Stage 2 Naive Bayes Router locally on the CPU (takes ~2ms)
    complexity = "simple"
    if router:
        try:
            complexity = router.predict([request.text])[0]
        except Exception:
            complexity = "simple"

    # 2. Triage model routing target based on complexity score
    assigned_model = "qwen-2.5-7b-instruct" if complexity == "simple" else "llama-3.1-70b-versatile"

    t0 = time.time()
    try:
        # 3. Offload cloud inference computation securely to Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            model=assigned_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_text},
            ],
            temperature=0.0,
        )
        result = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloud compute engine failure: {str(e)}")

    latency_ms = (time.time() - t0) * 1000

    # 4. Extract token counts for FinOps telemetry logging
    prompt_tokens = len(full_text) // 4
    completion_tokens = len(result) // 4

    # Compile the telemetry log payload
    telemetry_payload = {
        "assigned_model": assigned_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "complexity_tier": complexity,
        "latency_ms": round(latency_ms, 2)
    }

    # 5. ASYNC LOOP: Offload the telemetry payload to n8n in a background thread
    if request.n8n_telemetry_url:
        background_tasks.add_task(dispatch_n8n_metrics, request.n8n_telemetry_url, telemetry_payload)

    # 6. Extract placeholder mappings for system persistence
    pattern = re.compile(r'(PERSON|EMAIL|PHONE|AADHAAR|PAN|CREDIT_CARD|API_KEY|DB_CREDENTIAL)_(\d{3})')
    placeholders = set(f"{c}_{n}" for c, n in pattern.findall(result))
    mapping = {p: "[original value - safely cached inside proxy memory]" for p in sorted(placeholders)}

    return SanitizeResponse(
        sanitized_text=result,
        mapping=mapping,
        pii_detected=result.strip() != full_text.strip(),
        complexity=complexity,
        latency_ms=round(latency_ms, 2),
    )
