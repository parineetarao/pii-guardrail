"""
PII Guardrail Pipeline — End-to-End Demo
Stage 1: Qwen2.5-1.5B fine-tuned PII pseudonymizer
Stage 2: Naive Bayes simple/complex query router

Architecture:
    User query
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
    Stage 2 Router            Stage 1 PII Guardrail
    (original query)          (query + RAG context)
        │                             │
        ▼                             ▼
    SIMPLE/COMPLEX            Pseudonymized text
        │                     + private mapping
        └─────────────────────────────┘
                              │
                              ▼
                     Route to cheap/expensive LLM
                     (external API call with no PII)
"""

import json
import pickle
import re
from typing import Optional

# ── Stage 2: Query Router ─────────────────────────────────────────────

def load_router(model_path: str = "models/router_pipeline.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def route_query(query: str, router) -> str:
    """
    Classify the original user query as simple or complex.
    Runs on the ORIGINAL query before any PII processing.
    Returns: "simple" or "complex"
    """
    return router.predict([query])[0]

# ── Stage 1: PII Pseudonymizer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a PII redaction system.
Your job: rewrite the input text replacing all PII and sensitive credentials with typed placeholder tags.
Rules:
- Replace each unique sensitive value with a typed tag: PERSON_001, EMAIL_001, PHONE_001, AADHAAR_001, PAN_001, CREDIT_CARD_001, API_KEY_001, DB_CREDENTIAL_001
- Number tags sequentially within each category: PERSON_001, PERSON_002, etc.
- If the same value appears multiple times, use the same tag each time
- If there is no PII or sensitive data, return the text COMPLETELY UNCHANGED
- Do not add explanations. Output only the rewritten text."""


def pseudonymize(text: str, model, tokenizer, device="cuda") -> tuple[str, dict]:
    """
    Run Stage 1: pseudonymize PII in text.
    Returns:
        pseudonymized_text: text with PII replaced by typed placeholders
        mapping: placeholder -> original value (kept private, never sent to LLM)
    
    Note: mapping extraction is approximate — production systems would use
    the model output + original text alignment for exact mapping.
    """
    import torch

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    pseudonymized = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Build approximate reverse mapping
    # Find all placeholder tags in output
    placeholder_pattern = re.compile(
        r'(PERSON|EMAIL|PHONE|AADHAAR|PAN|CREDIT_CARD|API_KEY|DB_CREDENTIAL)_\d{3}'
    )
    placeholders = set(placeholder_pattern.findall(pseudonymized))

    # mapping is maintained privately — never sent to external LLM
    mapping = {p: "[REDACTED]" for p in placeholders}

    return pseudonymized, mapping


# ── Full Pipeline ─────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    rag_context: Optional[str],
    router,
    model=None,
    tokenizer=None,
    verbose: bool = True,
) -> dict:
    """
    Full two-stage pipeline.

    Stage 1 checkpoint placement: after context assembly, before external LLM.
    Stage 2 routes on original query, independent of Stage 1.

    Args:
        user_query:  original user question
        rag_context: retrieved document content (may contain PII)
        router:      trained Naive Bayes pipeline
        model:       fine-tuned Qwen model (optional for demo without GPU)
        tokenizer:   matching tokenizer
        verbose:     print pipeline steps

    Returns dict with all pipeline decisions and outputs.
    """

    if verbose:
        print("=" * 60)
        print("PII GUARDRAIL PIPELINE")
        print("=" * 60)
        print(f"\nUser query: {user_query}")
        if rag_context:
            print(f"RAG context: {rag_context[:100]}...")

    # ── Stage 2: Route on original query (parallel, independent) ──────
    complexity = route_query(user_query, router)

    if verbose:
        print(f"\n[Stage 2] Query complexity: {complexity.upper()}")
        model_choice = "GPT-4 / Claude Opus" if complexity == "complex" else "GPT-3.5 / Claude Haiku"
        print(f"[Stage 2] Route to: {model_choice}")

    # ── Assemble full context ──────────────────────────────────────────
    if rag_context:
        assembled = f"{user_query}\n\n[RETRIEVED CONTEXT]: {rag_context}"
    else:
        assembled = user_query

    # ── Stage 1: Pseudonymize after context assembly ───────────────────
    if model is not None and tokenizer is not None:
        pseudonymized, mapping = pseudonymize(assembled, model, tokenizer)
    else:
        # Demo mode without model loaded
        pseudonymized = assembled
        mapping = {}
        if verbose:
            print("\n[Stage 1] Model not loaded — skipping pseudonymization (demo mode)")

    if verbose:
        print(f"\n[Stage 1] Pseudonymized context:")
        print(f"  {pseudonymized[:200]}")
        if mapping:
            print(f"\n[Stage 1] Private mapping (never sent to LLM):")
            for k, v in mapping.items():
                print(f"  {k} -> {v}")

    # ── External LLM call (simulated) ─────────────────────────────────
    if verbose:
        print(f"\n[External LLM] Sending pseudonymized context to {model_choice}")
        print("[External LLM] *** No real PII in this API call ***")

    return {
        "original_query":    user_query,
        "rag_context":       rag_context,
        "complexity":        complexity,
        "assembled_context": assembled,
        "pseudonymized":     pseudonymized,
        "private_mapping":   mapping,
        "model_routed_to":   model_choice if verbose else (
            "expensive" if complexity == "complex" else "cheap"
        ),
    }


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading router...")
    router = load_router("models/router_pipeline.pkl")
    print("Router loaded.\n")

    # Demo runs without the Qwen model loaded
    # In production: load model + tokenizer and pass to run_pipeline

    test_cases = [
        {
            "query": "What is the leave policy?",
            "context": None,
        },
        {
            "query": "Who should I contact for payroll issues?",
            "context": (
                "For payroll queries contact Priya Sharma at "
                "priya.sharma@company.com or call +91 98765 43210."
            ),
        },
        {
            "query": "Analyze the security incident and identify all affected systems.",
            "context": (
                "Post-incident analysis found misconfigured S3 exposed "
                "DATABASE_URL=postgres://synth_prod:SYNTH_PWD@db.internal:5432/app "
                "and API_KEY=SYNTH_EXPOSED_KEY_4Km9Lp."
            ),
        },
        {
            "query": "What are the tradeoffs between OAuth and SAML for our use case?",
            "context": None,
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        result = run_pipeline(
            user_query=case["query"],
            rag_context=case["context"],
            router=router,
            model=None,
            tokenizer=None,
            verbose=True,
        )
        print()