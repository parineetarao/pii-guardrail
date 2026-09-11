---
title: PII Guardrail
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# PII Guardrail for RAG Systems

Fine-tuned Qwen2.5-1.5B that pseudonymizes PII and sensitive credentials before they reach external LLM APIs.

Detects: PERSON, EMAIL, PHONE, AADHAAR, PAN, CREDIT_CARD, API_KEY, DB_CREDENTIAL
