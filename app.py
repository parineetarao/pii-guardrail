import gradio as gr
import torch
import re
import spaces
import json

SYSTEM_PROMPT = """You are a PII redaction system.
Your job: rewrite the input text replacing all PII and sensitive credentials with typed placeholder tags.
Rules:
- Replace each unique sensitive value with a typed tag: PERSON_001, EMAIL_001, PHONE_001, AADHAAR_001, PAN_001, CREDIT_CARD_001, API_KEY_001, DB_CREDENTIAL_001
- Number tags sequentially within each category: PERSON_001, PERSON_002, etc.
- If the same value appears multiple times, use the same tag each time
- If there is no PII or sensitive data, return the text COMPLETELY UNCHANGED
- Do not add explanations. Output only the rewritten text."""

MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "parineeta8/pii-guardrail-adapter"

tokenizer = None
model     = None


def load_model():
    global tokenizer, model
    if model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()


@spaces.GPU(duration=120)
def sanitize(user_query, rag_context):
    if not user_query.strip():
        return "Please enter some text.", ""
    load_model()
    full_text = (
        f"{user_query}\n\n[RETRIEVED CONTEXT]: {rag_context}"
        if rag_context.strip() else user_query
    )
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{full_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pattern = re.compile(
        r'(PERSON|EMAIL|PHONE|AADHAAR|PAN|CREDIT_CARD|API_KEY|DB_CREDENTIAL)_(\d{3})'
    )
    placeholders = set(f"{c}_{n}" for c, n in pattern.findall(result))
    mapping = {p: "[ original value — stored privately ]" for p in sorted(placeholders)}
    mapping_str = json.dumps(mapping, indent=2) if mapping else "{  (no PII detected)  }"
    return result, mapping_str


examples = [
    ["Who should I contact for payroll issues?",
     "For payroll queries contact Priya Sharma at priya.sharma@company.com or call +91 98765 43210."],
    ["Was any credential exposed in the incident?",
     "Post-incident analysis found misconfigured S3 exposed DATABASE_URL=postgres://admin:PASS123@db.internal:5432/prod and API_KEY=SYNTH_EXPOSED_KEY_9Km."],
    ["Verify the customer KYC details.",
     "Customer Amit Kumar, Aadhaar 3456 7890 1234, PAN AMTKM5678K, phone 9823456710, email amit.kumar@gmail.com."],
    ["What is the company work from home policy?", ""],
    ["Who approved the Q3 budget?",
     "The Q3 budget was approved by Vikram Nair (vikram.nair@company.com, PAN VKRNR1234F) on 15 July."],
]

demo = gr.Interface(
    fn=sanitize,
    inputs=[
        gr.Textbox(label="User Query", placeholder="e.g. Who should I contact for payroll issues?", lines=3),
        gr.Textbox(label="RAG Retrieved Context (optional)", placeholder="Paste retrieved document content here...", lines=5),
    ],
    outputs=[
        gr.Textbox(label="Sanitized Output (safe to send to external LLM)", lines=6),
        gr.Textbox(label="Private Mapping (stays in your system)", lines=6),
    ],
    title="🛡️ PII Guardrail for RAG Systems",
    description="Fine-tuned Qwen2.5-1.5B that pseudonymizes PII before it reaches external LLM APIs. Detects: PERSON · EMAIL · PHONE · AADHAAR · PAN · CREDIT_CARD · API_KEY · DB_CREDENTIAL\n\n⏱️ First request takes ~30 seconds to load the model.",
    examples=examples,
    cache_examples=False,
)

demo.launch()