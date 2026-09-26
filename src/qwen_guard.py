import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"

ADAPTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "qwen-pii-guardrail-adapter"
)

MAX_NEW_TOKENS = 128


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a privacy-preserving PII pseudonymization model.

Your task is to identify personally identifiable information (PII)
and replace it with typed placeholders.

Rules:
1. Replace PII only. Do not rewrite or rephrase the sentence.
2. Use these categories:
   PERSON, EMAIL, PHONE, AADHAAR, PAN,
   API_KEY, CREDIT_CARD, DB_CREDENTIAL.
3. Use sequential numbering within each category:
   PERSON_001, PERSON_002, EMAIL_001, etc.
4. If the same PII value appears multiple times, use the same placeholder.
5. Preserve all non-PII text exactly.
6. If there is no PII, return the input unchanged.
7. Return ONLY the pseudonymized text.
8. Do not provide explanations.
"""


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Qwen PII Guard...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

print("Qwen PII Guard loaded.")


# ============================================================
# PLACEHOLDER NORMALIZATION
# ============================================================

PLACEHOLDER_PATTERN = re.compile(
    r"\b("
    r"PERSON|"
    r"EMAIL|"
    r"PHONE|"
    r"AADHAAR|"
    r"PAN|"
    r"API_KEY|"
    r"CREDIT_CARD|"
    r"DB_CREDENTIAL"
    r")(?:_(\d{3}))?\b"
)


def normalize_placeholders(text: str) -> str:
    """
    Ensure every PII placeholder has the required numeric suffix.

    Examples:
        API_KEY       -> API_KEY_001
        PERSON        -> PERSON_001
        PERSON_001    -> PERSON_001

    Qwen remains responsible for detecting and categorizing PII.
    This function only normalizes the placeholder format.
    """

    counters = {}

    def replace(match):
        category = match.group(1)
        number = match.group(2)

        # Qwen already produced a numbered placeholder.
        if number is not None:
            counters[category] = max(
                counters.get(category, 0),
                int(number)
            )

            return match.group(0)

        # Qwen omitted the number.
        counters[category] = counters.get(category, 0) + 1

        return f"{category}_{counters[category]:03d}"

    return PLACEHOLDER_PATTERN.sub(replace, text)


# ============================================================
# SANITIZATION
# ============================================================

def sanitize(text: str) -> str:
    """
    Detect and pseudonymize PII in the supplied text.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Put inputs on the same device as the model's input embeddings.
    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Remove the original prompt tokens.
    generated_tokens = outputs[
        :,
        inputs["input_ids"].shape[1]:
    ]

    sanitized_text = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    ).strip()

    # Normalize placeholders after Qwen generation.
    sanitized_text = normalize_placeholders(
        sanitized_text
    )

    return sanitized_text