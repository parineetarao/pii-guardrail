"""
Dataset builder for Stage 1: PII Detection

Maps the ai4privacy dataset labels to project-specific PII categories
and combines them with hand-written examples for categories/formats
that require additional coverage.

Final processed dataset:
    1,233 examples
    986 training
    247 test

The dataset is intentionally not perfectly balanced. High-frequency
categories such as PHONE and EMAIL contain more examples, while
India-specific and credential-related categories are supplemented
with hand-written examples.
"""

# ── Final PII category list ──────────────────────────────────────────
# Scoped to categories where leakage creates immediate, concrete risk.
# Excludes lower-risk fields such as city, zipcode, building number, etc.

PII_CATEGORIES = [
    "PERSON",        # Real person names (context-dependent)
    "EMAIL",         # Email addresses
    "PHONE",         # Phone numbers, any supported format
    "AADHAAR",       # 12-digit Indian government ID
    "PAN",           # Indian tax ID: format ABCDE1234F
    "CREDIT_CARD",   # Credit/debit card numbers
    "API_KEY",       # Service authentication keys and tokens
    "DB_CREDENTIAL", # Database connection strings and credentials
]


# ── Label mapping: ai4privacy → project categories ──────────────────
# Only labels that map to our defined categories are retained.
# Everything else is skipped during dataset construction.

LABEL_MAP = {
    "USERNAME":         "PERSON",
    "EMAIL":            "EMAIL",
    "TELEPHONENUM":     "PHONE",
    "CREDITCARDNUMBER": "CREDIT_CARD",
}


# ── Labels explicitly excluded from ai4privacy ───────────────────────
#
# Examples:
# DATEOFBIRTH, STREET, CITY, ZIPCODE, BUILDINGNUM,
# COUNTY, STATE, COUNTRY, AGE, SEX, and other labels.
#
# Reason:
# These fields are outside the project's current PII risk scope.


# ── Gap categories ───────────────────────────────────────────────────
#
# The following categories are not directly covered by the selected
# ai4privacy labels and therefore receive hand-written examples:
#
#   AADHAAR
#   PAN
#   API_KEY
#   DB_CREDENTIAL
#
# Hand-written examples also provide additional contextual coverage
# for PERSON, PHONE, RAG scenarios, and clean/no-PII inputs.


# ── Final dataset composition ────────────────────────────────────────
#
# The final processed dataset currently contains 1,233 examples.
#
# Category distribution:
#
#   PHONE          400
#   EMAIL          346
#   PERSON         158
#   AADHAAR        107
#   PAN             98
#   API_KEY         93
#   CREDIT_CARD     89
#   DB_CREDENTIAL   75
#   CLEAN          120
#
# The categories are intentionally not forced to exactly equal sizes.
# Existing examples are retained rather than artificially duplicating
# or oversampling categories simply to achieve numerical balance.


# ── Dataset split ────────────────────────────────────────────────────
#
# Final split:
#
#   Training: 986 examples
#   Test:     247 examples
#
# This corresponds to an approximately 80/20 train/test split.
#
# The test set is held out during fine-tuning and is used for evaluating
# PII detection performance using precision, recall, and F1.


# ── Dataset sources ──────────────────────────────────────────────────
#
# Public dataset:
#   ai4privacy/pii-masking-400k
#
# Hand-written data:
#   - India-specific PII examples
#   - Credential/API-key examples
#   - PERSON contextual examples
#   - RAG/context examples
#   - Clean/no-PII examples
#
# Important:
# The public ai4privacy data is not treated as India-specific.
# Indian formats such as Aadhaar and PAN are therefore covered using
# dedicated hand-written examples.