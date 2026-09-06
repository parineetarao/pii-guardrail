"""
Dataset builder for Stage 1: PII Detection
Maps ai4privacy dataset labels to project-specific categories.
Hand-written examples cover gaps not in public dataset.
"""

# ── Final PII category list ──────────────────────────────────────────
# Scoped to categories where leakage creates immediate, concrete risk.
# Excludes low-sensitivity fields like city, zipcode, building number.

PII_CATEGORIES = [
    "PERSON",        # Real person names (context-dependent, hardest for regex)
    "EMAIL",         # Email addresses
    "PHONE",         # Phone numbers, any format
    "AADHAAR",       # 12-digit Indian government ID
    "PAN",           # Indian tax ID: format ABCDE1234F
    "CREDIT_CARD",   # 16-digit card numbers
    "API_KEY",       # Service authentication keys and tokens
    "DB_CREDENTIAL", # Database connection strings and passwords
]

# ── Label mapping: ai4privacy → our categories ───────────────────────
# Only keep labels that map to our defined categories.
# Everything not in this map gets skipped during dataset construction.

LABEL_MAP = {
    "USERNAME":         "PERSON",       # synthetic names in dataset
    "EMAIL":            "EMAIL",
    "TELEPHONENUM":     "PHONE",
    "CREDITCARDNUMBER": "CREDIT_CARD",
}

# Labels we explicitly skip from ai4privacy:
# DATEOFBIRTH, STREET, CITY, ZIPCODE, BUILDINGNUM,
# COUNTY, STATE, COUNTRY, AGE, SEX, and others
# Reason: not in scope for this project's risk model

# ── Gap categories not covered by ai4privacy ─────────────────────────
# These must be entirely hand-written:
# AADHAAR, PAN, API_KEY, DB_CREDENTIAL

# ── Dataset targets ──────────────────────────────────────────────────
TARGET_FROM_PUBLIC  = 350  # examples pulled from ai4privacy
TARGET_API_KEYS     = 50   # hand-written API key + DB credential examples
TARGET_RAG          = 100  # hand-written RAG-scenario examples
TARGET_CLEAN        = 150  # clean examples with no PII (null behavior)
# Total: ~650 examples, 80/20 train/test split