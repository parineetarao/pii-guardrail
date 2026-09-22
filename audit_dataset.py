import json
import re
import os
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "dataset.json"

CLEAN_OUTPUT = "clean_dataset.json"
FLAGGED_OUTPUT = "flagged_dataset.json"
REPORT_OUTPUT = "audit_report.json"

# These are the ONLY valid placeholder categories.
ALLOWED_CATEGORIES = [
    "PERSON",
    "EMAIL",
    "PHONE",
    "AADHAAR",
    "PAN",
    "API_KEY",
    "CREDIT_CARD",
    "DB_CREDENTIAL"
]

# Match ONLY valid placeholders.
VALID_PLACEHOLDER_PATTERN = re.compile(
    r'\b(?:' +
    '|'.join(map(re.escape, ALLOWED_CATEGORIES)) +
    r')_\d+\b'
)

# Match anything that looks like CATEGORY_123.
GENERIC_PLACEHOLDER_PATTERN = re.compile(
    r'\b[A-Z][A-Z0-9_]*_\d+\b'
)

INSTRUCTION_LEAKAGE_KEYWORDS = [
    "replace with",
    "substitute with",
    "mask as",
    "placeholder:",
    "should be masked",
    "replace the name",
    "anonymize"
]

# Existing placeholders in SOURCE are suspicious.
PREMASKED_PATTERN = re.compile(
    r'\b(?:' +
    '|'.join(map(re.escape, ALLOWED_CATEGORIES)) +
    r')_\d+\b'
)

# Credential assignment patterns.
CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r'\b(?:'
    r'db_pass|db_user|db_password|'
    r'password|passwd|secret_key'
    r')\s*[:=]\s*'
    r'([^\s,;"\'\']+)',
    re.IGNORECASE
)


# ============================================================
# PLACEHOLDER VALIDATION
# ============================================================

def is_valid_placeholder(token):
    """
    Checks whether a placeholder exactly matches:

        PERSON_001
        EMAIL_001
        PHONE_001
        AADHAAR_001
        PAN_001
        API_KEY_001
        CREDIT_CARD_001
        DB_CREDENTIAL_001
    """

    for category in ALLOWED_CATEGORIES:
        pattern = rf'^{re.escape(category)}_\d+$'

        if re.fullmatch(pattern, token):
            return True

    return False


def extract_placeholders(text):
    """
    Extract only tokens that look like placeholders.
    """

    return GENERIC_PLACEHOLDER_PATTERN.findall(text)


# ============================================================
# SINGLE RECORD AUDIT
# ============================================================

def audit_record(record):

    issues = []

    if not isinstance(record, dict):
        return ["INVALID_RECORD_STRUCTURE_NOT_A_DICT"]

    src = record.get("source_text", "")
    msk = record.get("masked_text", "")

    if not isinstance(src, str) or not isinstance(msk, str):
        return ["SOURCE_OR_MASKED_NOT_STRING"]

    src = src.strip()
    msk = msk.strip()

    if not src or not msk:
        return ["EMPTY_SOURCE_OR_MASKED_FIELDS"]

    # --------------------------------------------------------
    # 1. PRE-MASKED TOKENS IN SOURCE
    # --------------------------------------------------------

    source_placeholders = PREMASKED_PATTERN.findall(src)

    if source_placeholders:
        issues.append("PRE_MASKED_TOKENS_IN_SOURCE")

    # --------------------------------------------------------
    # 2. EXPLICIT INSTRUCTION LEAKAGE
    # --------------------------------------------------------

    src_lower = src.lower()

    for keyword in INSTRUCTION_LEAKAGE_KEYWORDS:

        if keyword in src_lower:

            # Only flag if the source actually references
            # a masking/placeholder operation.
            issues.append("EXPLICIT_INSTRUCTION_LEAKAGE")
            break

    # --------------------------------------------------------
    # 3. PLACEHOLDER VALIDATION
    # --------------------------------------------------------

    masked_candidates = extract_placeholders(msk)

    for token in masked_candidates:

        if not is_valid_placeholder(token):

            issues.append(
                f"INVALID_OR_MALFORMED_PLACEHOLDER_SYNTAX ({token})"
            )

    # --------------------------------------------------------
    # 4. UNMASKED CREDENTIAL VALUE LEAKAGE
    # --------------------------------------------------------

    assignments = CREDENTIAL_ASSIGNMENT_PATTERN.findall(src)

    for value in assignments:

        value = value.strip()

        if len(value) <= 3:
            continue

        # If the actual credential value remains in masked text,
        # it was not masked.
        if value in msk:

            issues.append(
                "UNMASKED_CREDENTIAL_VALUE_LEAKAGE"
            )

            break

    # --------------------------------------------------------
    # 5. SAME SOURCE / MASKED TEXT
    # --------------------------------------------------------

    if src == msk:

        # Only flag if the source actually contains apparent PII
        # placeholder-like or credential information.
        if (
            PREMASKED_PATTERN.search(src)
            or CREDENTIAL_ASSIGNMENT_PATTERN.search(src)
        ):
            issues.append(
                "AMBIGUOUS_UNCHANGED_SENSITIVE_RECORD"
            )

    # --------------------------------------------------------
    # 6. EXTREME LENGTH CHANGE
    # --------------------------------------------------------

    # Masking naturally makes text shorter, so allow a large margin.
    # Only flag extreme deletion/rewrite.
    if len(src) > 500 and len(msk) < len(src) * 0.40:

        issues.append(
            "CRITICAL_STRUCTURAL_LENGTH_MISMATCH_OR_REWRITE"
        )

    # --------------------------------------------------------
    # 7. PLACEHOLDER DUPLICATION CONSISTENCY
    # --------------------------------------------------------

    # We don't automatically flag different categories/numbers.
    # Repeated placeholder values are acceptable.

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    return sorted(set(issues))


# ============================================================
# DATA LOADING
# ============================================================

def load_records():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_content = f.read().strip()

    decoder = json.JSONDecoder()

    pos = 0
    content_length = len(raw_content)

    raw_entries = []

    while pos < content_length:

        while pos < content_length and raw_content[pos].isspace():
            pos += 1

        if pos >= content_length:
            break

        try:

            result, next_pos = decoder.raw_decode(
                raw_content,
                pos
            )

            if isinstance(result, list):
                raw_entries.extend(result)
            else:
                raw_entries.append(result)

            pos = next_pos

        except json.JSONDecodeError:

            pos += 1

    all_records = []

    for entry in raw_entries:

        if not entry:
            continue

        if isinstance(entry, dict):

            data_list = entry.get("data", entry)

        else:

            data_list = entry

        if isinstance(data_list, list):

            for item in data_list:

                if isinstance(item, dict):

                    if "json" in item:
                        item = item["json"]

                    if (
                        "source_text" in item
                        and "masked_text" in item
                    ):
                        all_records.append(item)

        elif isinstance(data_list, dict):

            if (
                "source_text" in data_list
                and "masked_text" in data_list
            ):
                all_records.append(data_list)

    return all_records


# ============================================================
# MAIN AUDIT
# ============================================================

if __name__ == "__main__":

    print("🔍 ========================================================")
    print("🔎 RUNNING DATASET QUALITY AUDITOR")
    print("🔍 ========================================================\n")

    if not os.path.exists(INPUT_FILE):

        print(
            f"❌ Cannot locate '{INPUT_FILE}'"
        )

        exit()

    try:

        all_records = load_records()

        print(
            f"📋 Loaded {len(all_records)} records."
        )

        clean_records = []
        flagged_records = []

        issue_counter = Counter()

        # ----------------------------------------------------
        # AUDIT EVERY RECORD
        # ----------------------------------------------------

        for idx, record in enumerate(all_records):

            issues = audit_record(record)

            if issues:

                issue_counter.update(issues)

                flagged_records.append({
                    "record_index": idx + 1,
                    "issues": issues,
                    "data": record
                })

            else:

                clean_records.append(record)

        # ----------------------------------------------------
        # SAVE CLEAN DATA
        # ----------------------------------------------------

        with open(
            CLEAN_OUTPUT,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                clean_records,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # SAVE FLAGGED DATA
        # ----------------------------------------------------

        with open(
            FLAGGED_OUTPUT,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                flagged_records,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # SAVE REPORT
        # ----------------------------------------------------

        report = {

            "summary": {

                "total_records_evaluated":
                    len(all_records),

                "clean_records_count":
                    len(clean_records),

                "flagged_records_count":
                    len(flagged_records),

                "flagged_percentage":
                    round(
                        (
                            len(flagged_records)
                            / len(all_records)
                        ) * 100,
                        2
                    )
                    if all_records
                    else 0

            },

            "issue_distribution_counts":
                dict(issue_counter)

        }

        with open(
            REPORT_OUTPUT,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print("\n📊 ========================================================")
        print("📈 AUDIT SUMMARY")
        print("📊 ========================================================")

        print(
            f"✨ Total records : {len(all_records)}"
        )

        print(
            f"✅ Clean        : {len(clean_records)}"
        )

        print(
            f"⚠️ Flagged      : {len(flagged_records)}"
        )

        print(
            f"📊 Flagged %    : "
            f"{(len(flagged_records) / len(all_records) * 100):.2f}%"
            if all_records
            else "📊 Flagged %    : 0%"
        )

        print("\n🔍 ISSUE DISTRIBUTION:")

        if issue_counter:

            for issue, count in issue_counter.most_common():

                print(
                    f"   ↳ {issue}: {count}"
                )

        else:

            print(
                "   ✨ No issues detected."
            )

        print(
            "\n============================================================"
        )

        print(
            f"\n📁 Clean dataset  → {CLEAN_OUTPUT}"
        )

        print(
            f"📁 Flagged data   → {FLAGGED_OUTPUT}"
        )

        print(
            f"📁 Audit report   → {REPORT_OUTPUT}"
        )

    except Exception as e:

        print(
            f"❌ Audit failed: {e}"
        )