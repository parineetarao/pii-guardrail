import re


PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:PERSON|EMAIL|PHONE|AADHAAR|PAN|API_KEY|CREDIT_CARD|DB_CREDENTIAL)_\d{3}\b"
)


def add_mapping(mapping: dict, placeholder: str, value: str):
    """
    Add a placeholder -> original PII mapping.

    If the same placeholder appears again, the original value
    must be identical.
    """

    if placeholder in mapping:
        if mapping[placeholder] != value:
            raise ValueError(
                f"Inconsistent mapping for {placeholder}: "
                f"{mapping[placeholder]!r} vs {value!r}"
            )
    else:
        mapping[placeholder] = value


def build_mapping(original: str, sanitized: str) -> dict:
    """
    Recover the original PII values from Qwen's sanitized output.

    Example:

        original:
            Call Rahul Mehta at +91 98765 43210.

        sanitized:
            Call PERSON_001 at PHONE_001.

        returns:
            {
                "PERSON_001": "Rahul Mehta",
                "PHONE_001": "+91 98765 43210"
            }
    """

    matches = list(PLACEHOLDER_PATTERN.finditer(sanitized))

    if not matches:
        return {}

    mapping = {}

    original_pos = 0
    sanitized_pos = 0

    previous_placeholder = None

    for match in matches:

        # Text between the previous placeholder
        # and the current placeholder.
        unchanged_text = sanitized[
            sanitized_pos:match.start()
        ]

        # Find that unchanged text in the original.
        original_index = original.find(
            unchanged_text,
            original_pos
        )

        if original_index == -1:
            raise ValueError(
                "Could not reliably align original and sanitized text."
            )

        # Text removed from the original between the
        # previous anchor and this anchor is the PII
        # belonging to the PREVIOUS placeholder.
        pii = original[
            original_pos:original_index
        ]

        if previous_placeholder is not None:

            if not pii:
                raise ValueError(
                    f"No original value found for "
                    f"{previous_placeholder}"
                )

            add_mapping(
                mapping,
                previous_placeholder,
                pii
            )

        # Move past the unchanged text.
        original_pos = (
            original_index + len(unchanged_text)
        )

        sanitized_pos = match.end()

        # This is now the placeholder whose value
        # will be recovered at the next anchor.
        previous_placeholder = match.group()

    # --------------------------------------------------
    # Handle the text after the final placeholder
    # --------------------------------------------------

    remaining = sanitized[sanitized_pos:]

    # Use rfind() here because the remaining text
    # may occur earlier inside the PII itself.
    if remaining:
        original_index = original.rfind(
            remaining,
            original_pos
        )

        if original_index == -1:
            raise ValueError(
                "Could not reliably align final text."
            )

        pii = original[
            original_pos:original_index
        ]

    else:
        # No text after the final placeholder.
        pii = original[original_pos:]

    if previous_placeholder is not None:

        if not pii:
            raise ValueError(
                f"No original value found for "
                f"{previous_placeholder}"
            )

        add_mapping(
            mapping,
            previous_placeholder,
            pii
        )

    return mapping
def restore_pii(text: str, mapping: dict) -> str:
    """
    Replace placeholders in text with their original PII values.

    The mapping stays local and is never sent to the external LLM.
    """

    def replace(match):
        placeholder = match.group()

        if placeholder not in mapping:
            raise ValueError(
                f"Unknown placeholder returned by LLM: "
                f"{placeholder}"
            )

        return mapping[placeholder]

    return PLACEHOLDER_PATTERN.sub(replace, text)

def validate_mapping(
    original: str,
    sanitized: str,
    mapping: dict
) -> bool:
    """
    Verify that restoring all placeholders in the sanitized
    text reconstructs the original text exactly.
    """

    restored = restore_pii(
        sanitized,
        mapping
    )

    return restored == original