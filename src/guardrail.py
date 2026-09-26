from typing import Callable

from src.router_inference import classify_query
from src.mapping import build_mapping, validate_mapping, restore_pii
from src.llm_client import generate_response


def run_guardrail(
    query: str,
    sanitize_fn: Callable[[str], str],
) -> str:
    """
    Run the complete privacy guardrail pipeline.

    Flow:
        Query
        -> Complexity Router
        -> PII Sanitizer
        -> Mapping
        -> Mapping Validation
        -> External LLM
        -> PII Restoration
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # 1. Classify query complexity
    complexity = classify_query(query)

    print(f"Complexity: {complexity}")

    # 2. Sanitize PII using the Qwen gatekeeper
    sanitized_query = sanitize_fn(query)

    print(f"Sanitized query: {sanitized_query}")

    # 3. Build mapping between placeholders and original PII
    mapping = build_mapping(
        original=query,
        sanitized=sanitized_query,
    )

    print(f"PII mapping: {mapping}")

    # 4. Validate that restoring the placeholders
    #    produces the exact original query
    if not validate_mapping(
        original=query,
        sanitized=sanitized_query,
        mapping=mapping,
    ):
        raise ValueError(
            "PII mapping validation failed. "
            "Request blocked for safety."
        )

    # 5. Send ONLY sanitized text to the external LLM
    llm_response = generate_response(sanitized_query)

    # 6. Restore original PII locally
    final_response = restore_pii(
        llm_response,
        mapping,
    )

    return final_response