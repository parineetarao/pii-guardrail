from src.mapping import build_mapping, restore_pii, validate_mapping


tests = [
    (
        "Call Rahul Mehta at +91 98765 43210.",
        "Call PERSON_001 at PHONE_001.",
        "Please contact PERSON_001 at PHONE_001 regarding the project."
    ),

    (
        "Send the API key sk_test_51AbCdEfGh123456789 to Rahul.",
        "Send the API key API_KEY_001 to PERSON_001.",
        "The API key belongs to PERSON_001."
    ),

    (
        "Use postgres://admin:secret123@db.example.com:5432/app.",
        "Use DB_CREDENTIAL_001.",
        "Connect using DB_CREDENTIAL_001."
    ),

    (
        "Rahul Mehta can be reached at rahul@gmail.com or +91 98765 43210.",
        "PERSON_001 can be reached at EMAIL_001 or PHONE_001.",
        "Contact PERSON_001 using EMAIL_001 or PHONE_001."
    ),
]


for original, sanitized, llm_response in tests:

    print("\n" + "=" * 80)

    print("\nORIGINAL:")
    print(original)

    print("\nSANITIZED:")
    print(sanitized)

    # Build local mapping
    mapping = build_mapping(
        original,
        sanitized
    )

    print("\nMAPPING:")
    print(mapping)

    # Simulate external LLM response
    restored = restore_pii(
        llm_response,
        mapping
    )

    print("\nLLM RESPONSE:")
    print(llm_response)

    print("\nRESTORED RESPONSE:")
    print(restored)

    valid = validate_mapping(
    original,
    sanitized,
    mapping
    )

    print("\nROUND-TRIP VALID:", valid)