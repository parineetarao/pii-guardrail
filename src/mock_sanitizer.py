def sanitize(text: str) -> str:
    replacements = {
        "Rahul Mehta": "PERSON_001",
        "rahul@gmail.com": "EMAIL_001",
        "+91 98765 43210": "PHONE_001",
    }

    sanitized = text

    for original, placeholder in replacements.items():
        sanitized = sanitized.replace(original, placeholder)

    return sanitized