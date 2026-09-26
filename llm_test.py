from src.llm_client import generate_response


response = generate_response(
    "Explain what a confusion matrix is in simple terms."
)

print(response)