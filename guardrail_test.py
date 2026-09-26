from src.guardrail import run_guardrail
from src.mock_sanitizer import sanitize


query = (
    "Ask Rahul Mehta to send the report "
    "to rahul@gmail.com. You can contact him at "
    "+91 98765 43210."
)

response = run_guardrail(
    query=query,
    sanitize_fn=sanitize,
)

print("\nFinal response:")
print(response)