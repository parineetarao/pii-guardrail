from src.router_inference import classify_query


queries = [
    "What is MFA?",
    "Who is the HR contact?",
    "What is the company's refund policy?",
    "Compare Rahul and Priya's performance over the last two quarters.",
    "Analyze the security incident and identify all affected systems.",
    "How should we restructure the data pipeline for DPDP compliance?",
]


for query in queries:

    result = classify_query(query)

    print(f"[{result}] {query}")