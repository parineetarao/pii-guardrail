import pickle
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "router_pipeline.pkl"
)


with open(MODEL_PATH, "rb") as f:
    router = pickle.load(f)


def classify_query(query: str) -> str:
    """
    Classify a user query as SIMPLE or COMPLEX.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    prediction = router.predict([query])[0]

    return prediction.upper()