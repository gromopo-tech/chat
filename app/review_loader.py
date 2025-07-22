import json
from typing import List, Dict, Any


def load_reviews_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads reviews from a JSON file. Each review should be a dict with at least:
      - comment: str
      - starRating: str (e.g., "FIVE")
      - createTime: str (ISO8601)
      - reviewer: { displayName: str }
    Returns a list of review dicts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If the file is a dict with a top-level key, extract the list
    if isinstance(data, dict):
        # Try common keys
        for key in ("reviews", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Fallback: treat as list of dicts
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unexpected JSON structure for reviews.")


def format_review_for_vertex(review: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a review dict for ingestion into Vertex Vector Search.
    Ensures required fields and flattens nested reviewer.displayName.
    """
    formatted = {
        "comment": review.get("comment", ""),
        "starRating": review.get("starRating", ""),
        "createTime": review.get("createTime", ""),
        "reviewer_displayName": review.get("reviewer", {}).get("displayName", "")
    }
    # Optionally add more fields or transformations here
    return formatted


def load_and_format_reviews(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads and formats all reviews from a JSON file for ingestion.
    """
    reviews = load_reviews_from_json(json_path)
    return [format_review_for_vertex(r) for r in reviews]
