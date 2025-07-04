import json
from pathlib import Path
from qdrant_client.models import PointStruct
from tqdm import tqdm
from qdrant_client.models import VectorParams, Distance
from app.models import embeddings_model
from app.vectorstore import get_qdrant, COLLECTION_NAME

REVIEWS_PATH = Path("sample_reviews.json")
VECTOR_SIZE = 3072


def main():
    with open(REVIEWS_PATH, "r") as f:
        reviews = json.load(f)["reviews"]

    print(f"🧠 Embedding {len(reviews)} reviews one by one...")

    points = []
    for i, review in enumerate(tqdm(reviews)):
        text = review["text"]["text"]
        embedding = embeddings_model.embed_query(text)
        place_id = review["name"].split("/")[1]
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "place_id": place_id,
                    "rating": review["rating"],
                    "publishTime": review.get("publishTime"),
                    "author": review.get("authorAttribution", {}).get(
                        "displayName", "Unknown"
                    ),
                },
            )
        )

    get_qdrant().recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    get_qdrant().upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Inserted {len(points)} reviews into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
