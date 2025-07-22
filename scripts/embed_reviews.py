import argparse
from pathlib import Path
from app.review_loader import load_and_format_reviews
from google.cloud import aiplatform
import os
from tqdm import tqdm

# Set up your embedding model and Vertex Vector Search index info
EMBEDDING_MODEL = "textembedding-gecko@003"  # Or your preferred model
PROJECT = os.getenv("VERTEX_PROJECT", os.getenv("PROJECT_ID"))
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
INDEX_ID = os.getenv("VERTEX_INDEX_ID")  # Set this in your .env or environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed reviews and upload to Vertex Vector Search."
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=".",
        help="Path to the directory containing review files (default: current directory)",
    )
    return parser.parse_args()


def get_embedding(text: str, client):
    # Use Vertex AI Embedding API
    response = client.get_embeddings(model=EMBEDDING_MODEL, content=[text])
    return response.embeddings[0].values


def main():
    args = parse_args()
    reviews_dir = Path(args.dir)
    review_files = sorted(
        [f for f in reviews_dir.iterdir() if f.is_file() and f.name.startswith("reviews-")]
    )
    if not review_files:
        print(f"No files starting with 'reviews-' found in {reviews_dir}")
        return

    aiplatform.init(project=PROJECT, location=LOCATION)
    embedding_client = aiplatform.TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    vector_index = aiplatform.MatchingEngineIndex(index_name=INDEX_ID)

    all_points = []
    idx = 0
    for review_file in review_files:
        reviews = load_and_format_reviews(str(review_file))
        print(f"🧠 Embedding {len(reviews)} reviews from {review_file.name}...")
        for review in tqdm(reviews, desc=review_file.name):
            text = review["comment"]
            embedding = embedding_client.get_embeddings([text])[0]
            # Prepare the upsert record
            point = {
                "id": str(idx),
                "embedding": embedding,
                "metadata": review,  # You can flatten or filter fields as needed
            }
            all_points.append(point)
            idx += 1

    # Upsert to Vertex Vector Search
    print(f"🔼 Upserting {len(all_points)} vectors to Vertex Vector Search index '{INDEX_ID}'...")
    vector_index.upsert_datapoints(datapoints=all_points)
    print(f"✅ Inserted {len(all_points)} reviews into index '{INDEX_ID}'.")


if __name__ == "__main__":
    main()
