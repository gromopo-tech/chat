import argparse
import json
from app.config import Config
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.cloud.aiplatform.matching_engine import MatchingEngineIndex

TASK = "QUESTION_ANSWERING"
BATCH_SIZE = 100

def load_reviews(dirpath: Path) -> List[Dict]:
    chunks = []
    for file in tqdm(dirpath.glob("reviews*.json")):
        for review in tqdm(json.load(open(file)).get("reviews", [])):
            if "comment" in review:
                rid = review.get("name", "") or review.get("id", "")
                chunks.append({
                    "id": rid.split("/")[-1],
                    "text": review["comment"],
                    "metadata": {
                        "starRating": review.get("starRating", ""),
                        "reviewer": review.get("reviewer", {}).get("displayName", ""),
                        "createTime": review.get("createTime", ""),
                    }
                })
    return chunks

def embed_hybrid(texts: List[str]) -> List[Dict]:
    """
    Generates dense and sparse embeddings for a list of texts using the Vertex AI SDK.
    NOTE: Using the deprecated aiplatform.TextEmbeddingModel because the recommended
    google.genai library does not currently support sparse embeddings from Vertex.
    TODO: Migrate to google.genai when the library is fixed.
    """
    model = TextEmbeddingModel.from_pretrained(Config.EMBEDDING_MODEL)
    response = model.get_embeddings(
        texts,
        output_dimensionality=Config.VECTOR_DIMENSIONS,
    )

    results = []
    for embedding in response:
        results.append({
            "dense": embedding.values,
            "sparse": {
                "values": embedding.sparse_values,
                "dimensions": embedding.sparse_indices
            } if hasattr(embedding, 'sparse_values') and embedding.sparse_values else None
        })
    return results

def main(args):
    aiplatform.init(project=args.project, location=args.location)
    index = MatchingEngineIndex(index_name=args.index_name)
    print("Loading Reviews...")
    chunks = load_reviews(Path(args.dir))
    print(f"Processing {len(chunks)} reviews...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeds = embed_hybrid(texts)

        datapoints = []
        for item, embed in zip(batch, embeds):
            # Convert metadata dict to list of Restriction objects for filtering
            restricts = []
            for key, value in item["metadata"].items():
                if value:  # Only add if value is not empty
                    restricts.append(
                        {"namespace": key, "allow_list": [str(value)]}
                    )

            dp = {
                "datapoint_id": item["id"],
                "feature_vector": embed["dense"],
                # only include sparse if available
                **({"sparse_embedding": embed["sparse"]} if embed["sparse"] else {}),
                "restricts": restricts,
            }
            datapoints.append(dp)

        print(f"Upserting {len(datapoints)} points...")
        index.upsert_datapoints(datapoints=datapoints)

    print("✅ Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory with reviews*.json files")
    parser.add_argument("--project", default=Config.PROJECT, help="GCP project ID")
    parser.add_argument("--location", default=Config.LOCATION, help="GCP region")
    parser.add_argument("--index-name", default=Config.INDEX_NAME, help="Full index resource name")
    args = parser.parse_args()
    main(args)
