# Review based RAG LLM FastAPI App

This project is a Retrieval-Augmented Generation (RAG) API using FastAPI, Qdrant (vector database), and Google Vertex AI for embeddings and LLM. \
It supports both local development (with Docker Compose) and production (with Qdrant Cloud).

**Note:** This project was developed and tested with Python 3.13.5. Other Python 3.13.x versions should work, but earlier versions may not be compatible.

---

## 🚀 Getting Started (Local Development)

### 1. **Clone the Repository**
```sh
git clone https://github.com/gromopo-tech/chat.git
cd chat
```

### 2. **Set Up Google Application Default Credentials**
- Make sure you have Vertex AI API enabled and run:
  ```sh
  gcloud auth application-default login
  ```
- This creates the ADC file at `~/.config/gcloud/application_default_credentials.json`.

### 3. **Set Environment Variables**
```sh
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json
export VERTEX_LOCATION=<gcp-region-associated-with-vertex-ai-api-credentials>
export VERTEX_PROJECT=<gcp-project-id-associated-with-vertex-ai-api-credentials>
```

### 4. **Start Qdrant and FastAPI with Docker Compose**
This will start a persistent Qdrant instance for local development.

```sh
docker-compose up -d
```

- Qdrant will be available at [http://localhost:6333](http://localhost:6333)
- The API will be available at [http://localhost:8080](http://localhost:8080)
- Interactive docs: [http://localhost:8080/docs](http://localhost:8080/docs)

### 5. **Install Python Dependencies (for local runs)**
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. **Embed Reviews into Qdrant**
This script will embed the sample reviews and upload them to your Qdrant instance.

```sh
python -m scripts.embed_reviews --dir reviews/
```

- You should see output like:
  ```
  🧠 Embedding 14 reviews one by one...
  ✅ Inserted 14 reviews into collection 'reviews'.
  ```

### 8. **Test the API**
Send a POST request to the RAG endpoint:
```sh
curl -X POST "http://localhost:8080/rag/streaming-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What do people like about the Duck?"}'
```

You should see the response streaming.
---

## 🐳 Using Docker Compose for App and Qdrant

You can run both Qdrant and the FastAPI app with Docker Compose:

```sh
docker-compose up -d
```

- The app will be available at [http://localhost:8080](http://localhost:8080)
- Qdrant will be at [http://localhost:6333](http://localhost:6333)

You can check the status of the containers with:

```sh
docker-compose ps
```

and you can check the logs of both containers with:

```sh
docker-compose logs -f
```
---

## 🛠️ Troubleshooting

- **Qdrant connection errors:**
  - Make sure Qdrant is running (`docker-compose up -d qdrant`).
  - Check that your app is using the correct host/port or URL/API key.
- **Vertex AI authentication errors:**
  - Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set and points to your ADC file (locally).
  - On Cloud Run, use the default service account or set up Workload Identity.
- **No context returned:**
  - Make sure you ran the embedding script after Qdrant was started.
  # No need to specify `place_id` in your query; all reviews are for the same place.

---

## 🧹 Cleaning Up

To stop and remove all containers:
```sh
docker-compose down
```
To remove all data (including Qdrant data):
```sh
docker-compose down -v
```

---

## 📦 Production

- Use Qdrant Cloud by setting `QDRANT_URL` and `QDRANT_API_KEY`.
- Run the embedding script from a cloud VM for large datasets for better speed and reliability.
- On Cloud Run, `GOOGLE_CLOUD_PROJECT` is set automatically; set `VERTEX_LOCATION` as an env var.

---

## 📄 License
MIT
