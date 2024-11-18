from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import json
import os
import logging

# Load environment variables
load_dotenv()

# Load configuration from environment variables
try:
    config = json.loads(os.getenv("CONFIG_JSON", "{}"))
    if not config:
        raise ValueError("CONFIG_JSON is missing or malformed in the .env file.")
    port = int(os.getenv("PORT", 8080))
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        raise FileNotFoundError("Service account key file not found or not specified in GOOGLE_APPLICATION_CREDENTIALS.")
except Exception as e:
    raise RuntimeError(f"Error loading configuration: {str(e)}")

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
)
logger = logging.getLogger(__name__)

logger.info("Configuration loaded successfully.")

# Initialize Vertex AI with service account credentials
try:
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    vertexai.init(project=config["project_id"], location=config["location"], credentials=credentials)
    logger.info("Vertex AI initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Vertex AI: {str(e)}")
    raise

# Define TaskType Enum dynamically
TaskType = Enum("TaskType", {task: task for task in config["supported_task_types"]})
logger.info("TaskType Enum defined successfully.")

# FastAPI app
app = FastAPI()

# Schema definitions
class QueryData(BaseModel):
    question: str
    answers: List[str]
    correct_answer: str
    question_task_type: TaskType  # type: ignore
    answer_task_type: TaskType  # type: ignore

class Payload(BaseModel):
    data: List[QueryData]

# Helper functions
def get_embeddings(texts: list[str], task_type: str):
    try:
        model = TextEmbeddingModel.from_pretrained(config["model_name"])
        inputs = [TextEmbeddingInput(text, task_type) for text in texts]
        embeddings = model.get_embeddings(inputs)
        return [emb.values for emb in embeddings]
    except Exception as e:
        logger.error(f"Error fetching embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching embeddings: {str(e)}")

def get_top100_similar_answers(similarities):
    return sorted(range(len(similarities)), key=lambda i: -similarities[i])

def calculate_mrr(query_ranks):
    reciprocal_ranks = [1 / (i + 1) for ranks in query_ranks for i, rank in enumerate(ranks) if rank == 1]
    return sum(reciprocal_ranks) / len(query_ranks)

# Endpoint
@app.post("/process_data/")
async def process_data(payload: Payload):
    logger.info(f"Processing data with {len(payload.data)} queries.")
    output, all_query_ranks = [], []
    try:
        for item in payload.data:
            logger.info(f"Processing question: {item.question}")
            question_embedding = get_embeddings([item.question], item.question_task_type.value)[0]
            answer_embeddings = get_embeddings(item.answers, item.answer_task_type.value)
            similarities = cosine_similarity([question_embedding], answer_embeddings)[0]
            ranked_indices = get_top100_similar_answers(similarities)
            ranked_answers = [item.answers[i] for i in ranked_indices]
            query_ranks = [1 if item.answers[i] == item.correct_answer else 0 for i in ranked_indices]
            all_query_ranks.append(query_ranks)
            output.append({
                "question": item.question,
                "answers": item.answers,
                "cosine_similarities": similarities.tolist(),
                "ranked_answers": ranked_answers,
                "query_ranks": query_ranks
            })
        mrr = calculate_mrr(all_query_ranks)
        return {"results": output, "mean_reciprocal_rank": mrr}
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

# Run the app dynamically using the port from the configuration
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
