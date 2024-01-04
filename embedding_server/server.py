import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import time
from timeit import default_timer
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "all-MiniLM-L6-v2"
AUTH = "Bearer YOUR_KEY"

def initialize_model():
  model = SentenceTransformer(MODEL_NAME, device='cuda')
  return model

model = initialize_model()

app = FastAPI()

class EmbeddingBody(BaseModel):
  input: str | list[str] = Field(description="Your text string goes here")
  model: str | None = Field(default=None, max_length=500)


@app.post("/v1/embeddings")
async def root(body: EmbeddingBody, Authorization: Optional[str] = Header(None)):

  if AUTH != Authorization:
    raise HTTPException(status_code=401, detail="Authorization header required")

  start = default_timer()

  embeddings = model.encode(body.input, device='cuda', normalize_embeddings=True,show_progress_bar=True)

  elapsed = default_timer() - start

  logger.info("%s took %f", body.input, elapsed)

  return {
    "data": {
      "embedding": embeddings.tolist(),
      "index": 0,
      "object": "embedding"
    } ,
    "model": MODEL_NAME
  }
