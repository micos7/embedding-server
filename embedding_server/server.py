import logging
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import time
from timeit import default_timer
import torch
import numpy as np
import logging
import PyPDF2
import tracemalloc
import requests
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "distiluse-base-multilingual-cased-v2"
AUTH = "Bearer YOUR_KEY"

def initialize_model():
  model = SentenceTransformer(MODEL_NAME, device='cuda')
  return model

model = initialize_model()

app = FastAPI()

class EmbeddingBody(BaseModel):
  input: str | list[str] = Field(description="Your text string goes here")
  model: str | None = Field(default=None, max_length=500)

class PDFBody(BaseModel):
   url: str | list[str] = Field(description="Your text string goes here")

class PDFEmbeddingResponse(BaseModel):
    data: List[dict]
    model: str

@app.post("/v1/embeddings")
async def root(body: EmbeddingBody, Authorization: Optional[str] = Header(None)):

  if AUTH != Authorization:
    raise HTTPException(status_code=401, detail="Authorization header required")

  start = default_timer()

  embeddings = model.encode(body.input, device='cuda', normalize_embeddings=True,show_progress_bar=True)

  elapsed = default_timer() - start

#   logger.info("%s took %f", body.input, elapsed)

  return {
    "data": {
      "embedding": embeddings.tolist(),
      "index": 0,
      "object": "embedding"
    } ,
    "model": MODEL_NAME
  }

from typing import Optional, List
from fastapi import Header, HTTPException



@app.post("/v1/pdf")
async def pdf_embeddings(body: PDFBody, Authorization: Optional[str] = Header(None)):

    if AUTH != Authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    start = default_timer()
    tracemalloc.start()

    # Fetch PDF content from the URL
    pdf_content = requests.get(body.url).content

    with io.BytesIO(pdf_content) as file:
        pdf_reader = PyPDF2.PdfReader(file)
        embedding_responses = []

        for page_number, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            sentences = [sentence.strip() for sentence in page_text.split('.')]

            chunked_sentences = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= 600:  # +1 for the period
                    if current_chunk:
                        current_chunk += ' ' + sentence + '.'
                    else:
                        current_chunk = sentence + '.'
                else:
                    chunked_sentences.append(current_chunk)
                    current_chunk = sentence + '.'

            if current_chunk:
                chunked_sentences.append(current_chunk)

            embeddings = [model.encode(chunk, device='cuda', normalize_embeddings=True) for chunk in chunked_sentences]

            for i, (embedding, sentence) in enumerate(zip(embeddings, chunked_sentences)):
                embedding_responses.append({
                    "embedding": embedding.tolist(),
                    "text": sentence,
                    "index": i,
                    "object": "embedding",
                    "page": page_number  # Page numbers are usually 1-based
                })

    elapsed = default_timer() - start
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()

    return PDFEmbeddingResponse(data=embedding_responses, model=MODEL_NAME)



