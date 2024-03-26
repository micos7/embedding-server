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
from pypdf import PdfReader
import tracemalloc
import requests
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# MODEL_NAME = "distiluse-base-multilingual-cased-v2"
# MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
AUTH = "Bearer YOUR_KEY"

def initialize_model():
    try:
        model = SentenceTransformer(MODEL_NAME, device='cuda')
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise
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

    try:
        embeddings = model.encode(body.input, device='cuda', normalize_embeddings=True, show_progress_bar=True)
    except Exception as e:
        logger.error(f"Error encoding input: {e}")
        raise HTTPException(status_code=500, detail="Error encoding input")

    elapsed = default_timer() - start

    return {
        "data": {
            "embedding": embeddings.tolist(),
            "index": 0,
            "object": "embedding"
        },
        "model": MODEL_NAME
    }

@app.post("/v1/pdf")
async def pdf_embeddings(body: PDFBody, Authorization: Optional[str] = Header(None)):
    if AUTH != Authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    start = default_timer()
    tracemalloc.start()

    try:
        # Fetch PDF content from the URL
        pdf_content = requests.get(body.url).content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching PDF content: {e}")
        raise HTTPException(status_code=500, detail="Error fetching PDF content")

    with io.BytesIO(pdf_content) as file:
        try:
            pdf_reader = PdfReader(file)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise HTTPException(status_code=500, detail="Error reading PDF")

        embedding_responses = []

        for page_number, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
            except Exception as e:
                logger.error(f"Error extracting text from page {page_number}: {e}")
                continue

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

            try:
                embeddings = [model.encode(chunk, device='cuda', normalize_embeddings=True) for chunk in chunked_sentences]
            except Exception as e:
                logger.error(f"Error encoding text chunks for page {page_number}: {e}")
                continue

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
