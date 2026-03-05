import logging
import time
import io
import re
import tracemalloc
from typing import Optional, List
from timeit import default_timer

import torch
import numpy as np
import httpx
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from pypdf import PdfReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# MODEL_NAME = "distiluse-base-multilingual-cased-v2"
# MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
AUTH = "Bearer YOUR_KEY"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")

def initialize_model():
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
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
        # handle single string or list of strings
        embeddings = model.encode(body.input, device=DEVICE, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Error encoding input: {e}")
        raise HTTPException(status_code=500, detail="Error encoding input")

    elapsed = default_timer() - start
    logger.info(f"Embedding generated in {elapsed:.3f}s")

    # OpenAI-like response format
    if isinstance(body.input, str):
        data = {
            "embedding": embeddings.tolist(),
            "index": 0,
            "object": "embedding"
        }
    else:
        data = [
            {
                "embedding": emb.tolist(),
                "index": i,
                "object": "embedding"
            } for i, emb in enumerate(embeddings)
        ]

    return {
        "data": data,
        "model": MODEL_NAME,
        "usage": {
            "total_tokens": 0  # We don't have token count here easily
        }
    }

def chunk_text(text: str, max_chars: int = 600) -> List[str]:
    if not text:
        return []
    
    # Split by common sentence terminators followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If a single sentence is longer than max_chars, we might need to split it further
        # but for now we just keep it as is or split it crudely if it's massive
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            # Simple crude split for massive sentences
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i+max_chars])
            continue

        if current_len + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(sentence)
        else:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

@app.post("/v1/pdf")
async def pdf_embeddings(body: PDFBody, Authorization: Optional[str] = Header(None)):
    if AUTH != Authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    start = default_timer()
    tracemalloc.start()

    url = body.url if isinstance(body.url, str) else body.url[0]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            pdf_content = response.content
    except Exception as e:
        logger.error(f"Error fetching PDF content from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching PDF content: {str(e)}")

    embedding_responses = []
    all_chunks = []
    chunk_metadata = []

    with io.BytesIO(pdf_content) as file:
        try:
            pdf_reader = PdfReader(file)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise HTTPException(status_code=500, detail="Error reading PDF")

        for page_number, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if not page_text:
                    continue
            except Exception as e:
                logger.error(f"Error extracting text from page {page_number}: {e}")
                continue

            page_chunks = chunk_text(page_text)
            for i, chunk in enumerate(page_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "text": chunk,
                    "index": i,
                    "page": page_number
                })

    if all_chunks:
        try:
            # Batch encode all chunks for performance
            embeddings = model.encode(all_chunks, device=DEVICE, normalize_embeddings=True, show_progress_bar=False)
            
            for i, emb in enumerate(embeddings):
                meta = chunk_metadata[i]
                embedding_responses.append({
                    "embedding": emb.tolist(),
                    "text": meta["text"],
                    "index": meta["index"],
                    "object": "embedding",
                    "page": meta["page"]
                })
        except Exception as e:
            logger.error(f"Error encoding chunks: {e}")
            raise HTTPException(status_code=500, detail="Error generating embeddings")

    elapsed = default_timer() - start
    mem_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"PDF processed in {elapsed:.3f}s. Memory peaked at {mem_usage[1] / 1024 / 1024:.2f} MB")

    return PDFEmbeddingResponse(data=embedding_responses, model=MODEL_NAME)
