import asyncio
import hmac
import io
import ipaddress
import logging
import os
import re
import socket
import time
import tracemalloc
from contextlib import asynccontextmanager
from typing import List, Optional
from urllib.parse import urlparse

import httpx
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")

# Expected value of the Authorization header ("Bearer <token>").
# Set EMBEDDING_SERVER_TOKEN in the environment; falls back to the legacy key.
AUTH = f"Bearer {os.getenv('EMBEDDING_SERVER_TOKEN', 'YOUR_KEY')}"

MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", 100 * 1024 * 1024))  # 100 MB
MAX_PAGES = int(os.getenv("MAX_PAGES", 2000))

# tracemalloc adds significant overhead; enable only when debugging memory.
TRACE_MEMORY = os.getenv("TRACE_MEMORY", "").lower() in ("1", "true", "yes")


def tracemalloc_start():
    if TRACE_MEMORY:
        tracemalloc.start()
        return True
    return None


def get_device() -> str:
    # Allow forcing a device via env (e.g. EMBEDDING_DEVICE=cpu when the
    # installed torch build has no kernels for the available GPU).
    forced = os.getenv("EMBEDDING_DEVICE", "").lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

# The device the model is actually running on (may fall back from DEVICE).
model_device = DEVICE

model: Optional[SentenceTransformer] = None


def _load_model(device: str) -> SentenceTransformer:
    m = SentenceTransformer(MODEL_NAME, device=device)
    # Warmup so the first real request doesn't pay lazy-init cost.
    m.encode(["warmup"], device=device, normalize_embeddings=True, show_progress_bar=False)
    return m


def initialize_model() -> SentenceTransformer:
    global model_device
    try:
        return _load_model(DEVICE)
    except Exception as e:
        if DEVICE != "cpu":
            logger.error(f"Error initializing model on {DEVICE}: {e} — falling back to CPU")
            model_device = "cpu"
            try:
                return _load_model("cpu")
            except Exception as cpu_e:
                logger.error(f"Error initializing model on CPU: {cpu_e}")
                raise
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = initialize_model()
    logger.info(f"Model {MODEL_NAME} loaded on {model_device}")
    yield


app = FastAPI(lifespan=lifespan)


def require_auth(Authorization: Optional[str]) -> None:
    # Constant-time comparison avoids leaking the token via timing.
    if Authorization is None or not hmac.compare_digest(AUTH, Authorization):
        raise HTTPException(status_code=401, detail="Authorization header required")


class EmbeddingBody(BaseModel):
    input: str | list[str] = Field(description="Your text string goes here")
    model: str | None = Field(default=None, max_length=500)


class PDFBody(BaseModel):
    url: str | list[str] = Field(description="URL of the PDF to embed")
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)


class PDFEmbeddingResponse(BaseModel):
    data: List[dict]
    model: str


@app.get("/health")
async def health():
    return {"status": "ok", "device": model_device, "model": MODEL_NAME}


async def _encode(input_, *, normalize: bool = True):
    # Inference is CPU/GPU-bound: run it in a worker thread so the
    # event loop stays responsive while embeddings are generated.
    return await asyncio.to_thread(
        model.encode,
        input_,
        device=model_device,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )


@app.post("/v1/embeddings")
async def root(body: EmbeddingBody, Authorization: Optional[str] = Header(None)):
    require_auth(Authorization)

    start = time.perf_counter()

    try:
        embeddings = await _encode(body.input)
    except Exception as e:
        logger.error(f"Error encoding input: {e}")
        raise HTTPException(status_code=500, detail="Error encoding input")

    logger.info(f"Embedding generated in {time.perf_counter() - start:.3f}s")

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


def _resolve_host_ips(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return [info[4][0] for info in infos]


def _is_public_hostname(url: str) -> bool:
    """SSRF guard: only allow http(s) URLs resolving to public addresses."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False

    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal")):
        return False

    try:
        ips = [ipaddress.ip_address(ip) for ip in _resolve_host_ips(host)]
    except OSError:
        return False

    if not ips:
        return False

    return all(ip.is_global for ip in ips)


async def _fetch_pdf(url: str) -> bytes:
    loop = asyncio.get_running_loop()

    if not await loop.run_in_executor(None, _is_public_hostname, url):
        raise HTTPException(status_code=400, detail="URL must point to a public host")

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            # Streamed download with a hard cap so a huge file can't exhaust memory.
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if content_type and "pdf" not in content_type.lower():
                    raise HTTPException(status_code=400, detail="URL does not serve a PDF")

                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > MAX_PDF_BYTES:
                        raise HTTPException(status_code=413, detail="PDF exceeds maximum allowed size")
                return bytes(buffer)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching PDF content from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching PDF content: {str(e)}")


def chunk_text(text: str, max_chars: int = 600, overlap_chars: int = 120) -> List[str]:
    if not text:
        return []

    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    # Split by common sentence terminators followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            # Step with overlap for massive sentences
            step = max(1, max_chars - overlap_chars)
            for i in range(0, len(sentence), step):
                piece = sentence[i:i + max_chars]
                if piece:
                    chunks.append(piece)
                if i + max_chars >= len(sentence):
                    break
            continue

        if current_len + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Carry over trailing sentences that fit within overlap_chars
                overlap_chunk: List[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    added_len = len(prev_sent) + (1 if overlap_chunk else 0)
                    if overlap_len + added_len <= overlap_chars and overlap_len + added_len + 1 + len(sentence) <= max_chars:
                        overlap_chunk.insert(0, prev_sent)
                        overlap_len += added_len
                    else:
                        break
                current_chunk = overlap_chunk + [sentence]
                current_len = len(" ".join(current_chunk))
            else:
                current_chunk = [sentence]
                current_len = len(sentence)
        else:
            current_chunk.append(sentence)
            current_len += len(sentence) + (1 if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@app.post("/v1/pdf")
async def pdf_embeddings(body: PDFBody, Authorization: Optional[str] = Header(None)):
    require_auth(Authorization)

    start = time.perf_counter()
    trace_ctx = tracemalloc_start()

    url = body.url if isinstance(body.url, str) else body.url[0]

    pdf_content = await _fetch_pdf(url)

    embedding_responses = []
    all_chunks = []
    chunk_metadata = []

    def extract_chunks() -> tuple[list, list]:
        extracted_chunks = []
        metadata = []

        with io.BytesIO(pdf_content) as file:
            try:
                pdf_reader = PdfReader(file)
            except Exception as e:
                logger.error(f"Error reading PDF: {e}")
                raise HTTPException(status_code=500, detail="Error reading PDF")

            pages = pdf_reader.pages[:MAX_PAGES]
            if len(pdf_reader.pages) > MAX_PAGES:
                logger.warning(f"PDF has more than {MAX_PAGES} pages; truncating")

            for page_number, page in enumerate(pages):
                try:
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_number}: {e}")
                    continue

                for i, chunk in enumerate(chunk_text(page_text, max_chars=body.chunk_size, overlap_chars=body.chunk_overlap)):
                    extracted_chunks.append(chunk)
                    metadata.append({
                        "text": chunk,
                        "index": i,
                        "page": page_number
                    })

        return extracted_chunks, metadata

    # pypdf parsing is CPU-bound too; keep it off the event loop.
    all_chunks, chunk_metadata = await asyncio.to_thread(extract_chunks)

    if all_chunks:
        try:
            # Batch encode all chunks for performance
            embeddings = await _encode(all_chunks)

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

    elapsed = time.perf_counter() - start
    message = f"PDF processed in {elapsed:.3f}s ({len(all_chunks)} chunks)"
    if trace_ctx is not None:
        peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()
        message += f". Memory peaked at {peak:.2f} MB"
    logger.info(message)

    return PDFEmbeddingResponse(data=embedding_responses, model=MODEL_NAME)
