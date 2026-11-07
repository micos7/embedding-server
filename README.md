# Embedding server

This FastAPI service provides the local embeddings API used by the Laravel app. It listens on port `9002` and loads the `paraphrase-multilingual-mpnet-base-v2` model by default.

## Install

From the Laravel project root, create the virtual environment in the location used by `composer run dev` and install the server dependencies:

```bash
cd embedding-server/embedding_server
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r ../requirements.txt
```

This uses Python's built-in `venv` and `pip`; Poetry is not required. The model files are downloaded the first time the server starts and can take several hundred megabytes.

## Run the server by itself

Run these commands from `embedding-server/embedding_server`:

```bash
printf 'Embedding server token: '; read -rs EMBEDDING_SERVER_TOKEN; printf '\n'
export EMBEDDING_SERVER_TOKEN
./venv/bin/uvicorn server:app --host 127.0.0.1 --port 9002 --workers=1
```

Enter the same token configured as `EMBEDDING_SERVER_TOKEN` in the Laravel app's `.env`. The `read` command keeps the token out of shell history. Leave this terminal open while using chat.

The `server:app` import path assumes the current directory is `embedding-server/embedding_server`. If starting from `embedding-server`, use `embedding_server.server:app` as the import path instead.

## Run the full app in development

The root `composer.json` defines `composer run dev`. It starts Laravel, Vite, the queue listener, and Uvicorn. Run it from the Laravel project root:

```bash
printf 'Embedding server token: '; read -rs EMBEDDING_SERVER_TOKEN; printf '\n'
export EMBEDDING_SERVER_TOKEN
composer run dev
```

The Composer script starts Uvicorn from `embedding-server/embedding_server` using `./venv/bin/uvicorn server:app --port 9002 --workers=1`, so run the installation steps above first. Laravel reads its `.env` itself, but child processes started by Composer need `EMBEDDING_SERVER_TOKEN` exported in the shell too.

## Configuration

The Laravel app defaults to `http://localhost:9002`. Configure `EMBEDDING_SERVER_URL`, `EMBEDDING_SERVER_TOKEN`, and optionally `EMBEDDING_SERVER_MODEL` in the Laravel `.env`; the server token must match the exported server token. The server reads `EMBEDDING_MODEL` to override its default model.

## Troubleshooting

- `Could not import module "server"`: run `uvicorn server:app` from `embedding-server/embedding_server`, or use `embedding_server.server:app` from `embedding-server`.
- `ModuleNotFoundError` (for example, `httpx`): use `./venv/bin/uvicorn`, not the system `uvicorn`, and install dependencies with `./venv/bin/pip install -r ../requirements.txt`.
- Chat reports that the answer could not be completed, or Laravel logs show a connection failure to port `9002`: make sure this server is running and that `EMBEDDING_SERVER_URL` points to it.
- Authentication errors: make sure the server's exported `EMBEDDING_SERVER_TOKEN` matches the Laravel `.env` value.

## API example

```bash
curl http://127.0.0.1:9002/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $EMBEDDING_SERVER_TOKEN" \
  -d '{"input":"Your text goes here"}'
```

The service supports sentence-transformers models. The model cache is stored by the Python runtime under its normal cache directory. To clear the PyTorch cache, run `rm -rf ~/.cache/torch`.
