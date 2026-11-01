# BGE-M3 encoding

FastAPI wrapper around [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), using
FlagEmbedding for dense, sparse, and ColBERT representations.

From this directory:

```bash
uv sync --locked
uv run python main.py
```

Default URL: http://127.0.0.1:9105. Override with `--host` and `--port`.
The model loads during startup and remains resident in a spawned worker. First
startup downloads weights using the normal Hugging Face cache. Allow sufficient
VRAM alongside the other services. One request runs at a time; concurrent calls
receive 503 with Retry-After. There is no authentication; use a trusted network.

Settings use the `BGE_M3_` prefix and a local `.env`; see `.env.example`.
Default device is `cuda:0` with FP16. Set `BGE_M3_DEVICE=cpu` for explicit CPU
execution with FP32. CUDA failures do not silently fall back to CPU.

```bash
curl http://127.0.0.1:9105/v1/encode \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Hello world","Selamat pagi"],"return_sparse":true,"return_colbert_vecs":true}'
```

`texts` is always a list. Defaults return dense only. For sparse only, set
`return_dense=false, return_sparse=true`; for ColBERT only, set
`return_dense=false, return_colbert_vecs=true`. Any combination is supported;
at least one must be enabled.

Response fields are `model` and requested `dense_vecs`, `lexical_weights`, and
`colbert_vecs`. Each outer list follows input order. Dense vectors have 1024
components. Sparse maps use token ID strings and floating-point weights, not
human-readable words. ColBERT matrices contain variable numbers of token vectors.
Embeddings use upstream normalization; no query instruction is added.

`max_length` defaults to 512 tokens and accepts 2–8192; longer text is truncated
by the tokenizer. Defaults limit a request to 16 texts, 32,768 characters per
text, and 131,072 total characters. The internal inference batch size is 4.
Large ColBERT responses may exceed the 64 MiB result limit (413). Invalid input
returns 422, a busy/unavailable worker or GPU OOM returns 503, inference timeout
returns 504, and unexpected inference failures return 500. Defaults allow 600
seconds for startup and 120 for inference. Timeout, cancellation, OOM, and worker
crash discard the worker and reload the model. Initial load failure stops startup.
If a later reload fails, readiness remains false until the service is restarted.

`/health/live` checks the API; `/health/ready` checks the loaded worker. Swagger
is at `/docs` in local and staging environments. Requests include X-Request-ID
and timing logs without recording input text.

```bash
uv run pytest -q
uv run ruff check src tests main.py
```

For real inference, launch the service and run `test_apis.py` from the parent
folder with the other services. Its BGE-M3 check runs two batches with all outputs.
GPU and CPU execution must be validated on the deployment host.

## Original resources

- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — upstream library providing `BGEM3FlagModel`.
- [BAAI/bge-m3 model card](https://huggingface.co/BAAI/bge-m3) — model weights, usage examples, and model documentation.
