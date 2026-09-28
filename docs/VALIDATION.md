# Validation status

## Completed in the build workspace

- `python -m compileall -q .` passed.
- `python -m unittest discover -s tests -v` passed.
- Result: **52 tests / OK**.
- Secret scan did not find a real-looking `sk-...` key or non-empty `LLM_API_KEY` in project source/config.

## What the 52 tests include

- Existing V1 routing / policy / runtime / guard / state / HTTP regression tests.
- Redis Conversation Memory serialization, recent-window trimming and TTL refresh semantics.
- Redis CaseState JSON round-trip and TTL semantics.
- Redis Trace round-trip, recent index and max-item trimming semantics.
- Deterministic vector embedding reproducibility / normalization.
- Vector retrieval relevance / threshold / verified-only indexing.
- Chroma adapter synchronization and query behavior through a deterministic fake client.

## Environment limitation

The artifact build environment cannot access PyPI and does not have Docker, so it could not install `redis` / `chromadb` or launch real Redis + Chroma-backed containers here. Therefore **real external-service integration still needs one Docker smoke test on your machine**.

Run:

```powershell
docker compose down
docker compose build --no-cache
docker compose up
```

Then verify `/health` reports `redis / redis / chroma`, run all tests inside Docker, and send one resource query plus one restart-persistence test as described in `docs/V1_1_UPGRADE.md`.
