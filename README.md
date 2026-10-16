# Signal Workbench | Ryan Vo | AI & Machine Learning

> Work in progress. This checkpoint may be incomplete or fail tests. See PLAN.md and STATUS.md before use.


**Calibrating 1-bit LLMs: a deterministic workbench for BitNet b1.58 activation audits, memory planning, and compiler diagnostics.**

Signal Workbench is a full-stack application for analyzing and calibrating Microsoft BitNet b1.58 1-bit quantized LLMs. It provides a deterministic, offline core (no LLM key required) that simulates the perplexity regression documented in upstream Issue #588 — where the FFN activation was incorrectly `SiLU` instead of squared `ReLU²` — along with hardware memory sizing, compiler diagnostics, and an opt-in LLM advisory layer. The intended user is an ML engineer or researcher working with 1-bit quantized models who needs reproducible, evidence-based tooling to evaluate activation choices and deployment feasibility.

## AI/ML Evaluation

### Perplexity Calibration (Issue #588)

The core engine reproduces the activation bug from [upstream Issue #588](https://github.com/microsoft/BitNet/issues/588), where `LLM_FFN_SILU` was used instead of `LLM_FFN_RELU_SQR` in the BitNet b1.58 FFN sublayer.

**Reproducible command:**

```bash
# Backend tests (includes perplexity evaluation)
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests/test_perplexity_eval.py backend/tests/test_ffn_activations.py -v

# Interactive benchmark via API
curl -X POST http://127.0.0.1:8000/api/benchmarks \
  -H "Content-Type: application/json" \
  -d '{"model_name": "BitNet-b1.58-2B-4T", "dataset": "wikitext-2", "num_tokens": 512, "activation_tested": "relu_sqr"}'
```

**Data provenance:** The synthetic perplexity simulation (`evaluate_synthetic_perplexity`) is grounded in clean-room benchmarks from Issue #588, conducted on three Azure VMs (Intel x86, AMD x86, ARM Ampere) using `llama-perplexity` over wikitext-2-raw. The simulation uses a fixed seed (42) for reproducibility and incorporates the measured noise bands:

| Activation | Mean Loss | Perplexity | Reference (Issue #588) |
|---|---|---|---|
| ReLU² (official) | 2.8396 | ~17.1 | 17.1090 ± 0.129 |
| SiLU (bug) | 4.6033 | ~99.8 | 99.8178 ± 0.906 |

**Measured results:** In test mode, `run_benchmark` produces perplexity values within 16.0–18.5 for ReLU² and 90.0–110.0 for SiLU, matching the empirical bands above.

**Failure cases:**
- The simulation is deterministic and does not run an actual quantized model — it reproduces the *numerical effect* of the activation bug on cross-entropy loss, not the model weights themselves.
- Different seeds produce minor variance within the reported bands. Results outside these ranges indicate a regression.
- LLM advisory responses are not authoritative; verify against official documentation.

### Ternary Quantization Engine

The absmean weight quantization and 2-bit i2_s packing are tested against known BitNet b1.58 parameters (~45% zero, ~27.5% each for −1 and +1) and verified for round-trip correctness.

## Features

### 1. Model Registry & In-Memory Sessions
- Register and load 1-bit BitNet models into system RAM for persistent inference (Issue #206).
- Models stay resident without reloading; SQLite records persist independently.
- Unload (evict from RAM) while keeping the database record.
- Audit log of all model lifecycle events.

### 2. Inference Against Warm Sessions
- Execute token inference against loaded in-memory sessions.
- Configurable activation override (ReLU² vs SiLU) for A/B comparison.
- Records prompt/completion, token counts, latency, and tokens/sec.

### 3. Perplexity Calibration Benchmarks
- Run perplexity evaluations comparing ReLU² (correct) vs SiLU (regression) activations.
- Results stored in SQLite with weight distribution metadata.
- Historical benchmark history with pagination.

### 4. Hardware Memory Planner
- Estimate FP16 vs 1.58-bit ternary weight memory for 0.6B to 405B parameter scales.
- KV-cache sizing accounting for context length.
- Throughput estimation bounded by memory bandwidth.
- Fit analysis against device memory.

### 5. Compiler & Hardware Diagnostics
- CPU SIMD feature detection (AVX-512, AVX2, NEON, SSE4.2).
- Clang/CMake toolchain detection with version strings.
- Known-issue resolution matrix for Issues #180, #334, #185, #480.

### 6. Opt-in LLM Advisory
- Server-side advisory via OpenAI-compatible, Anthropic, or Gemini providers.
- Grounded in actual issue findings; never fabricates responses.
- Graceful fallback to grounded response when no LLM endpoint is configured.
- Bounded 30s timeout; redacted provider errors.

## API Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | None | Health check, version, demo mode status |
| POST | `/api/models` | operator | Load a model into RAM |
| GET | `/api/models` | user | List all registered models |
| GET | `/api/models/{id}` | user | Get model details |
| DELETE | `/api/models/{id}` | operator | Evict model from RAM |
| POST | `/api/inference` | operator | Run inference against a warm session |
| POST | `/api/benchmarks` | operator | Run a perplexity benchmark |
| GET | `/api/benchmarks` | user | List recent benchmark runs |
| POST | `/api/sizing` | user | Calculate memory/throughput estimates |
| GET | `/api/compiler` | user | Run CPU/toolchain diagnostics |
| POST | `/api/advisory` | operator | Opt-in LLM advisory (requires LLM_API_KEY) |
| GET | `/api/audit` | admin | Paginated audit log |

## Architecture

```mermaid
graph TB
  subgraph "Client"
    FE[React SPA<br/>(Vite, port 3000)]
  end

  subgraph "Backend"
    BE[FastAPI<br/>(port 8000)]
    MS[Model Service<br/>+ In-Memory Sessions]
    BS[Benchmark Service]
    LLM[LLM Advisory Service<br/>(opt-in)]
    DB[(SQLite<br/>workbench.db)]
    AUDIT[Audit Log]
  end

  subgraph "Auth"
    KC[Keycloak<br/>(port 8080)]
  end

  FE -->|REST API| BE
  BE --> MS
  BE --> BS
  BE --> LLM
  MS --> DB
  BS --> DB
  LLM --> DB
  MS --> AUDIT
  BS --> AUDIT
  BE -->|OIDC Discovery| KC
```

## Installation & Running

```bash
# Backend
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests

# Frontend
cd frontend
npm install
npm test          # vitest
npm run build     # tsc + vite build

# Run with Docker Compose (includes Keycloak, demo mode)
docker compose up --build
# Frontend:   http://127.0.0.1:3000
# Backend:    http://127.0.0.1:8000
# Keycloak:   http://127.0.0.1:8080 (admin: admin_password_demo_only)
```

## Provider Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai-compatible` | `openai-compatible`, `anthropic`, `gemini`, or `ollama` |
| `LLM_MODEL` | `meta-llama/llama-3.1-8b-instruct` | Model identifier for the selected provider |
| `LLM_BASE_URL` | *(not set)* | Endpoint URL for OpenAI-compatible providers. When unset, LLM advisory returns a grounded response only. |
| `LLM_API_KEY` | *(not set)* | Server-side LLM credential. Required for `/api/advisory`. Never exposed to the client. |
| `DEMO_MODE` | `true` | Local demo mode (localhost only). **Refuses to start in production.** |
| `OIDC_ISSUER` | `http://127.0.0.1:8080/realms/workbench-realm` | Expected JWT issuer claim |
| `OIDC_CLIENT_ID` | `workbench-client` | Keycloak client ID for resource role extraction |

## Security

- **Demo mode**: Binds to localhost, auto-authenticates as a demo user with full roles. Startup **refuses** demo mode when `ENVIRONMENT=production`.
- **Production mode**: Requires a valid Bearer JWT with matching issuer and appropriate realm roles.
- **JWT signature validation** is not yet implemented (planned: JWKS-based signature verification). Issuer and expiry claims are checked.
- **Roles**: `viewer` (read), `operator` (read + write), `admin` (read + write + audit log).
- **Tokens** are never logged. LLM API keys are server-side only.
- **CORS** is restricted to `http://127.0.0.1:3000` and `http://localhost:3000`.

## SSO / SAML Setup

Keycloak is provisioned via `keycloak-realm.json` and imported on startup through Docker Compose. The realm (`workbench-realm`) defines three roles: `admin`, `operator`, `viewer`.

To add a SAML identity provider:

1. Configure Keycloak → Realm Settings → Identity Providers → Add SAML provider.
2. Set the IdP Entity ID and Single Sign-On Service URL.
3. Upload the IdP's X.509 certificate.
4. Configure attribute mappings for `username`, `email`, and roles.
5. Enable "Sync On Login" to provision users on first SAML assertion.

The realm JSON includes a disabled SAML IdP stub (`enterprise-saml-idp`) that can be enabled and configured with real endpoint URLs.

## .env.example

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

## Limitations

- No GPU acceleration; all computation is CPU-based numpy.
- The perplexity simulation reproduces the *numerical effect* of the activation bug, not actual model inference on real weights.
- JWT signature verification is pending (issuer + expiry only).
- The SPA OIDC PKCE login flow is a planned feature; the frontend currently relies on demo-mode auto-auth.

## License

MIT © Ryan Vo <ryandtvo@gmail.com>