# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

backend-auth-security-hardening

## Roadmap

### backend-auth-security-hardening — Replace placeholder auth with proper OIDC role-based authorization

Status: done

Acceptance:
- All endpoints use core/security.py get_current_user and require_role
- Demo mode returns admin+operator+viewer roles for full access
- Production mode requires valid JWT with correct issuer and role
- No TODO/FIXME markers remain in main.py
- Preflight passes with no unfinished code
- 125 backend tests pass

Implementation: `backend/app/main.py`, `backend/app/core/security.py`, `backend/app/services/llm_client.py`

Tests: `backend/tests/test_security.py`

### frontend-spa — Complete frontend SPA with typed API client and core workflows

Status: done

Acceptance:
- React + TypeScript + Vite app builds and passes typecheck
- Typed API client derived from backend routes
- Dashboard, Models, Inference, Benchmarks, Sizing pages
- Loading, error, and empty states on all pages
- 11 frontend tests pass (3 test files)
- npm run build succeeds

Implementation: `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/api/client.ts`, `frontend/src/api/types.ts`, `frontend/src/components/Layout.tsx`, `frontend/src/components/LoadingState.tsx`, `frontend/src/components/ErrorState.tsx`, `frontend/src/components/EmptyState.tsx`, `frontend/src/pages/DashboardPage.tsx`, `frontend/src/pages/ModelsPage.tsx`, `frontend/src/pages/InferencePage.tsx`, `frontend/src/pages/BenchmarksPage.tsx`, `frontend/src/pages/SizingPage.tsx`, `frontend/src/styles.css`, `frontend/src/setupTests.ts`, `frontend/src/vite-env.d.ts`

Tests: `frontend/src/tests/ApiClient.test.ts`, `frontend/src/tests/Dashboard.test.tsx`, `frontend/src/tests/Models.test.tsx`

### llm-advisory-implement — Implement real LLM advisory client with provider adapters and fallback

Status: done

Acceptance:
- OpenAI-compatible, Anthropic, and Gemini httpx adapters
- Bounded 30s timeout with redacted errors
- Falls back to grounded response when LLM_BASE_URL not configured
- Audits advisory requests
- No real network calls in CI tests

Implementation: `backend/app/services/llm_client.py`

Tests: `backend/tests/test_api.py::TestAdvisoryEndpoint`

### ci-frontend — Frontend CI pipeline passes (build + tests)

Status: done

Acceptance:
- npm ci, npm test, npm run build all succeed
- Docker frontend build produces production image

Implementation: `frontend/src/`, `.github/workflows/ci.yml`

Tests: `frontend/src/tests/`

### future-oidc-pkce-client — SPA OIDC PKCE login flow with bearer token management

Status: planned

Acceptance:
- Authorization code + PKCE flow via Keycloak
- Access token stored in memory only, never localStorage
- State + nonce validation on callback

### future-jwt-signature-validation — Full JWT signature validation via JWKS discovery

Status: planned

Acceptance:
- Validate token signature via Keycloak JWKS endpoint
- Check aud (audience) claim
- Reject expired tokens

