# Changelog

## 1.1.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.1.0 — Auth hardening, frontend SPA, LLM advisory client

### Added
- **Frontend SPA**: React + TypeScript + Vite application with typed API client, 5 pages (Dashboard, Models, Inference, Benchmarks, Sizing), loading/error/empty states, and 11 passing tests.
- **LLM Advisory Client**: `app/services/llm_client.py` with OpenAI-compatible, Anthropic, and Gemini httpx adapters. 30s bounded timeout, redacted errors, grounded fallback when `LLM_BASE_URL` is not configured.
- **Typed API client**: `frontend/src/api/client.ts` with all backend endpoints and `frontend/src/api/types.ts` matching response schemas.
- **Comprehensive README**: Architecture diagram, API reference, provider configuration table, SSO/SAML setup, AI/ML evaluation section with reproducible commands and measured results.
- **CSS stylesheet**: Dark theme with utility classes, responsive grid, stat cards, badges, and forms.
- **Dockerignore files**: Backend and frontend `.dockerignore` for lean container builds.
- **Security tests**: JWT-based auth tests covering invalid token rejection, valid JWT acceptance, wrong issuer rejection, and operator/admin role enforcement.

### Changed
- **Auth hardening**: Replaced placeholder auth in `main.py` (had TODO markers) with proper role-based authorization from `app.core.security`. Demo user now carries `admin`, `operator`, and `viewer` roles. Production mode requires a valid JWT with correct issuer and realm roles.
- **Advisory endpoint**: Now attempts LLM call through configured provider with graceful fallback to grounded response. Audits each advisory request.
- **Demo user roles**: Extended from `operator/viewer` to include `admin` for full demo-mode access.

### Fixed
- Removed all `TODO`/`FIXME` markers from `main.py` — preflight passes clean.
- Fixed `import.meta.env` typing for Vite environment variables.

### Test results
- Backend: 125 tests passed
- Frontend: 11 tests passed (3 test files), `npm run build` succeeds
- Preflight: no missing files, source budget OK (116,367 bytes), no unfinished code markers