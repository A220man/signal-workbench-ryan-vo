# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

future-jwt-signature-validation

## Roadmap

### future-jwt-signature-validation — Full JWT signature validation via JWKS discovery

Status: done

Acceptance:
- Validate token signature via Keycloak JWKS endpoint
- Check aud (audience) claim
- Reject expired tokens
- JWKS cached with 1-hour TTL
- Falls back to cached JWKS on fetch failure
- All 125 backend tests pass
- All 17 frontend tests pass

Implementation: `backend/app/core/security.py`

Tests: `backend/tests/test_security.py`

### spa-oidc-pkce-login — SPA OIDC PKCE login flow with bearer token management

Status: planned

Acceptance:
- Authorization code + PKCE flow via Keycloak
- Access token stored in memory only, never localStorage
- State + nonce validation on callback
- React Router integration with protected routes
- Auto-redirect to login when unauthenticated in production mode
- All 17 frontend tests pass (4 test files)
- npm run build succeeds

Implementation: `frontend/src/auth/oidc.ts`, `frontend/src/auth/AuthContext.tsx`, `frontend/src/auth/LoginPage.tsx`, `frontend/src/auth/CallbackPage.tsx`, `frontend/src/App.tsx`, `frontend/src/api/client.ts`

Tests: `frontend/src/tests/OIDCClient.test.ts`, `frontend/src/tests/ApiClient.test.ts`, `frontend/src/tests/Dashboard.test.tsx`, `frontend/src/tests/Models.test.tsx`

