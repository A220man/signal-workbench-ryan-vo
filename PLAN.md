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

### continue-implementation — Continue implementation and resolve known failures

Status: in_progress

Acceptance:
- Run checks, fix failures, and implement the next planned feature

