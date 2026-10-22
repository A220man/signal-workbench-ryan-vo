/** OIDC Authorization Code flow with PKCE for SPA.
 * Access tokens kept in memory only; verifier/state stored in sessionStorage.
 */

const OIDC_DISCOVERY_URL = import.meta.env?.VITE_OIDC_DISCOVERY_URL ?? "";
const OIDC_CLIENT_ID = import.meta.env?.VITE_OIDC_CLIENT_ID ?? "workbench-client";
const OIDC_REDIRECT_URI = `${window.location.origin}/callback`;

interface OIDCDiscovery {
  authorization_endpoint: string;
  token_endpoint: string;
  jwks_uri: string;
  issuer: string;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  id_token?: string;
  scope?: string;
}

let _discovery: OIDCDiscovery | null = null;
let _accessToken: string | null = null;
let _tokenExpiry: number = 0;
let _refreshToken: string | null = null;

/** Fetch OIDC discovery document. */
export async function fetchDiscovery(): Promise<OIDCDiscovery> {
  if (_discovery) return _discovery;
  if (!OIDC_DISCOVERY_URL) {
    throw new Error("VITE_OIDC_DISCOVERY_URL not configured");
  }
  const resp = await fetch(OIDC_DISCOVERY_URL);
  if (!resp.ok) {
    throw new Error(`Failed to fetch OIDC discovery: ${resp.status}`);
  }
  _discovery = await resp.json();
  if (!_discovery) {
    throw new Error("OIDC discovery document is empty");
  }
  return _discovery;
}

/** Generate PKCE pair and store verifier in sessionStorage. */
export async function generateAndStorePKCE(): Promise<{ codeChallenge: string; codeChallengeMethod: string }> {
  const verifier = generateCodeVerifier();
  const challenge = await generateCodeChallenge(verifier);
  sessionStorage.setItem("pkce_verifier", verifier);
  sessionStorage.setItem("pkce_state", generateState());
  return {
    codeChallenge: challenge,
    codeChallengeMethod: "S256",
  };
}

/** Generate cryptographically random code_verifier (43-128 chars). */
export function generateCodeVerifier(): string {
  const array = new Uint8Array(64);
  crypto.getRandomValues(array);
  return base64URLEncode(array);
}

/** Generate code_challenge from verifier (S256). */
export async function generateCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64URLEncode(new Uint8Array(digest));
}

/** Base64URL encode (no padding). */
export function base64URLEncode(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");
}

/** Generate random state parameter. */
function generateState(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return base64URLEncode(array);
}

/** Build authorization URL and redirect. */
export async function initiateLogin(): Promise<void> {
  const discovery = await fetchDiscovery();
  const { codeChallenge, codeChallengeMethod } = await generateAndStorePKCE();
  const state = sessionStorage.getItem("pkce_state") ?? "";

  const params = new URLSearchParams({
    response_type: "code",
    client_id: OIDC_CLIENT_ID,
    redirect_uri: OIDC_REDIRECT_URI,
    scope: "openid profile email roles",
    code_challenge: codeChallenge,
    code_challenge_method: codeChallengeMethod,
    state,
  });

  window.location.href = `${discovery.authorization_endpoint}?${params.toString()}`;
}


/** Handle callback: exchange code for tokens, validate state, store tokens in memory. */
export async function handleCallback(code: string, state: string): Promise<void> {
  const storedState = sessionStorage.getItem("pkce_state");
  const verifier = sessionStorage.getItem("pkce_verifier");

  // Clear PKCE params from sessionStorage after use
  sessionStorage.removeItem("pkce_state");
  sessionStorage.removeItem("pkce_verifier");

  if (!storedState || state !== storedState) {
    throw new Error("Invalid OAuth state parameter");
  }
  if (!verifier) {
    throw new Error("Missing PKCE code verifier");
  }

  const discovery = await fetchDiscovery();
  const params = new URLSearchParams({
    grant_type: "authorization_code",
    code,
    redirect_uri: OIDC_REDIRECT_URI,
    client_id: OIDC_CLIENT_ID,
    code_verifier: verifier,
  });

  const resp = await fetch(discovery.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: params.toString(),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Token exchange failed: ${resp.status} ${errText}`);
  }

  const tokens = (await resp.json()) as TokenResponse;
  storeTokens(tokens);
}

/** Store tokens in memory (module scope) — never localStorage. */
export function storeTokens(tokens: TokenResponse): void {
  _accessToken = tokens.access_token;
  _tokenExpiry = Date.now() + (tokens.expires_in - 60) * 1000; // 60s buffer
  if (tokens.refresh_token) {
    _refreshToken = tokens.refresh_token;
  }
}

/** Get current access token, refreshing if needed. */
export async function getAccessToken(): Promise<string | null> {
  if (_accessToken && Date.now() < _tokenExpiry) {
    return _accessToken;
  }
  if (_refreshToken) {
    await refreshAccessToken();
    return _accessToken;
  }
  return null;
}

/** Refresh access token using refresh_token. */
async function refreshAccessToken(): Promise<void> {
  if (!_refreshToken) {
    _accessToken = null;
    return;
  }
  const discovery = await fetchDiscovery();
  const params = new URLSearchParams({
    grant_type: "refresh_token",
    refresh_token: _refreshToken,
    client_id: OIDC_CLIENT_ID,
  });

  const resp = await fetch(discovery.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: params.toString(),
  });

  if (!resp.ok) {
    _accessToken = null;
    _refreshToken = null;
    return;
  }

  const tokens = (await resp.json()) as TokenResponse;
  storeTokens(tokens);
}

/** Clear all tokens and session state. */
export function logout(): void {
  _accessToken = null;
  _tokenExpiry = 0;
  _refreshToken = null;
  sessionStorage.removeItem("pkce_verifier");
  sessionStorage.removeItem("pkce_state");
}

/** Check if user appears to be authenticated (has valid token in memory). */
export function isAuthenticated(): boolean {
  return _accessToken !== null && Date.now() < _tokenExpiry;
}

/** Get stored access token without refresh (for immediate use). */
export function getStoredAccessToken(): string | null {
  if (_accessToken && Date.now() < _tokenExpiry) {
    return _accessToken;
  }
  return null;
}
