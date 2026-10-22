/** React context and hook for authentication state. */
import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { getAccessToken, logout as oidcLogout, initiateLogin, getStoredAccessToken } from "./oidc";

interface AuthContextValue {
  isAuthenticated: boolean;
  isLoading: boolean;
  login: () => Promise<void>;
  logout: () => void;
  getToken: () => string | null;
  refreshAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }): React.JSX.Element {
  const [authenticated, setAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  const refreshAuth = useCallback(async () => {
    const token = await getAccessToken();
    setAuthenticated(!!token);
  }, []);

  useEffect(() => {
    void (async () => {
      const token = await getAccessToken();
      setAuthenticated(!!token);
      setLoading(false);
    })();
  }, []);

  const login = useCallback(async () => {
    setLoading(true);
    try {
      await initiateLogin();
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    oidcLogout();
    setAuthenticated(false);
  }, []);

  const getToken = useCallback(() => {
    return getStoredAccessToken();
  }, []);

  return (
    <AuthContext.Provider value={{ isAuthenticated: authenticated, isLoading: loading, login, logout, getToken, refreshAuth }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
