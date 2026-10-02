const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export function normalizeApiBaseUrl(configuredUrl?: string): string {
  const baseUrl = configuredUrl?.trim() || DEFAULT_API_BASE_URL;
  return baseUrl.replace(/\/+$/, '');
}

export const API_BASE_URL = normalizeApiBaseUrl(import.meta.env.VITE_API_URL);
