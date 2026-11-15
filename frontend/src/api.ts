import { useEffect, useState } from "react";

export async function request<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      body?.error?.message ||
        "Could not load experiment data. Check the backend connection.",
    );
  return body as T;
}

type Remote<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  key: string;
};
export function useRemote<T>(url: string | null, refresh = 0) {
  const key = `${url}|${refresh}`;
  const [state, setState] = useState<Remote<T>>({
    data: null,
    loading: !!url,
    error: null,
    key,
  });
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState({ data: null, loading: !!url, error: null, key });
    if (url)
      request<T>(url, { signal: controller.signal })
        .then((data) => {
          if (active) setState({ data, loading: false, error: null, key });
        })
        .catch((error) => {
          if (active)
            setState({ data: null, loading: false, error: error.message, key });
        });
    return () => {
      active = false;
      controller.abort();
    };
  }, [url, refresh, key]);
  return state.key === key
    ? state
    : { data: null, loading: !!url, error: null, key };
}

export function query(
  path: string,
  values: Record<string, string | number | null | undefined>,
) {
  const parameters = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "")
      parameters.set(key, String(value));
  });
  return `/api/v1/${path}?${parameters.toString()}`;
}
