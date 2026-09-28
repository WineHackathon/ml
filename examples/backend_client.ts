export async function recognize(baseUrl: string, image: Blob, token?: string) {
  const body = new FormData();
  body.append("image", image, "query.webp");
  const signal = AbortSignal.timeout(12_000);
  const headers = token ? { "X-Token": token } : undefined;
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/v1/recognize`, { method: "POST", body, headers, signal });
  if (!response.ok) throw new Error(`wine API ${response.status}: ${await response.text()}`);
  return response.json();
}
