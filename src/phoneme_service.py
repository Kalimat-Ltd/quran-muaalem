#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to obtain phoneme strings from a remote phoneme/phonetic service.

This module provides a single high-level function, ``fetch_phonemes_from_service``.
Callers pass a text string and the function will POST a JSON
payload to the configured remote endpoint and return the phoneme string.

Behaviour summary:
- HTTP method: POST
- Payload: {"text": <your text>} encoded as UTF-8 JSON
- Expected JSON response shapes (in order of preference):
    * {"phonemes": "..."} or {"phoneme": "..."}
    * {"text": "..."} or {"result": "..."}
    * A JSON list containing string items (first non-empty string is used)
    * A bare JSON string
    * If the response is not valid JSON, the raw body is returned as a string

Error handling:
- Network errors (URLError) raise RuntimeError with a short message.
- HTTP error responses raise RuntimeError with the HTTP status and up to the
    first ~200 characters of the response body (if present).
- A 404 is handled by retrying the same URL with a trailing slash appended
    (useful for services that require a terminal slash).
"""
import json
from urllib import request as urllib_request
from urllib import error as urllib_error


def fetch_phonemes(
    text: str,
    timeout: float = 10.0,
) -> str:
    """Fetch phoneme string for the given text from the configured phoneme
    service.

    This function is generic and intended for use outside of Waqf-specific
    codepaths; pass any vocalised text and it will return the phoneme string
    as produced by the remote service.

    Raises RuntimeError on network/HTTP errors or if the response cannot be
    interpreted as a phoneme string.
    """
    service_url = "https://joey-wondrous-wasp.ngrok-free.app/phonetize"

    text = text.strip()
    if not text:
        raise RuntimeError("Cannot fetch phonemes for an empty text string")

    payload = json.dumps({"text": text}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def _perform_request(url: str) -> str:
        request = urllib_request.Request(
            url, data=payload, headers=headers, method="POST"
        )
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            charset = "utf-8"
            if hasattr(response, "headers") and hasattr(
                response.headers, "get_content_charset"
            ):
                charset = response.headers.get_content_charset() or charset
            return raw_body.decode(charset, errors="replace").strip()

    def _raise_http_error(exc: urllib_error.HTTPError, url: str) -> RuntimeError:
        snippet: str = ""
        try:
            body = exc.read()
            if body:
                decoded = body.decode("utf-8", errors="replace").strip()
                if decoded:
                    if len(decoded) > 200:
                        decoded = decoded[:200] + "…"
                    snippet = f" - Response: {decoded}"
        except (UnicodeDecodeError, AttributeError):
            # Ignore failures while decoding an error body or reading attributes;
            # proceed with best effort using the HTTP status and reason only.
            snippet = ""
        reason = exc.reason if hasattr(exc, "reason") else "Unknown error"
        return RuntimeError(
            f"Phoneme service returned HTTP {exc.code} at {url} ({reason}){snippet}"
        )

    try:
        body_text = _perform_request(service_url)
    except urllib_error.HTTPError as exc:
        if exc.code == 404:
            alternate_url = service_url.rstrip("/") + "/"
            if alternate_url != service_url:
                try:
                    body_text = _perform_request(alternate_url)
                    service_url = alternate_url
                except urllib_error.HTTPError as exc_alt:
                    raise _raise_http_error(exc_alt, alternate_url) from exc_alt
                except urllib_error.URLError as exc_alt:
                    raise RuntimeError(
                        f"Unable to reach phoneme service ({exc_alt})"
                    ) from exc_alt
            else:
                raise _raise_http_error(exc, service_url) from exc
        else:
            raise _raise_http_error(exc, service_url) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Unable to reach phoneme service ({exc})") from exc

    if not body_text:
        raise RuntimeError("Phoneme service returned an empty response")

    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError:
        return body_text

    if isinstance(parsed, dict):
        for key in ("phonemes", "phoneme", "text", "result"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise RuntimeError("Phoneme service response did not contain a phoneme string")

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                return item.strip()
        raise RuntimeError(
            "Phoneme service response list did not contain a phoneme string"
        )

    if isinstance(parsed, str):
        return parsed.strip()

    raise RuntimeError("Unexpected phoneme service response format")
