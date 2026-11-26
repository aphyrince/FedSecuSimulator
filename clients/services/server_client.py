"""클라이언트에서 중앙 서버로 REST 요청을 보내는 서비스."""

import json
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from common.runtime_log import runtime_logger
from common.update_transport import UPDATE_CONTENT_TYPE, encode_update


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 3.0,
    *,
    compact: bool = False,
) -> dict[str, Any]:
    """중앙 서버 API에 JSON payload를 전송한다."""
    return _post(url, payload, timeout, compact=compact, binary=False)


def post_tensor_update(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _post(url, payload, timeout, compact=False, binary=True)


def _post(url: str, payload: dict[str, Any], timeout: float, *, compact: bool, binary: bool) -> dict[str, Any]:
    round_id, client_id = payload.get("round"), payload.get("client_id")
    is_update = type(round_id) is int and type(client_id) is int and url.endswith(f"/api/rounds/{round_id}/updates")
    headers = {"Content-Type": UPDATE_CONTENT_TYPE if binary else "application/json"}
    if is_update:
        headers["X-FedSec-Client-Id"] = str(client_id)
        runtime_logger.info("client serializing update round=%s client=%s", round_id, client_id)
    started = time.monotonic()
    if binary:
        body = encode_update(payload)
    else:
        separators = (",", ":") if compact else None
        body = json.dumps(payload, separators=separators).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    if is_update:
        runtime_logger.info(
            "client sending update experiment=%s round=%s client=%s bytes=%s serialize_s=%.2f timeout_s=%s content_type=%s",
            payload.get("experiment_id"), round_id, client_id, len(body), time.monotonic() - started, timeout,
            headers["Content-Type"],
        )
    sent_at = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            if is_update:
                runtime_logger.info(
                    "client update response round=%s client=%s http=%s elapsed_s=%.2f",
                    round_id, client_id, response.status, time.monotonic() - sent_at,
                )
            return result
    except HTTPError as error:
        if is_update:
            try:
                detail = json.loads(error.read(4096))
            except (OSError, ValueError):
                detail = None
            if isinstance(detail, dict) and isinstance(detail.get("error"), str):
                error.msg = f"{error.msg}: {detail['error'][:300]}"
            runtime_logger.warning(
                "client update rejected round=%s client=%s error=%s elapsed_s=%.2f",
                round_id, client_id, error, time.monotonic() - sent_at,
            )
        raise
