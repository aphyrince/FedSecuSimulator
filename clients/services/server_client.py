"""클라이언트에서 중앙 서버로 REST 요청을 보내는 서비스."""

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 3.0,
    *,
    compact: bool = False,
) -> dict[str, Any]:
    """중앙 서버 API에 JSON payload를 전송한다."""
    separators = (",", ":") if compact else None
    body = json.dumps(payload, separators=separators).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
