"""클라이언트에서 중앙 서버로 REST 요청을 보내는 서비스."""

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict[str, Any], timeout: float = 3.0) -> dict[str, Any]:
    """중앙 서버 API에 JSON payload를 전송한다."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
