import json
from urllib.request import Request, urlopen


class RestClient:
    """실험 관리자와 자식 Flask 서버 사이의 JSON POST 요청을 담당한다."""

    def post_json(self, url: str, payload: dict, timeout: float = 3.0):
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_json(self, url: str, timeout: float = 3.0):
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
