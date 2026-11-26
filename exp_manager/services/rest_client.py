import json
from urllib.error import HTTPError
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
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                body = json.loads(error.read(4096))
            except (OSError, ValueError):
                body = None
            if isinstance(body, dict) and isinstance(body.get("error"), str):
                error.msg = f"{error.msg}: {body['error']}"
            raise

    def get_json(self, url: str, timeout: float = 3.0):
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
