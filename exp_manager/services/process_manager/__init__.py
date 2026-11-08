import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH
from exp_manager.services.rest_client import RestClient


class ProcessManager:
    def __init__(
        self,
        host: str,
        client_count: int,
        server_port: int,
        client_base_port: int,
        manager_port: int = 8000,
    ):
        self.host = host
        self.client_count = client_count
        self.server_port = server_port
        self.client_base_port = client_base_port
        self.manager_port = manager_port
        self.server_process = None
        self.client_processes = {}
        self.lock = threading.RLock()
        self.python = sys.executable
        self.rest_client = RestClient()

    def _command(self, script: Path, *args: str):
        return [self.python, str(script), *args]

    def _is_reachable(self, url: str, timeout: float = 0.3) -> bool:
        try:
            with urlopen(url + "/health", timeout=timeout):
                return True
        except (URLError, TimeoutError, OSError):
            return False

    def _post(self, url: str, timeout: float = 1.0):
        request = Request(url + "/api/shutdown", data=b"", method="POST")
        with urlopen(request, timeout=timeout):
            return

    # 실제 자식 프로세스 생성
    def start(self) -> dict:
        with self.lock:
            self.stop()

            # 중앙 서버 프로세스 생성
            server_script = PROJECT_ROOT_PATH / "server" / "app.py"
            self.server_process = subprocess.Popen(
                self._command(
                    server_script,
                    "--host",
                    self.host,
                    "--port",
                    str(self.server_port),
                ),
                cwd=PROJECT_ROOT_PATH,
                env=os.environ.copy(),
            )

            # 클라이언트 프로세스들 생성
            client_script = PROJECT_ROOT_PATH / "clients" / "app.py"
            for client_id in range(self.client_count):
                port = self.client_base_port + client_id
                process = subprocess.Popen(
                    self._command(
                        client_script,
                        "--client-id",
                        str(client_id),
                        "--host",
                        self.host,
                        "--port",
                        str(port),
                        "--server-url",
                        f"http://{self.host}:{self.server_port}",
                    ),
                    cwd=PROJECT_ROOT_PATH,
                    env=os.environ.copy(),
                )
                self.client_processes[client_id] = process

        try:
            self._wait_until_ready()
            self.sync_client_registry()
        except Exception:
            self.stop()
            raise
        return self.status()

    def client_registry(self):
        """중앙 서버에 전달할 클라이언트 id와 네트워크 주소 목록을 만든다."""
        return [
            {
                "client_id": client_id,
                "ip": self.host,
                "host": self.host,
                "port": self.client_base_port + client_id,
                "url": self.client_url(client_id),
            }
            for client_id in range(self.client_count)
        ]

    def sync_client_registry(self):
        """모든 자식 프로세스가 준비된 뒤 중앙 서버에 클라이언트 목록을 등록한다."""
        return self.rest_client.post_json(
            f"{self.server_url()}/api/clients/register",
            {"clients": self.client_registry()},
        )

    # 모든 프로세스가 /health 에 응답하는지 확인
    def _wait_until_ready(self, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        server_url = f"http://{self.host}:{self.server_port}"
        client_urls = [
            f"http://{self.host}:{self.client_base_port + client_id}"
            for client_id in range(self.client_count)
        ]
        while time.monotonic() < deadline:
            if self._is_reachable(server_url) and all(
                self._is_reachable(url) for url in client_urls
            ):
                return
            time.sleep(0.1)
        raise RuntimeError("서버/클라이언트가 준비되지 않았습니다. 터미널의 프로세스 오류를 확인하세요.")

    def stop(self):
        with self.lock:
            try:
                self._post(f"http://{self.host}:{self.server_port}")
            except (URLError, TimeoutError, OSError):
                pass

            for client_id, process in list(self.client_processes.items()):
                client_url = f"http://{self.host}:{self.client_base_port + client_id}"
                try:
                    self._post(client_url)
                except (URLError, TimeoutError, OSError):
                    pass

                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                if process.poll() is None:
                    process.terminate()
                self.client_processes.pop(client_id, None)

            if self.server_process is not None:
                try:
                    self.server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                if self.server_process.poll() is None:
                    self.server_process.terminate()
                self.server_process = None

    def status(self) -> dict:
        server_url = f"http://{self.host}:{self.server_port}"
        clients = []
        for client_id in range(self.client_count):
            port = self.client_base_port + client_id
            url = f"http://{self.host}:{port}"
            process = self.client_processes.get(client_id)
            clients.append(
                {
                    "client_id": client_id,
                    "url": url,
                    "running": process is not None and process.poll() is None,
                    "reachable": self._is_reachable(url),
                }
            )
        return {
            "server": {
                "url": server_url,
                "running": self.server_process is not None
                and self.server_process.poll() is None,
                "reachable": self._is_reachable(server_url),
            },
            "clients": clients,
        }

    def client_url(self, client_id: int) -> str:
        return f"http://{self.host}:{self.client_base_port + client_id}"

    def server_url(self) -> str:
        return f"http://{self.host}:{self.server_port}"

    def manager_url(self) -> str:
        return f"http://{self.host}:{self.manager_port}"
