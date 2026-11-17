import threading


def schedule_shutdown(app, client_id: int):
    server = app.extensions["server"]
    if server is None:
        return None
    threading.Thread(target=server.shutdown, daemon=True).start()
