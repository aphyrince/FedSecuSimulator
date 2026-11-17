import threading


def schedule_shutdown(app):
    server = app.extensions["server"]
    if server is None:
        return
    threading.Thread(target=server.shutdown, daemon=True).start()
