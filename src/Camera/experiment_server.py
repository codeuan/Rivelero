"""UE Editor experiment bridge. Configure the shared root and terrain registration once."""
import json
import queue
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import unreal
from trace_visibility import trace_steps

_active = None


def start(shared_root, northwest_cm, datum_z_cm=0, host="127.0.0.1", port=8765, othonna_mesh=None):
    global _active
    if _active is not None:
        raise RuntimeError("Experiment server is already running.")
    root = Path(shared_root).resolve(strict=True)
    pending = queue.Queue()
    states = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            match = re.fullmatch(r"/jobs/([0-9a-f]{32})", self.path)
            if not match:
                self.send_error(404)
                return
            job = match[1]
            if not (root / job / "request.json").is_file():
                self.send_error(409, "Batch missing in shared experiment folder")
                return
            with lock:
                if job not in states:
                    states[job] = "Queued"
                    pending.put(job)
                message = states[job]
            body = json.dumps({"status": message}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer((host, port), Handler)
    current = None
    steps = None

    def update(value):
        with lock:
            states[current] = value

    def job_steps(directory):
        yield from trace_steps(directory, northwest_cm, datum_z_cm)
        request = json.loads((directory / "request.json").read_text())
        if request.get("experiment") == 2:
            from validate_othonna import validate_plants
            yield from validate_plants(directory, northwest_cm, datum_z_cm, othonna_mesh, request)

    def tick(delta):
        nonlocal current, steps
        try:
            if current is None:
                try:
                    current = pending.get_nowait()
                except queue.Empty:
                    return
                steps = job_steps(root / current)
            if (root / current / "cancelled").exists():
                steps.close()
                update("Cancelled")
                current = None
                return
            layout, camera = next(steps)
            update(f"{layout}, {camera}")
        except StopIteration:
            update("Complete")
            current = None
        except Exception as exc:
            if steps:
                steps.close()
            update(f"Error: {exc}")
            unreal.log_error(str(exc))
            current = None

    handle = unreal.register_slate_post_tick_callback(tick)
    threading.Thread(target=http.serve_forever, daemon=True).start()

    def close():
        unreal.unregister_slate_post_tick_callback(handle)
        http.shutdown()
        http.server_close()
        if steps is not None:
            steps.close()

    _active = close
    unreal.log(f"Experiment server listening on {host}:{port}")


def stop():
    global _active
    if _active is not None:
        _active()
        _active = None
