"""ZMQ-based transport for IPC handle exchange.

HandleServer (REP socket) serves serialized IPC handles.
HandleClient (REQ socket) fetches them with exponential backoff retry.
"""

import threading
import time
import logging

import zmq

from .config import ZMQ_ENDPOINT

log = logging.getLogger(__name__)


class HandleServer:
    """Serves serialized IPC handles over a ZMQ REP socket.

    Usage:
        server = HandleServer(endpoint)
        server.set_handles(encoded_bytes)
        server.serve_in_background()
        # ... later ...
        server.stop()
    """

    def __init__(self, endpoint: str = ZMQ_ENDPOINT):
        self._endpoint = endpoint
        self._handles_data: bytes | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx: zmq.Context | None = None

    def set_handles(self, data: bytes) -> None:
        """Set (or hot-swap) the handle payload to serve."""
        with self._lock:
            self._handles_data = data
        log.info("Handle payload updated (%d bytes)", len(data))

    def serve_in_background(self) -> threading.Thread:
        """Start serving in a daemon thread. Returns the thread."""
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal shutdown and clean up."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Context is terminated inside _serve_loop after socket close

    def _serve_loop(self) -> None:
        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self._endpoint)
        log.info("HandleServer listening on %s", self._endpoint)

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while not self._stop_event.is_set():
            events = dict(poller.poll(timeout=1000))  # 1 second
            if sock not in events:
                continue

            try:
                sock.recv()  # consume the REQ message (content ignored)
                with self._lock:
                    data = self._handles_data
                if data is None:
                    log.warning("Client connected but no handles available")
                    sock.send(b"")
                    continue
                sock.send(data)
                log.info("Sent handles to client (%d bytes)", len(data))
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                log.exception("Error serving client")

        sock.close()
        self._ctx.term()
        self._ctx = None
        log.info("HandleServer stopped")


class HandleClient:
    """Connects to a HandleServer and retrieves serialized IPC handles.

    Retries with exponential backoff if the server is not yet available.
    """

    def __init__(self, endpoint: str = ZMQ_ENDPOINT, max_retries: int = 10):
        self._endpoint = endpoint
        self._max_retries = max_retries

    def fetch_handles(self) -> bytes:
        """Connect to the server and return the raw handle payload."""
        ctx = zmq.Context()
        delay = 0.1
        try:
            for attempt in range(1, self._max_retries + 1):
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, 5000)
                sock.connect(self._endpoint)
                try:
                    sock.send(b"fetch")
                    data = sock.recv()
                    sock.close()
                    if data:
                        log.info(
                            "Received handles from server (%d bytes, attempt %d)",
                            len(data),
                            attempt,
                        )
                        return data
                    # Empty response means server had no handles yet
                    raise zmq.Again("Server returned empty response")
                except zmq.Again:
                    # Timeout or empty — recreate socket (REQ/REP state machine reset)
                    sock.close()
                    if attempt == self._max_retries:
                        raise ConnectionRefusedError(
                            f"Server at {self._endpoint} did not respond after "
                            f"{self._max_retries} attempts"
                        )
                    log.info(
                        "No response (attempt %d/%d), retrying in %.1fs",
                        attempt,
                        self._max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                except zmq.ZMQError as e:
                    sock.close()
                    if attempt == self._max_retries:
                        raise ConnectionRefusedError(
                            f"ZMQ error connecting to {self._endpoint}: {e}"
                        )
                    log.info(
                        "Connection error (attempt %d/%d): %s, retrying in %.1fs",
                        attempt,
                        self._max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
        finally:
            ctx.term()
