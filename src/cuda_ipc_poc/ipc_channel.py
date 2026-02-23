"""Unix Domain Socket transport for IPC handle exchange.

Uses a length-prefixed binary protocol:
  [8 bytes big-endian length][payload bytes]

The HandleServer holds the current serialized handles in memory and serves
them to any connecting HandleClient. Supports hot-swap via set_handles().
"""

import os
import socket
import struct
import threading
import time
import logging

from .config import SOCKET_PATH

log = logging.getLogger(__name__)

_HEADER_SIZE = 8  # bytes, big-endian uint64


def _send_msg(sock: socket.socket, data: bytes) -> None:
    """Send a length-prefixed message."""
    header = struct.pack(">Q", len(data))
    sock.sendall(header + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from the socket."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError(
                f"Connection closed with {remaining} bytes remaining"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_msg(sock: socket.socket) -> bytes:
    """Receive a length-prefixed message."""
    header = _recv_exact(sock, _HEADER_SIZE)
    length = struct.unpack(">Q", header)[0]
    return _recv_exact(sock, length)


class HandleServer:
    """Serves serialized IPC handles over a Unix Domain Socket.

    Usage:
        server = HandleServer(socket_path)
        server.set_handles(encoded_bytes)
        server.serve_in_background()
        # ... later ...
        server.stop()
    """

    def __init__(self, socket_path: str = SOCKET_PATH):
        self._socket_path = socket_path
        self._handles_data: bytes | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

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
        """Signal shutdown and clean up the socket file."""
        self._stop_event.set()
        # Unblock accept() by connecting to ourselves
        if self._server_sock:
            try:
                wake = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                wake.connect(self._socket_path)
                wake.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        self._cleanup_socket()

    def _cleanup_socket(self) -> None:
        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

    def _serve_loop(self) -> None:
        # Clean stale socket file
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(8)
        self._server_sock.settimeout(1.0)
        log.info("HandleServer listening on %s", self._socket_path)

        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise

            try:
                with self._lock:
                    data = self._handles_data
                if data is None:
                    log.warning("Client connected but no handles available")
                    conn.close()
                    continue
                _send_msg(conn, data)
                log.info("Sent handles to client (%d bytes)", len(data))
            except Exception:
                log.exception("Error serving client")
            finally:
                conn.close()

        self._cleanup_socket()
        log.info("HandleServer stopped")


class HandleClient:
    """Connects to a HandleServer and retrieves serialized IPC handles.

    Retries with exponential backoff if the server is not yet available.
    """

    def __init__(self, socket_path: str = SOCKET_PATH, max_retries: int = 10):
        self._socket_path = socket_path
        self._max_retries = max_retries

    def fetch_handles(self) -> bytes:
        """Connect to the server and return the raw handle payload."""
        delay = 0.1
        for attempt in range(1, self._max_retries + 1):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self._socket_path)
                data = _recv_msg(sock)
                sock.close()
                log.info(
                    "Received handles from server (%d bytes, attempt %d)",
                    len(data),
                    attempt,
                )
                return data
            except ConnectionRefusedError:
                if attempt == self._max_retries:
                    raise
                log.info(
                    "Connection refused (attempt %d/%d), retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
            except FileNotFoundError:
                if attempt == self._max_retries:
                    raise ConnectionRefusedError(
                        f"Socket {self._socket_path} does not exist after "
                        f"{self._max_retries} attempts"
                    )
                log.info(
                    "Socket not found (attempt %d/%d), retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
