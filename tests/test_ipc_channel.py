"""Unit tests for ipc_channel (ZMQ transport) — no GPU required."""

import os
import threading
import time

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.ipc_channel import HandleClient, HandleServer


@pytest.fixture
def endpoint(tmp_path):
    """Provide a unique ZMQ IPC endpoint for each test."""
    return f"ipc://{tmp_path}/test.zmq"


class TestHandleServerClient:
    def test_basic_send_recv(self, endpoint):
        payload = b"hello world" * 100
        server = HandleServer(endpoint)
        server.set_handles(payload)
        server.serve_in_background()

        try:
            time.sleep(0.3)  # let server bind
            client = HandleClient(endpoint, max_retries=3)
            received = client.fetch_handles()
            assert received == payload
        finally:
            server.stop()

    def test_large_payload(self, endpoint):
        # 10 MB payload — simulates a large model's handles
        payload = os.urandom(10 * 1024 * 1024)
        server = HandleServer(endpoint)
        server.set_handles(payload)
        server.serve_in_background()

        try:
            time.sleep(0.3)
            client = HandleClient(endpoint, max_retries=3)
            received = client.fetch_handles()
            assert received == payload
        finally:
            server.stop()

    def test_multiple_clients_sequential(self, endpoint):
        payload = b"test_data_12345"
        server = HandleServer(endpoint)
        server.set_handles(payload)
        server.serve_in_background()

        try:
            time.sleep(0.3)
            for _ in range(5):
                client = HandleClient(endpoint, max_retries=3)
                assert client.fetch_handles() == payload
        finally:
            server.stop()

    def test_multiple_clients_concurrent(self, endpoint):
        payload = b"concurrent_test" * 1000
        server = HandleServer(endpoint)
        server.set_handles(payload)
        server.serve_in_background()

        results = {}
        errors = []

        def _fetch(i):
            try:
                client = HandleClient(endpoint, max_retries=5)
                results[i] = client.fetch_handles()
            except Exception as e:
                errors.append((i, e))

        try:
            time.sleep(0.3)
            threads = [threading.Thread(target=_fetch, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Errors: {errors}"
            for i in range(5):
                assert results[i] == payload
        finally:
            server.stop()

    def test_hot_swap(self, endpoint):
        payload_v1 = b"version_1"
        payload_v2 = b"version_2_new_data"

        server = HandleServer(endpoint)
        server.set_handles(payload_v1)
        server.serve_in_background()

        try:
            time.sleep(0.3)
            client = HandleClient(endpoint, max_retries=3)
            assert client.fetch_handles() == payload_v1

            # Hot swap
            server.set_handles(payload_v2)

            client2 = HandleClient(endpoint, max_retries=3)
            assert client2.fetch_handles() == payload_v2
        finally:
            server.stop()

    def test_client_retry_on_no_server(self, endpoint):
        """Client retries when server is not yet available."""
        payload = b"delayed_start"

        def _delayed_server():
            time.sleep(1.0)
            server = HandleServer(endpoint)
            server.set_handles(payload)
            server.serve_in_background()
            return server

        server_holder = [None]

        def _start():
            server_holder[0] = _delayed_server()

        t = threading.Thread(target=_start)
        t.start()

        try:
            client = HandleClient(endpoint, max_retries=10)
            result = client.fetch_handles()
            assert result == payload
        finally:
            t.join(timeout=5)
            if server_holder[0]:
                server_holder[0].stop()

    def test_server_stop_rejects_clients(self, endpoint):
        """After server.stop(), new clients should fail."""
        server = HandleServer(endpoint)
        server.set_handles(b"test_data")
        server.serve_in_background()
        time.sleep(0.3)

        # Confirm it works before stopping
        client = HandleClient(endpoint, max_retries=3)
        assert client.fetch_handles() == b"test_data"

        server.stop()
        time.sleep(0.5)

        # New client should fail after retries
        with pytest.raises(ConnectionRefusedError):
            HandleClient(endpoint, max_retries=2).fetch_handles()
