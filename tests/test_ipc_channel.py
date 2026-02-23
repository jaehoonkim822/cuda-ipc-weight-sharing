"""Unit tests for ipc_channel — no GPU required."""

import os
import tempfile
import threading
import time

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.ipc_channel import HandleClient, HandleServer


@pytest.fixture
def socket_path(tmp_path):
    """Provide a unique socket path for each test."""
    return str(tmp_path / "test.sock")


class TestHandleServerClient:
    def test_basic_send_recv(self, socket_path):
        payload = b"hello world" * 100
        server = HandleServer(socket_path)
        server.set_handles(payload)
        server.serve_in_background()

        try:
            time.sleep(0.2)  # let server bind
            client = HandleClient(socket_path, max_retries=3)
            received = client.fetch_handles()
            assert received == payload
        finally:
            server.stop()

    def test_large_payload(self, socket_path):
        # 10 MB payload — simulates a large model's handles
        payload = os.urandom(10 * 1024 * 1024)
        server = HandleServer(socket_path)
        server.set_handles(payload)
        server.serve_in_background()

        try:
            time.sleep(0.2)
            client = HandleClient(socket_path, max_retries=3)
            received = client.fetch_handles()
            assert received == payload
        finally:
            server.stop()

    def test_multiple_clients_sequential(self, socket_path):
        payload = b"test_data_12345"
        server = HandleServer(socket_path)
        server.set_handles(payload)
        server.serve_in_background()

        try:
            time.sleep(0.2)
            for _ in range(5):
                client = HandleClient(socket_path, max_retries=3)
                assert client.fetch_handles() == payload
        finally:
            server.stop()

    def test_multiple_clients_concurrent(self, socket_path):
        payload = b"concurrent_test" * 1000
        server = HandleServer(socket_path)
        server.set_handles(payload)
        server.serve_in_background()

        results = {}
        errors = []

        def _fetch(i):
            try:
                client = HandleClient(socket_path, max_retries=5)
                results[i] = client.fetch_handles()
            except Exception as e:
                errors.append((i, e))

        try:
            time.sleep(0.2)
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

    def test_hot_swap(self, socket_path):
        payload_v1 = b"version_1"
        payload_v2 = b"version_2_new_data"

        server = HandleServer(socket_path)
        server.set_handles(payload_v1)
        server.serve_in_background()

        try:
            time.sleep(0.2)
            client = HandleClient(socket_path, max_retries=3)
            assert client.fetch_handles() == payload_v1

            # Hot swap
            server.set_handles(payload_v2)

            client2 = HandleClient(socket_path, max_retries=3)
            assert client2.fetch_handles() == payload_v2
        finally:
            server.stop()

    def test_client_retry_on_no_socket(self, socket_path):
        """Client retries when socket doesn't exist yet."""
        payload = b"delayed_start"

        def _delayed_server():
            time.sleep(1.0)
            server = HandleServer(socket_path)
            server.set_handles(payload)
            server.serve_in_background()
            return server

        server_holder = [None]

        def _start():
            server_holder[0] = _delayed_server()

        t = threading.Thread(target=_start)
        t.start()

        try:
            client = HandleClient(socket_path, max_retries=10)
            result = client.fetch_handles()
            assert result == payload
        finally:
            t.join(timeout=5)
            if server_holder[0]:
                server_holder[0].stop()

    def test_server_stop_cleans_socket(self, socket_path):
        server = HandleServer(socket_path)
        server.set_handles(b"cleanup_test")
        server.serve_in_background()
        time.sleep(0.2)
        assert os.path.exists(socket_path)

        server.stop()
        time.sleep(0.5)
        assert not os.path.exists(socket_path)
