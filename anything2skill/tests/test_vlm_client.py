"""Tests for VLM client."""

import os
import tempfile
from types import SimpleNamespace

import pytest

from anything2skill.vlm.client import (
    RetryableVLMResponseError,
    VLMClient,
    encode_image,
    encode_image_file,
)


class TestImageEncoding:
    def test_encode_image_bytes(self):
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64 = encode_image(data)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_encode_image_file_png(self):
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (10, 10)).save(f, format="PNG")
            f.flush()
            data_url = encode_image_file(f.name)
        os.unlink(f.name)
        assert data_url.startswith("data:image/png;base64,")

    def test_encode_image_file_jpg(self):
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new("RGB", (10, 10)).save(f, format="JPEG")
            f.flush()
            data_url = encode_image_file(f.name)
        os.unlink(f.name)
        assert data_url.startswith("data:image/jpeg;base64,")


class TestVLMClientInit:
    def test_init_with_explicit_params(self, monkeypatch):
        # Unset SOCKS proxy to avoid socksio dependency in tests
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)
        client = VLMClient(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )
        assert client.model == "test-model"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env-url/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        client = VLMClient(model="gpt-4o")
        assert client.model == "gpt-4o"


class TestVLMClientChat:
    def test_extract_text_returns_content(self, monkeypatch):
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)
        client = VLMClient(model="test-model", api_key="test-key")

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ACTION: WAIT"),
                )
            ]
        )

        assert client._extract_text(response) == "ACTION: WAIT"

    def test_extract_text_rejects_bad_response(self, monkeypatch):
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)
        client = VLMClient(model="test-model", api_key="test-key")

        bad_response = SimpleNamespace(choices=None)

        with pytest.raises(RetryableVLMResponseError):
            client._extract_text(bad_response)
