"""Smoke tests that do not require a model, a PDF or a Chroma index."""

import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api.indexer import chunk_text
from api.main import app
from api.rag_local import format_context


class CoreSmokeTests(unittest.TestCase):
    def test_chunk_text_with_overlap(self):
        self.assertEqual(chunk_text("abcdefgh", chunk_size=5, overlap=2), ["abcde", "defgh", "gh"])

    def test_format_context_includes_source_and_page(self):
        context = format_context([{"text": "Texte neutre", "metadata": {"source": "document.pdf", "page": 1}}])
        self.assertIn("document.pdf p.1", context)
        self.assertIn("Texte neutre", context)


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_serves_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Local RAG Assistant", response.text)

    @patch("api.main.os.path.isdir", return_value=False)
    @patch("api.main.httpx.Client")
    def test_health_without_services(self, client_class, _isdir):
        client_class.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("offline")
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": False, "llm_ok": False, "chroma_ok": False})

    def test_empty_question_is_rejected(self):
        response = self.client.post("/chat", json={"question": "  "})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
