"""
Unit and API integration tests for server.py and chat_manager.py.
"""

import unittest
from fastapi.testclient import TestClient
import pymupdf

from server import app
from src.chat_manager import session_manager


def make_test_pdf() -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Document A states that docRag supports multi-document workspaces.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestChatManagerAndAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

    def test_config_endpoint(self):
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["app_name"], "docRag")
        self.assertIn("model_name", data)

    def test_create_and_list_chats(self):
        res_create = self.client.post("/api/chats", json={"title": "Test Research Chat"})
        self.assertEqual(res_create.status_code, 200)
        session_id = res_create.json()["session_id"]
        self.assertIsNotNone(session_id)

        res_list = self.client.get("/api/chats")
        self.assertEqual(res_list.status_code, 200)
        session_ids = [s["session_id"] for s in res_list.json()]
        self.assertIn(session_id, session_ids)

    def test_upload_document_to_chat(self):
        # Create session
        res_create = self.client.post("/api/chats", json={"title": "Doc Chat"})
        session_id = res_create.json()["session_id"]

        pdf_bytes = make_test_pdf()
        files = [
            ("files", ("test_doc.pdf", pdf_bytes, "application/pdf")),
        ]
        res_upload = self.client.post(f"/api/chats/{session_id}/upload", files=files)
        self.assertEqual(res_upload.status_code, 200)
        data = res_upload.json()
        self.assertEqual(len(data["uploaded"]), 1)
        self.assertEqual(data["uploaded"][0]["file_name"], "test_doc.pdf")

        # Verify chat details now include document
        res_chat = self.client.get(f"/api/chats/{session_id}")
        self.assertEqual(res_chat.status_code, 200)
        self.assertEqual(len(res_chat.json()["documents"]), 1)


if __name__ == "__main__":
    unittest.main()
