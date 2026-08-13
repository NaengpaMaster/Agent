import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.api.inquiry_chat import create_inquiry_chat_answer
from app.core.config import Settings
from app.schemas.inquiry_chat import InquiryChatRequest


class InquiryChatApiTest(unittest.TestCase):
    def test_rejects_request_without_internal_api_key(self):
        settings = Settings("", "gpt-4.1-mini", "", "", 8000, "secret")

        with (
            patch("app.api.inquiry_chat.get_settings", return_value=settings),
            self.assertRaises(HTTPException) as raised,
        ):
            create_inquiry_chat_answer(InquiryChatRequest(question="문의 방법"), None)

        self.assertEqual(raised.exception.status_code, 401)

    def test_accepts_matching_internal_api_key(self):
        settings = Settings("", "gpt-4.1-mini", "", "", 8000, "secret")
        expected = object()

        with (
            patch("app.api.inquiry_chat.get_settings", return_value=settings),
            patch("app.api.inquiry_chat.answer_inquiry", return_value=expected),
        ):
            result = create_inquiry_chat_answer(InquiryChatRequest(question="문의 방법"), "secret")

        self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
