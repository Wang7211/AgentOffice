"""EmailTool 测试用例（mock smtplib 避免真实网络调用）。"""

from typing import Any
from unittest import mock

import pytest

from tools.email_tool import EmailTool
from utils.exception import ToolException


class TestEmailTool:
    def setup_method(self) -> None:
        self._tool = EmailTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "email"
        assert "发送邮件" in spec.description

    def test_run_missing_required(self) -> None:
        with pytest.raises(ToolException, match="收件人|主题|正文"):
            self._tool.run({"to": "", "subject": "", "body": ""})

    def test_run_missing_subject(self) -> None:
        with pytest.raises(ToolException, match="收件人|主题|正文"):
            self._tool.run({"to": "test@example.com", "subject": "", "body": "content"})

    def test_run_smtp_not_configured(self) -> None:
        with mock.patch("tools.email_tool.get_settings") as mock_settings:
            mock_settings.return_value.smtp_host = ""
            mock_settings.return_value.smtp_from_email = ""
            with pytest.raises(ToolException, match="SMTP 未配置"):
                self._tool.run({"to": "a@b.com", "subject": "Hi", "body": "Hello"})

    @mock.patch("tools.email_tool.smtplib.SMTP")
    def test_run_success(self, mock_smtp: mock.MagicMock) -> None:
        mock_server = mock.MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        with mock.patch("tools.email_tool.get_settings") as mock_settings:
            mock_settings.return_value.smtp_host = "smtp.example.com"
            mock_settings.return_value.smtp_port = 587
            mock_settings.return_value.smtp_username = "user"
            mock_settings.return_value.smtp_password = "pass"
            mock_settings.return_value.smtp_from_email = "from@example.com"
            mock_settings.return_value.request_timeout = 30.0

            result = self._tool.run({
                "to": "to@example.com",
                "subject": "Test Subject",
                "body": "Test Body",
            })

        assert "已发送" in result.content
        assert "to@example.com" in result.content
        assert result.metadata["to"] == "to@example.com"
        mock_server.send_message.assert_called_once()

    @mock.patch("tools.email_tool.smtplib.SMTP")
    def test_run_with_cc(self, mock_smtp: mock.MagicMock) -> None:
        mock_server = mock.MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        with mock.patch("tools.email_tool.get_settings") as mock_settings:
            mock_settings.return_value.smtp_host = "smtp.example.com"
            mock_settings.return_value.smtp_port = 587
            mock_settings.return_value.smtp_username = "user"
            mock_settings.return_value.smtp_password = "pass"
            mock_settings.return_value.smtp_from_email = "from@example.com"
            mock_settings.return_value.request_timeout = 30.0

            result = self._tool.run({
                "to": "to@example.com",
                "subject": "Test",
                "body": "Body",
                "cc": "cc@example.com",
            })

        assert result.metadata["cc"] == "cc@example.com"
        mock_server.send_message.assert_called_once()

    @mock.patch("tools.email_tool.smtplib.SMTP")
    def test_run_smtp_error(self, mock_smtp: mock.MagicMock) -> None:
        mock_smtp.return_value.__enter__.return_value.send_message.side_effect = (
            __import__("smtplib").SMTPException("Connection refused")
        )

        with mock.patch("tools.email_tool.get_settings") as mock_settings:
            mock_settings.return_value.smtp_host = "smtp.example.com"
            mock_settings.return_value.smtp_port = 587
            mock_settings.return_value.smtp_username = "user"
            mock_settings.return_value.smtp_password = "pass"
            mock_settings.return_value.smtp_from_email = "from@example.com"
            mock_settings.return_value.request_timeout = 30.0

            with pytest.raises(ToolException, match="邮件发送失败"):
                self._tool.run({
                    "to": "to@example.com",
                    "subject": "Test",
                    "body": "Body",
                })
