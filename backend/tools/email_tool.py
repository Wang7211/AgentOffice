"""邮件发送工具。"""

import smtplib
from email.message import EmailMessage
from typing import Any

from config.settings import get_settings
from tools.base import BaseTool
from tools.base import ToolResult
from utils.exception import ToolException


class EmailTool(BaseTool):
    """通过 SMTP 发送邮件的工具。"""

    name = "email"
    description = "发送邮件，支持收件人、抄送、主题和正文。"
    input_schema = {
        "to": "必填，收件人邮箱地址，多个用逗号分隔。",
        "subject": "必填，邮件主题。",
        "body": "必填，邮件正文。",
        "cc": "可选，抄送邮箱地址，多个用逗号分隔。",
    }

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """发送邮件。

        参数:
            tool_input: 包含 to、subject、body、可选 cc 的字典。

        返回:
            发送结果。

        异常:
            ToolException: 缺少必填参数、SMTP 未配置或发送失败时抛出。
        """
        to = str(tool_input.get("to", "")).strip()
        subject = str(tool_input.get("subject", "")).strip()
        body = str(tool_input.get("body", "")).strip()
        cc = str(tool_input.get("cc", "")).strip()

        if not to or not subject or not body:
            raise ToolException("收件人、主题和正文不能为空")

        settings = get_settings()
        if not settings.smtp_host or not settings.smtp_from_email:
            raise ToolException(
                "SMTP 未配置，请在环境变量中设置 SMTP_HOST、SMTP_FROM_EMAIL 等参数。"
            )

        msg = EmailMessage()
        msg["From"] = settings.smtp_from_email
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        msg.set_content(body)

        recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
        if cc:
            recipients.extend(addr.strip() for addr in cc.split(",") if addr.strip())

        try:
            if settings.smtp_port == 465:
                with smtplib.SMTP_SSL(
                    settings.smtp_host, settings.smtp_port, timeout=settings.request_timeout
                ) as server:
                    if settings.smtp_username:
                        server.login(settings.smtp_username, settings.smtp_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(
                    settings.smtp_host, settings.smtp_port, timeout=settings.request_timeout
                ) as server:
                    if settings.smtp_port in (25, 587):
                        server.starttls()
                    if settings.smtp_username:
                        server.login(settings.smtp_username, settings.smtp_password)
                    server.send_message(msg)
        except smtplib.SMTPException as exc:
            raise ToolException(f"邮件发送失败：{exc}") from exc
        except OSError as exc:
            raise ToolException(f"邮件服务器连接失败：{exc}") from exc

        return ToolResult(
            content=f"邮件已发送至 {to}，主题：{subject}",
            metadata={"to": to, "cc": cc, "subject": subject},
        )
