"""
Safety feature tests: delete → Trash and send → Drafts.

These tests verify the two core safety invariants:
1. Calling delete_emails never permanently removes emails — they are moved to Trash.
2. Calling send_email never sends via SMTP by default — emails go to Drafts.

Tests are structured at three levels:
- Handler level: ClassicEmailHandler delegates to move_emails / save_to_drafts
- IMAP level: the correct IMAP commands are issued (COPY before \\Deleted)
- MCP tool level: end-to-end from MCP tool call down to the handler
"""
from __future__ import annotations

import asyncio
from email.mime.text import MIMEText
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_email_server.app import delete_emails, send_email
from mcp_email_server.config import EmailServer, EmailSettings
from mcp_email_server.emails.classic import ClassicEmailHandler, EmailClient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def email_settings():
    return EmailSettings(
        account_name="test_account",
        full_name="Test User",
        email_address="test@example.com",
        trash_folder_name="Trash",
        draft_folder_name="Drafts",
        incoming=EmailServer(
            user_name="test_user",
            password="test_password",
            host="imap.example.com",
            port=993,
            use_ssl=True,
        ),
        outgoing=EmailServer(
            user_name="test_user",
            password="test_password",
            host="smtp.example.com",
            port=465,
            use_ssl=True,
        ),
    )


@pytest.fixture
def handler(email_settings):
    return ClassicEmailHandler(email_settings)


@pytest.fixture
def mock_imap():
    """IMAP mock that simulates COPY+DELETE fallback (no MOVE capability)."""
    m = AsyncMock()
    m._client_task = asyncio.Future()
    m._client_task.set_result(None)
    m.wait_hello_from_server = AsyncMock()
    m.login = AsyncMock()
    m.select = AsyncMock(return_value=("OK", []))
    m.uid = AsyncMock(return_value=("OK", []))
    m.expunge = AsyncMock(return_value=("OK", []))
    m.logout = AsyncMock()
    # No MOVE capability → forces COPY+DELETE path
    m.protocol = MagicMock()
    m.protocol.capabilities = []
    return m


# ---------------------------------------------------------------------------
# Delete → Trash: handler level
# ---------------------------------------------------------------------------


class TestDeleteMovesToTrash:
    """ClassicEmailHandler.delete_emails must move to Trash, not permanently delete."""

    @pytest.mark.asyncio
    async def test_calls_move_emails_not_direct_delete(self, handler):
        """move_emails is called; incoming_client.delete_emails is never called."""
        mock_move = AsyncMock(return_value=(["123", "456"], []))

        with patch.object(handler.incoming_client, "move_emails", mock_move) as _move:
            with patch.object(handler.incoming_client, "delete_emails") as mock_direct_delete:
                result = await handler.delete_emails(["123", "456"], "INBOX")

        mock_move.assert_called_once_with(["123", "456"], "Trash", "INBOX")
        mock_direct_delete.assert_not_called()
        assert result == (["123", "456"], [])

    @pytest.mark.asyncio
    async def test_uses_configured_trash_folder(self):
        """trash_folder_name from config is used as move destination."""
        settings = EmailSettings(
            account_name="test_account",
            full_name="Test User",
            email_address="test@example.com",
            trash_folder_name="[Gmail]/Trash",
            incoming=EmailServer(user_name="u", password="p", host="imap.gmail.com", port=993, use_ssl=True),
            outgoing=EmailServer(user_name="u", password="p", host="smtp.gmail.com", port=465, use_ssl=True),
        )
        h = ClassicEmailHandler(settings)
        mock_move = AsyncMock(return_value=(["42"], []))

        with patch.object(h.incoming_client, "move_emails", mock_move):
            await h.delete_emails(["42"], "INBOX")

        mock_move.assert_called_once_with(["42"], "[Gmail]/Trash", "INBOX")

    @pytest.mark.asyncio
    async def test_defaults_to_trash_when_not_configured(self):
        """When trash_folder_name is None, 'Trash' is the fallback."""
        settings = EmailSettings(
            account_name="test_account",
            full_name="Test User",
            email_address="test@example.com",
            # trash_folder_name not set → None → fallback "Trash"
            incoming=EmailServer(user_name="u", password="p", host="imap.example.com", port=993, use_ssl=True),
            outgoing=EmailServer(user_name="u", password="p", host="smtp.example.com", port=465, use_ssl=True),
        )
        h = ClassicEmailHandler(settings)
        mock_move = AsyncMock(return_value=(["1"], []))

        with patch.object(h.incoming_client, "move_emails", mock_move):
            await h.delete_emails(["1"], "INBOX")

        mock_move.assert_called_once_with(["1"], "Trash", "INBOX")


# ---------------------------------------------------------------------------
# Delete → Trash: IMAP command level
# ---------------------------------------------------------------------------


class TestDeleteImapCommands:
    """At the IMAP level, move_emails issues COPY to Trash before marking Deleted."""

    @pytest.mark.asyncio
    async def test_copy_issued_before_deleted_flag(self, handler, mock_imap):
        """COPY to Trash must appear before any +FLAGS \\Deleted in the UID call sequence."""
        with patch.object(handler.incoming_client, "_imap_connect", return_value=mock_imap):
            await handler.delete_emails(["123"], "INBOX")

        uid_calls = mock_imap.uid.call_args_list
        # All positional-arg strings for each uid() call
        call_args = [tuple(str(a) for a in c.args) for c in uid_calls]

        copy_indices = [i for i, a in enumerate(call_args) if a[0].lower() == "copy"]
        delete_flag_indices = [
            i for i, a in enumerate(call_args)
            if a[0].lower() == "store" and r"(\Deleted)" in " ".join(a)
        ]

        # A COPY to the Trash folder must have been issued
        assert copy_indices, "Expected at least one uid('copy', ...) call"
        assert any("Trash" in " ".join(call_args[i]) for i in copy_indices), (
            "COPY must target the Trash folder"
        )

        # If \Deleted was set, it must come AFTER the COPY
        for del_idx in delete_flag_indices:
            assert any(cp_idx < del_idx for cp_idx in copy_indices), (
                "\\Deleted flag must only be set after a successful COPY"
            )

    @pytest.mark.asyncio
    async def test_imap_delete_emails_method_bypassed(self, handler, mock_imap):
        """EmailClient.delete_emails (which does raw EXPUNGE) is never called directly."""
        with patch.object(handler.incoming_client, "_imap_connect", return_value=mock_imap):
            # Patch delete_emails on the EmailClient to detect if it's called
            with patch.object(handler.incoming_client, "delete_emails") as mock_raw_delete:
                await handler.delete_emails(["123"], "INBOX")

        mock_raw_delete.assert_not_called()


# ---------------------------------------------------------------------------
# Delete → Trash: MCP tool level
# ---------------------------------------------------------------------------


class TestDeleteMcpTool:
    """MCP delete_emails tool moves to Trash end-to-end."""

    @pytest.mark.asyncio
    async def test_mcp_delete_moves_to_trash(self, handler):
        """MCP delete_emails tool results in move_emails to Trash, not direct delete."""
        mock_move = AsyncMock(return_value=(["12345"], []))

        with patch("mcp_email_server.app.dispatch_handler", return_value=handler):
            with patch.object(handler.incoming_client, "move_emails", mock_move):
                result = await delete_emails(
                    account_name="test_account",
                    email_ids=["12345"],
                )

        mock_move.assert_called_once_with(["12345"], "Trash", "INBOX")
        assert "deleted 1 email" in result

    @pytest.mark.asyncio
    async def test_mcp_delete_never_calls_imap_raw_delete(self, handler):
        """No call to EmailClient.delete_emails (permanent delete) from MCP tool."""
        mock_move = AsyncMock(return_value=(["12345"], []))

        with patch("mcp_email_server.app.dispatch_handler", return_value=handler):
            with patch.object(handler.incoming_client, "move_emails", mock_move):
                with patch.object(handler.incoming_client, "delete_emails") as mock_raw_delete:
                    await delete_emails(
                        account_name="test_account",
                        email_ids=["12345"],
                    )

        mock_raw_delete.assert_not_called()


# ---------------------------------------------------------------------------
# Send → Drafts: handler level
# ---------------------------------------------------------------------------


class TestSendGoesToDrafts:
    """ClassicEmailHandler.send_email saves to Drafts by default (SEND_TO_DRAFTS=True)."""

    @pytest.mark.asyncio
    async def test_smtp_not_called_by_default(self, handler, monkeypatch):
        """When SEND_TO_DRAFTS is unset (defaults True), SMTP is never used."""
        monkeypatch.delenv("SEND_TO_DRAFTS", raising=False)
        mock_drafts = AsyncMock(return_value=True)

        with patch.object(handler.outgoing_client, "save_to_drafts", mock_drafts):
            with patch("aiosmtplib.SMTP") as mock_smtp_cls:
                await handler.send_email(
                    recipients=["r@example.com"],
                    subject="Test",
                    body="Hello",
                )

        mock_smtp_cls.assert_not_called()
        mock_drafts.assert_called_once()

    @pytest.mark.asyncio
    async def test_smtp_not_called_when_env_true(self, handler, monkeypatch):
        """When SEND_TO_DRAFTS=True explicitly, SMTP is not used."""
        monkeypatch.setenv("SEND_TO_DRAFTS", "True")
        mock_drafts = AsyncMock(return_value=True)

        with patch.object(handler.outgoing_client, "save_to_drafts", mock_drafts):
            with patch("aiosmtplib.SMTP") as mock_smtp_cls:
                await handler.send_email(
                    recipients=["r@example.com"],
                    subject="Test",
                    body="Hello",
                )

        mock_smtp_cls.assert_not_called()
        mock_drafts.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_to_drafts_receives_correct_folder(self, handler, monkeypatch):
        """save_to_drafts is called with the configured draft_folder_name."""
        monkeypatch.delenv("SEND_TO_DRAFTS", raising=False)
        mock_drafts = AsyncMock(return_value=True)

        with patch.object(handler.outgoing_client, "save_to_drafts", mock_drafts):
            await handler.send_email(
                recipients=["r@example.com"],
                subject="Test",
                body="Hello",
            )

        call_args = mock_drafts.call_args
        # Third positional arg (or kwarg) is draft_folder_name
        draft_folder = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get("draft_folder_name")
        assert draft_folder == "Drafts"

    @pytest.mark.asyncio
    async def test_smtp_used_when_env_false(self, handler, monkeypatch):
        """When SEND_TO_DRAFTS=False, email is sent via SMTP (not Drafts)."""
        monkeypatch.setenv("SEND_TO_DRAFTS", "False")
        mock_smtp = AsyncMock()
        mock_smtp.__aenter__.return_value = mock_smtp
        mock_smtp.__aexit__.return_value = None
        mock_smtp.login = AsyncMock()
        mock_smtp.send_message = AsyncMock()

        with patch.object(handler.outgoing_client, "save_to_drafts") as mock_drafts:
            with patch("aiosmtplib.SMTP", return_value=mock_smtp):
                await handler.send_email(
                    recipients=["r@example.com"],
                    subject="Test",
                    body="Hello",
                )

        mock_smtp.send_message.assert_called_once()
        mock_drafts.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_content_preserved_in_draft(self, handler, monkeypatch):
        """The message saved to Drafts contains correct subject, recipient, and body."""
        monkeypatch.delenv("SEND_TO_DRAFTS", raising=False)
        saved_msg = None

        async def capture_draft(msg, incoming_server, draft_folder_name=None):
            nonlocal saved_msg
            saved_msg = msg
            return True

        with patch.object(handler.outgoing_client, "save_to_drafts", side_effect=capture_draft):
            await handler.send_email(
                recipients=["recipient@example.com"],
                subject="Important Subject",
                body="Body text",
            )

        assert saved_msg is not None
        assert saved_msg["To"] == "recipient@example.com"
        assert "Important Subject" in str(saved_msg["Subject"])


# ---------------------------------------------------------------------------
# Send → Drafts: MCP tool level
# ---------------------------------------------------------------------------


class TestSendMcpTool:
    """MCP send_email tool routes to Drafts by default."""

    @pytest.mark.asyncio
    async def test_mcp_send_goes_to_drafts_by_default(self, handler, monkeypatch):
        """MCP send_email tool: SMTP not called, draft saved, when SEND_TO_DRAFTS unset."""
        monkeypatch.delenv("SEND_TO_DRAFTS", raising=False)
        mock_drafts = AsyncMock(return_value=True)

        with patch("mcp_email_server.app.dispatch_handler", return_value=handler):
            with patch.object(handler.outgoing_client, "save_to_drafts", mock_drafts):
                with patch("aiosmtplib.SMTP") as mock_smtp_cls:
                    await send_email(
                        account_name="test_account",
                        recipients=["r@example.com"],
                        subject="Test",
                        body="Hello",
                    )

        mock_smtp_cls.assert_not_called()
        mock_drafts.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_send_uses_smtp_when_disabled(self, handler, monkeypatch):
        """MCP send_email tool: SMTP called when SEND_TO_DRAFTS=False."""
        monkeypatch.setenv("SEND_TO_DRAFTS", "False")
        mock_smtp = AsyncMock()
        mock_smtp.__aenter__.return_value = mock_smtp
        mock_smtp.__aexit__.return_value = None
        mock_smtp.login = AsyncMock()
        mock_smtp.send_message = AsyncMock()

        with patch("mcp_email_server.app.dispatch_handler", return_value=handler):
            with patch.object(handler.outgoing_client, "save_to_drafts") as mock_drafts:
                with patch("aiosmtplib.SMTP", return_value=mock_smtp):
                    await send_email(
                        account_name="test_account",
                        recipients=["r@example.com"],
                        subject="Test",
                        body="Hello",
                    )

        mock_smtp.send_message.assert_called_once()
        mock_drafts.assert_not_called()
