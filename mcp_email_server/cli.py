import os

import typer

from mcp_email_server.app import mcp
from mcp_email_server.config import delete_settings

app = typer.Typer()


@app.command()
def stdio():
    mcp.run(transport="stdio")


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


@app.command()
def sse(
    host: str = "localhost",
    port: int = 9557,
):
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="sse")


@app.command()
def streamable_http(
    host: str = os.environ.get("MCP_HOST", "localhost"),
    port: int = os.environ.get("MCP_PORT", 9557),
):
    mcp.settings.host = host
    mcp.settings.port = port
    allowed_hosts = os.environ.get("MCP_ALLOWED_HOSTS")
    if allowed_hosts:
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.transport_security = TransportSecuritySettings(
            allowed_hosts=allowed_hosts.split(","),
        )
    elif host not in _LOOPBACK:
        mcp.settings.transport_security = None
    mcp.run(transport="streamable-http")


@app.command()
def ui():
    from mcp_email_server.ui import main as ui_main

    ui_main()


@app.command()
def reset():
    delete_settings()
    typer.echo("✅ Config reset")


if __name__ == "__main__":
    app(["stdio"])
