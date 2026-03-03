import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from dotenv import load_dotenv

# MCP Server Imports
import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# ADK Tool Imports
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.load_web_page import load_web_page
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

load_dotenv()

logger = logging.getLogger(__name__)

# --- Prepare the ADK Tool ---
logger.info("Initializing ADK load_web_page tool...")
adk_tool_to_expose = FunctionTool(load_web_page)
logger.info(f"ADK tool '{adk_tool_to_expose.name}' initialized.")


def create_mcp_server() -> Server:
    """Create and configure the MCP server wrapping ADK tools."""
    app = Server("adk-tool-exposing-mcp-server")

    @app.list_tools()
    async def list_mcp_tools() -> list[mcp_types.Tool]:
        logger.info("MCP Server: Received list_tools request.")
        mcp_tool_schema = adk_to_mcp_tool_type(adk_tool_to_expose)
        logger.info(f"MCP Server: Advertising tool: {mcp_tool_schema.name}")
        return [mcp_tool_schema]

    @app.call_tool()
    async def call_mcp_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[mcp_types.ContentBlock]:
        logger.info(f"MCP Server: call_tool '{name}' args={arguments}")

        if name == adk_tool_to_expose.name:
            try:
                adk_tool_response = await adk_tool_to_expose.run_async(
                    args=arguments,
                    tool_context=None,
                )
                logger.info(f"MCP Server: ADK tool '{name}' executed successfully.")
                return [mcp_types.TextContent(
                    type="text",
                    text=json.dumps(adk_tool_response, indent=2)
                )]
            except Exception as e:
                logger.error(f"MCP Server: Error executing tool '{name}': {e}")
                return [mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
                )]
        else:
            logger.warning(f"MCP Server: Unknown tool '{name}'.")
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"error": f"Tool '{name}' not implemented by this server."})
            )]

    return app


def create_starlette_app(port: int = 8080, json_response: bool = False) -> Starlette:
    """Build the ASGI app with Streamable HTTP session manager."""
    mcp_server = create_mcp_server()

    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=json_response,
        stateless=True,  # Stateless = safe for multi-instance / Cloud Run deployments
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info("MCP Streamable HTTP server started and ready.")
            try:
                yield
            finally:
                logger.info("MCP Streamable HTTP server shutting down.")

    return Starlette(
        debug=False,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    port = int(__import__("os").environ.get("PORT", 8080))
    starlette_app = create_starlette_app(port=port)

    logger.info(f"Launching MCP Streamable HTTP server on port {port}...")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
