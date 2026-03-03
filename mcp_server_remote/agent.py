import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

# URL of your running MCP server — update host/port to match your deployment
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080/mcp")

model_ollama = LiteLlm(
    model="ollama_chat/mistral-small3.2",
    api_base="http://localhost:11434"
)

root_agent = Agent(
    model=model_ollama,
    name='web_reader_mcp_client_agent',
    instruction="Use the 'load_web_page' tool to fetch content from a URL provided by the user.",
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=MCP_SERVER_URL,
                # headers={"Authorization": "Bearer your-token-here"},  # uncomment for auth
            )
        )
    ],
)
