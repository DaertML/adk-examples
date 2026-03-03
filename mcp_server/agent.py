import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Use an absolute path so the script is found regardless of working directory
PATH_TO_YOUR_MCP_SERVER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "adk_mcp_server.py"
)

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
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='python3',
                    args=[PATH_TO_YOUR_MCP_SERVER_SCRIPT],
                )
            )
        )
    ],
)
