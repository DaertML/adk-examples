import os
from google.adk.agents import Agent  # Use standard Agent for LiteLlm support
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# 1. Setup your absolute path
TARGET_FOLDER_PATH = os.path.abspath("/app/mcp_files")

# 2. Define the Agent using LiteLlm for Ollama
root_agent = Agent(
    # Ensure the model name matches what you have pulled in Ollama
    model=LiteLlm(model="ollama_chat/mistral-small3.2"), 
    name='filesystem_assistant_agent',
    instruction='Help the user manage their files. You can list files, read files, etc.',
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='npx',
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        TARGET_FOLDER_PATH,
                    ],
                ),
            ),
        )
    ],
)
