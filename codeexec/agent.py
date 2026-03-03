import os
import docker
import sys
from google.adk.agents import Agent
from google.adk.models import LiteLlm

# 1. Point to your local Ollama
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

# 2. Setup the Docker Sandbox logic
client = docker.from_env()
CONTAINER_NAME = "math_executor_sandbox"

def get_or_create_container():
    try:
        return client.containers.get(CONTAINER_NAME)
    except:
        return client.containers.run(
            "math-sandbox", 
            detach=True, 
            name=CONTAINER_NAME,
            network_mode="none" 
        )

# 3. Define the tool as a plain function (ADK handles the rest)
def docker_python_exec(code: str) -> str:
    """
    Executes Python code inside a secure, isolated Docker container.
    Always use print() to see your results.
    
    Args:
        code (str): The Python code to run.
    """
    container = get_or_create_container()
    
    # Escape code for shell execution
    escaped_code = code.replace("'", "'\\''") 
    cmd = f"python3 -c '{escaped_code}'"
    
    exec_result = container.exec_run(cmd)
    output = exec_result.output.decode('utf-8')
    
    if exec_result.exit_code != 0:
        return f"Error (Exit {exec_result.exit_code}): {output}"
    return output if output.strip() else "Success (no output)."

# 4. Initialize the Agent with Ollama + Docker Tool
root_agent = Agent(
    name="math_commander",
    # Using ollama_chat/ is recommended for better tool-calling support
    model=LiteLlm(model="ollama_chat/mistral-small3.2"),
    instruction="""You are a math assistant. 
    To solve problems, write Python code and run it using 'docker_python_exec'.
    Make sure your code prints the final answer clearly.
    """,
    tools=[docker_python_exec], # Just pass the function here
)
