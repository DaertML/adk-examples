import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm
from google.adk.planners import PlanReActPlanner

# --- 1. Define Tools ---

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.
    Args:
        city (str): The name of the city.
    """
    if city.lower() == "new york":
        return {"status": "success", "report": "Sunny, 25°C (77°F)."}
    return {"status": "error", "error_message": f"No data for {city}."}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    mapping = {"new york": "America/New_York", "london": "Europe/London"}
    tz_name = mapping.get(city.lower())
    
    if not tz_name:
        return {"status": "error", "error_message": "Timezone unknown."}
    
    now = datetime.datetime.now(ZoneInfo(tz_name))
    return {"status": "success", "report": now.strftime("%Y-%m-%d %H:%M:%S")}

# --- 2. Configure the Model ---
# We use LiteLlm to bridge to Ollama. 
# "ollama_chat/" prefix ensures proper tool-calling support.
local_model = LiteLlm(model="ollama_chat/mistral-small3.2") # Qwen2.5 is excellent for tools

# --- 3. Define the Root Agent ---
# The ADK Web UI specifically looks for the variable 'root_agent'
root_agent = LlmAgent(
    name="weather_and_time_agent",
    model=local_model,
    instruction="You are a helpful assistant. Use tools to find weather and time.",
    planner=PlanReActPlanner(), # Enables multi-step reasoning for local models
    tools=[get_weather, get_current_time]
)
