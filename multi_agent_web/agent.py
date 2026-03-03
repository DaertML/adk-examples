import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# 1. Define the Shared Model
# Note: Ensure Ollama is running locally
OLLAMA_MODEL = LiteLlm(model="ollama_chat/mistral-small3.2")

# --- STEP 1: Define Tools ---
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city."""
    if city.lower() == "new york":
        return {"status": "success", "report": "The weather in New York is sunny and 25°C."}
    return {"status": "error", "error_message": f"Weather for '{city}' not available."}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    if city.lower() == "new york":
        tz = ZoneInfo("America/New_York")
        now = datetime.datetime.now(tz)
        return {"status": "success", "report": f"Current time: {now.strftime('%H:%M:%S')}"}
    return {"status": "error", "error_message": "Timezone not found."}

def say_hello(name: Optional[str] = None) -> str:
    """Provides a simple greeting."""
    return f"Hello, {name}!" if name else "Hello there!"

def say_goodbye() -> str:
    """Provides a simple farewell message."""
    return "Goodbye! Have a great day."

# --- STEP 2: Define Sub-Agents ---
greeting_agent = Agent(
    model=OLLAMA_MODEL,
    name="greeting_agent",
    instruction="Your ONLY task is to greet the user using the 'say_hello' tool.",
    tools=[say_hello],
)

farewell_agent = Agent(
    model=OLLAMA_MODEL,
    name="farewell_agent",
    instruction="Your ONLY task is to say goodbye using the 'say_goodbye' tool.",
    tools=[say_goodbye],
)

# --- STEP 3: Define Root Agent ---
# The ADK Web CLI specifically looks for a variable named 'root_agent'
root_agent = Agent(
    name="weather_coordinator",
    model=OLLAMA_MODEL,
    instruction=(
        "You are the lead coordinator. "
        "1. If it's a greeting, delegate to 'greeting_agent'. "
        "2. If it's a farewell, delegate to 'farewell_agent'. "
        "3. If it's weather or time, handle it yourself using your tools."
    ),
    tools=[get_weather, get_current_time],
    sub_agents=[greeting_agent, farewell_agent]
)
