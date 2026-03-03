import datetime
from typing import Optional
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.adk.a2a.utils.agent_to_a2a import to_a2a


# 1. Define the Shared Model
# Note: Ensure Ollama is running locally
OLLAMA_MODEL = LiteLlm(model="ollama_chat/mistral-small3.2")

# --- STEP 1: Define Tools ---

def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather with temperature in user's preferred unit from session state."""
    print(f"--- Tool: get_weather_stateful called for {city} ---")
    
    # Read preference from state (default to Celsius)
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")
    
    city_normalized = city.lower().replace(" ", "")
    
    # Mock weather data (stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }
    
    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]
        
        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32
            temp_unit = "°F"
        else:  # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"
        
        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")
        
        # Save last city checked to state
        tool_context.state["last_city_checked"] = city
        print(f"--- Tool: Updated state 'last_city_checked': {city} ---")
        
        return result
    else:
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}


def set_temperature_unit(unit: str, tool_context: ToolContext) -> dict:
    """
    Sets the user's preferred temperature unit in session state.
    
    Args:
        unit: Either 'Celsius' or 'Fahrenheit'
        tool_context: Context providing access to session state
        
    Returns:
        Confirmation message about the unit change
    """
    # Normalize input
    unit = unit.strip().capitalize()
    
    if unit not in ["Celsius", "Fahrenheit"]:
        return {
            "status": "error", 
            "message": f"Invalid unit '{unit}'. Please use 'Celsius' or 'Fahrenheit'."
        }
    
    # Update state
    old_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    tool_context.state["user_preference_temperature_unit"] = unit
    
    print(f"--- Tool: set_temperature_unit called. Changed from {old_unit} to {unit} ---")
    
    return {
        "status": "success",
        "message": f"Temperature unit preference updated to {unit}.",
        "previous_unit": old_unit,
        "new_unit": unit
    }


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
    instruction="Your ONLY task is to greet the user using the 'say_hello' tool. If they provide a name, pass it to the tool.",
    description="Handles simple greetings and hellos using the 'say_hello' tool.",
    tools=[say_hello],
)

farewell_agent = Agent(
    model=OLLAMA_MODEL,
    name="farewell_agent",
    instruction="Your ONLY task is to say goodbye using the 'say_goodbye' tool.",
    description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
    tools=[say_goodbye],
)


# --- STEP 3: Define Root Agent ---
# The ADK Web CLI specifically looks for a variable named 'root_agent'

root_agent = Agent(
    name="weather_coordinator_stateful",
    model=OLLAMA_MODEL,
    instruction=(
        "You are the main Weather Coordinator Agent. Your responsibilities:\n\n"
        "1. GREETINGS: If the user says hello/hi/hey, delegate to 'greeting_agent'.\n"
        "2. FAREWELLS: If the user says bye/goodbye/see you, delegate to 'farewell_agent'.\n"
        "3. WEATHER: Use 'get_weather_stateful' tool for weather requests. "
        "The tool automatically formats temperature based on the user's saved preference.\n"
        "4. TEMPERATURE UNIT CHANGES: If the user wants to change temperature units "
        "(e.g., 'use Fahrenheit', 'switch to Celsius', 'show in F', 'I prefer Celsius'), "
        "use the 'set_temperature_unit' tool to update their preference. "
        "Valid units are 'Celsius' or 'Fahrenheit'.\n"
        "5. TIME: Use 'get_current_time' tool for time requests.\n\n"
        "Be attentive to user preferences about temperature units and proactively use "
        "the 'set_temperature_unit' tool when you detect they want to change it."
    ),
    description="Main coordinator: provides weather (with state-aware temperature units), time info, and delegates greetings/farewells.",
    tools=[get_weather_stateful, get_current_time, set_temperature_unit],
    sub_agents=[greeting_agent, farewell_agent],
    output_key="last_response"  # Auto-save agent's final response to state
)

print("✅ Stateful weather agent system initialized successfully!")
print("✅ Temperature preference will be automatically detected and saved to session state.")
print("✅ Try messages like:")
print("   - 'What's the weather in London?'")
print("   - 'Switch to Fahrenheit please'")
print("   - 'Show me New York weather'")
a2a_app = to_a2a(root_agent, port=8001)


