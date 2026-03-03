"""
Weather Agent with LLM-Based Guardrails
- Llama Guard for input safety validation
- LLM-based location validation before weather lookup
"""

import datetime
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types
import json

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

# Main agent model
OLLAMA_MODEL = LiteLlm(model="ollama_chat/mistral-small3.2")

# Llama Guard for safety checks
LLAMA_GUARD_MODEL = LiteLlm(model="ollama_chat/llama-guard3")

# LLM for location validation
LOCATION_VALIDATOR_MODEL = LiteLlm(model="ollama_chat/mistral-small3.2")

print("✅ Models configured:")
print(f"   - Main Agent: mistral-small3.2")
print(f"   - Safety Guard: llama-guard3")
print(f"   - Location Validator: mistral-small3.2")


# ============================================================================
# GUARDRAIL 1: LLAMA GUARD - INPUT SAFETY VALIDATION
# ============================================================================

def llama_guard_safety_check(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Uses Llama Guard to check if user input is safe before processing.
    
    Llama Guard is specifically trained to detect:
    - Harmful content
    - Malicious prompts
    - Jailbreak attempts
    - Prompt injection
    - Privacy violations
    - Other unsafe content
    
    Returns:
        LlmResponse to block if unsafe, None to allow
    """
    agent_name = callback_context.agent_name
    print(f"\n{'='*60}")
    print(f"🛡️  LLAMA GUARD - Safety Check for agent: {agent_name}")
    print(f"{'='*60}")

    # Extract the last user message
    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break

    print(f"📝 Checking message: '{last_user_message_text[:100]}...'")

    # Prepare prompt for Llama Guard
    # Llama Guard expects specific format for evaluation
    guard_prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections
S14: Code Interpreter Abuse
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {last_user_message_text}
<END CONVERSATION>

Provide your safety assessment for User's message in the following format:
- First line must be only 'safe' or 'unsafe'
- If unsafe, second line must start with the violated category from the list above

[/INST]"""

    try:
        print("🔍 Calling Llama Guard...")
        
        # Call Llama Guard model
        guard_response = LLAMA_GUARD_MODEL.generate_content(guard_prompt)
        
        # Parse response
        if hasattr(guard_response, 'text'):
            guard_result = guard_response.text.strip().lower()
        else:
            guard_result = str(guard_response).strip().lower()
        
        print(f"🛡️  Llama Guard Response: {guard_result[:100]}...")
        
        # Check if content is unsafe
        if guard_result.startswith('unsafe'):
            print("⛔ UNSAFE CONTENT DETECTED - Blocking request")
            
            # Extract violated category if present
            lines = guard_result.split('\n')
            violation_category = lines[1] if len(lines) > 1 else "policy violation"
            
            # Set state flag
            callback_context.state["guardrail_safety_blocked"] = True
            callback_context.state["guardrail_safety_reason"] = violation_category
            print(f"📊 State updated: guardrail_safety_blocked = True")
            print(f"📊 State updated: guardrail_safety_reason = {violation_category}")
            
            # Return blocking response
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(
                        text=f"I cannot process this request as it violates our safety policy ({violation_category}). "
                             f"Please rephrase your request in a safe and appropriate manner."
                    )],
                )
            )
        else:
            print("✅ Content is SAFE - Allowing request to proceed")
            return None
            
    except Exception as e:
        print(f"⚠️  Error during Llama Guard check: {e}")
        print("⚠️  Defaulting to ALLOW (fail-open for availability)")
        # In production, you might want to fail-closed (block) instead
        return None


# ============================================================================
# GUARDRAIL 2: LLM-BASED LOCATION VALIDATION
# ============================================================================

def validate_location_with_llm(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Uses an LLM to validate if the requested location is a real city/country/place
    before executing the weather tool.
    
    This prevents:
    - Invalid location queries
    - Non-geographic entities
    - Nonsense inputs
    - Potentially malicious location strings
    
    Returns:
        Error dict to block if invalid location, None to allow tool execution
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name
    
    print(f"\n{'='*60}")
    print(f"🌍 LOCATION VALIDATOR - Checking tool: {tool_name}")
    print(f"{'='*60}")
    
    # Only validate the weather tool
    if tool_name != "get_weather_stateful":
        print(f"ℹ️  Tool '{tool_name}' is not weather-related - Allowing")
        return None
    
    # Extract city argument
    city_argument = args.get("city", "")
    
    if not city_argument:
        print("⚠️  No city argument provided - Allowing tool to handle error")
        return None
    
    print(f"📍 Validating location: '{city_argument}'")
    
    # Prepare validation prompt
    validation_prompt = f"""You are a geographic location validator. Your task is to determine if the given text represents a real, valid geographic location (city, country, region, or place).

Location to validate: "{city_argument}"

Analyze whether this is:
1. A real city name (e.g., "London", "New York", "Tokyo")
2. A real country name (e.g., "France", "Japan", "Brazil")
3. A real region or place (e.g., "Silicon Valley", "The Alps")
4. A valid geographic location that would have weather data

NOT valid examples:
- Fictional places (e.g., "Hogwarts", "Gotham", "Narnia")
- People's names (e.g., "John Smith")
- Random words or nonsense (e.g., "asdfgh", "12345")
- Non-geographic entities (e.g., "Microsoft", "Pizza")
- Objects or concepts (e.g., "happiness", "table")

Respond in JSON format only:
{{
    "is_valid": true or false,
    "reason": "Brief explanation",
    "location_type": "city" or "country" or "region" or "invalid"
}}

JSON response:"""

    try:
        print("🔍 Calling Location Validator LLM...")
        
        # Call validation model
        validator_response = LOCATION_VALIDATOR_MODEL.generate_content(validation_prompt)
        
        # Parse response
        if hasattr(validator_response, 'text'):
            response_text = validator_response.text.strip()
        else:
            response_text = str(validator_response).strip()
        
        print(f"📝 Raw LLM response: {response_text[:200]}...")
        
        # Try to extract JSON from response
        # Sometimes LLM adds markdown formatting
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        # Parse JSON
        validation_result = json.loads(response_text)
        
        is_valid = validation_result.get("is_valid", False)
        reason = validation_result.get("reason", "Unknown reason")
        location_type = validation_result.get("location_type", "unknown")
        
        print(f"🌍 Validation Result:")
        print(f"   - Is Valid: {is_valid}")
        print(f"   - Location Type: {location_type}")
        print(f"   - Reason: {reason}")
        
        if not is_valid:
            print("⛔ INVALID LOCATION - Blocking tool execution")
            
            # Update state
            tool_context.state["guardrail_location_blocked"] = True
            tool_context.state["guardrail_location_reason"] = reason
            print(f"📊 State updated: guardrail_location_blocked = True")
            
            # Return error matching tool's format
            return {
                "status": "error",
                "error_message": (
                    f"Invalid location: '{city_argument}' does not appear to be a valid "
                    f"geographic location. {reason}. Please provide a real city, country, or place name."
                )
            }
        else:
            print(f"✅ Valid {location_type} - Allowing tool execution")
            return None
            
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing LLM response as JSON: {e}")
        print(f"⚠️  Response was: {response_text[:200]}")
        print("⚠️  Defaulting to ALLOW to avoid false positives")
        return None
        
    except Exception as e:
        print(f"⚠️  Error during location validation: {e}")
        print("⚠️  Defaulting to ALLOW to avoid false positives")
        return None


# ============================================================================
# TOOLS
# ============================================================================

def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather with temperature in user's preferred unit from session state."""
    print(f"\n{'='*60}")
    print(f"🌤️  WEATHER TOOL - Getting weather for {city}")
    print(f"{'='*60}")
    
    # Read preference from state (default to Celsius)
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(f"🌡️  Temperature unit preference: {preferred_unit}")
    
    city_normalized = city.lower().replace(" ", "")
    
    # Mock weather data (stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
        "paris": {"temp_c": 20, "condition": "partly cloudy"},
        "sydney": {"temp_c": 22, "condition": "clear"},
        "dubai": {"temp_c": 35, "condition": "hot and sunny"},
    }
    
    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]
        
        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32
            temp_unit = "°F"
        else:
            temp_value = temp_c
            temp_unit = "°C"
        
        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"✅ Weather report generated: {report}")
        
        # Save last city checked to state
        tool_context.state["last_city_checked"] = city
        print(f"📊 State updated: last_city_checked = {city}")
        
        return result
    else:
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"❌ City not in database: {city}")
        return {"status": "error", "error_message": error_msg}


def set_temperature_unit(unit: str, tool_context: ToolContext) -> dict:
    """Sets the user's preferred temperature unit in session state."""
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
    
    print(f"\n{'='*60}")
    print(f"🌡️  TEMPERATURE UNIT CHANGED: {old_unit} → {unit}")
    print(f"{'='*60}")
    
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
    return "Goodbye! Have a great day!"


# ============================================================================
# SUB-AGENTS
# ============================================================================

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

print("✅ Sub-agents created:")
print(f"   - {greeting_agent.name}")
print(f"   - {farewell_agent.name}")


# ============================================================================
# ROOT AGENT WITH LLM-BASED GUARDRAILS
# ============================================================================

root_agent = Agent(
    name="weather_coordinator_with_llm_guardrails",
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
        "Be attentive to user preferences and provide helpful, accurate information."
    ),
    description=(
        "Main coordinator with advanced LLM-based guardrails: "
        "Llama Guard for safety, location validation, temperature preferences, and delegation."
        "ALWAYS check the location when a user query is related to weather or time."
    ),
    tools=[get_weather_stateful, get_current_time, set_temperature_unit],
    sub_agents=[greeting_agent, farewell_agent],
    output_key="last_response",
    # LLM-based guardrails
    before_model_callback=llama_guard_safety_check,  # Llama Guard for input safety
    before_tool_callback=validate_location_with_llm,  # LLM validates location
)

print("\n" + "="*60)
print("✅ ROOT AGENT CREATED WITH LLM-BASED GUARDRAILS")
print("="*60)
print(f"Agent Name: {root_agent.name}")
print(f"Model: mistral-small3.2")
print(f"\n🛡️  Active Guardrails:")
print(f"   1. Llama Guard 3 - Input safety validation")
print(f"   2. LLM Location Validator - Geographic validation")
print(f"\n📋 Features:")
print(f"   - State-aware temperature units (C°/F°)")
print(f"   - Multi-agent delegation (greetings/farewells)")
print(f"   - Session state persistence")
print(f"   - Intelligent intent detection")
print("="*60)
