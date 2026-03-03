from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent

# Construct the URL properly using the ADK constant
# This results in: http://localhost:8001/.well-known/agent-card.json
WEATHER_CARD_URL = f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}"
DICE_CARD_URL = f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}"

weather_agent = RemoteA2aAgent(
    name="weather_service",
    description="Provides real-time weather updates and forecasts.",
    agent_card=WEATHER_CARD_URL
)

dice_agent = RemoteA2aAgent(
    name="dice_service",
    description="Provides random number generator for dice rolling.",
    agent_card=DICE_CARD_URL
)

root_agent = RemoteA2aAgent(
    name="weather_consumer",
    description="I am a consumer agent that talks to the Weather Service.",
    sub_agents=[weather_agent, dice_agent]
)
