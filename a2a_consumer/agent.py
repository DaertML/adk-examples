from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent

# Construct the URL properly using the ADK constant
# This results in: http://localhost:8001/.well-known/agent-card.json
REMOTE_CARD_URL = f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}"

root_agent = RemoteA2aAgent(
    name="weather_consumer",
    description="I am a consumer agent that talks to the Weather Service.",
    agent_card=REMOTE_CARD_URL
)
