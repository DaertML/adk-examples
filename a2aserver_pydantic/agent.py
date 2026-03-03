from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Define your Ollama model
ollama_model = OpenAIChatModel(
    'llama3.2',
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)

# Define your agent
agent = Agent(
    ollama_model,
    name="run_dice",
    system_prompt="Return a random number of a dice.",
)

# Expose as A2A server - url= sets the address published in the agent card
a2a_app = agent.to_a2a(url="http://localhost:8002")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(a2a_app, host="0.0.0.0", port=8002)
