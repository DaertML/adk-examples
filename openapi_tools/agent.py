import asyncio
import uuid
import json
import threading
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# --- Mock Server Imports ---
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

# --- Constants ---
APP_NAME_OPENAPI = "openapi_petstore_app"
USER_ID_OPENAPI = "user_openapi_1"
SESSION_ID_OPENAPI = f"session_openapi_{uuid.uuid4()}"
AGENT_NAME_OPENAPI = "petstore_manager_agent"

MOCK_SERVER_HOST = "127.0.0.1"
MOCK_SERVER_PORT = 8199
MOCK_SERVER_URL = f"http://{MOCK_SERVER_HOST}:{MOCK_SERVER_PORT}"

# --- In-memory pet store ---
_pets: dict[int, dict] = {}
_next_id = 1


# ---------------------------------------------------------------------------
# Mock HTTP server (Starlette)
# ---------------------------------------------------------------------------

async def list_pets(request: Request) -> JSONResponse:
    """GET /pets — return all pets, optionally filtered by status/limit."""
    limit = request.query_params.get("limit")
    status = request.query_params.get("status")

    pets = list(_pets.values())

    if status:
        pets = [p for p in pets if p.get("status") == status]
    if limit:
        pets = pets[:int(limit)]

    return JSONResponse({"pets": pets, "total": len(pets)})


async def create_pet(request: Request) -> JSONResponse:
    """POST /pets — create a new pet."""
    global _next_id
    body = await request.json()

    if "name" not in body:
        return JSONResponse({"error": "Field 'name' is required."}, status_code=400)

    pet = {
        "id": _next_id,
        "name": body["name"],
        "tag": body.get("tag", None),
        "status": body.get("status", "available"),
    }
    _pets[_next_id] = pet
    _next_id += 1

    return JSONResponse(pet, status_code=201)


async def show_pet_by_id(request: Request) -> JSONResponse:
    """GET /pets/{petId} — return a single pet."""
    pet_id = int(request.path_params["petId"])
    pet = _pets.get(pet_id)
    if pet is None:
        return JSONResponse({"error": f"Pet with ID {pet_id} not found."}, status_code=404)
    return JSONResponse(pet)


mock_app = Starlette(
    debug=False,
    routes=[
        Route("/pets",          list_pets,      methods=["GET"]),
        Route("/pets",          create_pet,     methods=["POST"]),
        Route("/pets/{petId}",  show_pet_by_id, methods=["GET"]),
    ],
)


def run_mock_server():
    """Run the mock Starlette server in a background thread."""
    config = uvicorn.Config(
        mock_app,
        host=MOCK_SERVER_HOST,
        port=MOCK_SERVER_PORT,
        log_level="warning",  # keep output clean
    )
    server = uvicorn.Server(config)
    server.run()


# ---------------------------------------------------------------------------
# OpenAPI spec — points at our local mock server
# ---------------------------------------------------------------------------

openapi_spec_string = f"""
{{
  "openapi": "3.0.0",
  "info": {{
    "title": "Simple Pet Store API (Mock)",
    "version": "1.0.0",
    "description": "An API to manage pets in a store, backed by a local mock server."
  }},
  "servers": [
    {{
      "url": "{MOCK_SERVER_URL}",
      "description": "Local mock server"
    }}
  ],
  "paths": {{
    "/pets": {{
      "get": {{
        "summary": "List all pets",
        "operationId": "listPets",
        "description": "Returns all pets, with optional filtering.",
        "parameters": [
          {{
            "name": "limit",
            "in": "query",
            "description": "Maximum number of pets to return",
            "required": false,
            "schema": {{ "type": "integer", "format": "int32" }}
          }},
          {{
            "name": "status",
            "in": "query",
            "description": "Filter pets by status",
            "required": false,
            "schema": {{ "type": "string", "enum": ["available", "pending", "sold"] }}
          }}
        ],
        "responses": {{
          "200": {{
            "description": "A list of pets.",
            "content": {{ "application/json": {{ "schema": {{ "type": "object" }} }} }}
          }}
        }}
      }},
      "post": {{
        "summary": "Create a pet",
        "operationId": "createPet",
        "description": "Adds a new pet to the store.",
        "requestBody": {{
          "description": "Pet object to add",
          "required": true,
          "content": {{
            "application/json": {{
              "schema": {{
                "type": "object",
                "required": ["name"],
                "properties": {{
                  "name": {{"type": "string", "description": "Name of the pet"}},
                  "tag":  {{"type": "string", "description": "Optional tag for the pet"}},
                  "status": {{
                    "type": "string",
                    "enum": ["available", "pending", "sold"],
                    "description": "Pet availability status"
                  }}
                }}
              }}
            }}
          }}
        }},
        "responses": {{
          "201": {{
            "description": "Pet created successfully.",
            "content": {{ "application/json": {{ "schema": {{ "type": "object" }} }} }}
          }}
        }}
      }}
    }},
    "/pets/{{petId}}": {{
      "get": {{
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "description": "Returns info for a single pet by its ID.",
        "parameters": [
          {{
            "name": "petId",
            "in": "path",
            "description": "The ID of the pet to retrieve",
            "required": true,
            "schema": {{ "type": "integer", "format": "int64" }}
          }}
        ],
        "responses": {{
          "200": {{
            "description": "Information about the pet.",
            "content": {{ "application/json": {{ "schema": {{ "type": "object" }} }} }}
          }},
          "404": {{ "description": "Pet not found" }}
        }}
      }}
    }}
  }}
}}
"""

# ---------------------------------------------------------------------------
# OpenAPIToolset + Agent
# ---------------------------------------------------------------------------

petstore_toolset = OpenAPIToolset(
    spec_str=openapi_spec_string,
    spec_str_type='json',
)

model_ollama = LiteLlm(
    # qwen2.5 and llama3.1 have the most reliable tool/function-call support in Ollama.
    # mistral-small3.2 returns empty parts=[] when tools are present — do not use it here.
    # To switch models: pull first with `ollama pull qwen2.5` or `ollama pull llama3.1`
    model="ollama_chat/qwen2.5:0.5b",
    api_base="http://localhost:11434",
)

root_agent = LlmAgent(
    name=AGENT_NAME_OPENAPI,
    model=model_ollama,
    tools=[petstore_toolset],
    instruction="""You are a Pet Store assistant. You MUST use the provided tools to answer every request.
    Never answer from memory — always call the appropriate tool first, then summarise the result.

    Tool usage rules:
    - To add a pet: call createPet with the name and any optional tag/status fields.
    - To list pets: call listPets, passing limit or status only when the user specifies them.
    - To look up a pet by ID: call showPetById with the numeric petId.

    After the tool returns, reply in plain English summarising what happened.
    """,
    description="Manages a Pet Store using tools generated from an OpenAPI spec.",
)

# ---------------------------------------------------------------------------
# Session / Runner
# ---------------------------------------------------------------------------

async def setup_session_and_runner():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME_OPENAPI,
        session_service=session_service,
    )
    await session_service.create_session(
        app_name=APP_NAME_OPENAPI,
        user_id=USER_ID_OPENAPI,
        session_id=SESSION_ID_OPENAPI,
    )
    return runner


async def call_openapi_agent_async(query: str, runner: Runner):
    print("\n--- Running OpenAPI Pet Store Agent ---")
    print(f"Query: {query}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = None
    last_text = None  # fallback: track the most recent text we've seen

    try:
        async for event in runner.run_async(
            user_id=USER_ID_OPENAPI,
            session_id=SESSION_ID_OPENAPI,
            new_message=content,
        ):
            # Log tool calls
            if event.get_function_calls():
                call = event.get_function_calls()[0]
                print(f"  Agent Action: Called '{call.name}' with args {call.args}")
                continue

            # Log tool responses
            if event.get_function_responses():
                response = event.get_function_responses()[0]
                print(f"  Agent Action: Received response for '{response.name}'")
                continue

            # Capture any text content from the agent (author != 'user')
            if event.content and event.content.parts and event.author != 'user':
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text and part.text.strip():
                        last_text = part.text.strip()

            # Prefer explicitly flagged final response
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text and part.text.strip():
                        final_response_text = part.text.strip()
                        break

        # Fall back to last seen text if final response wasn't flagged
        result = final_response_text or last_text or "Agent did not produce a text response."
        print(f"Agent Final Response: {result}")

    except Exception as e:
        print(f"An error occurred during agent run: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 30)


async def run_openapi_example():
    runner = await setup_session_and_runner()

    await call_openapi_agent_async("Add a dog named 'Dukey' with tag 'lab'.", runner)
    await call_openapi_agent_async("Add a cat named 'Whiskers' with status 'pending'.", runner)
    await call_openapi_agent_async("Show me all available pets.", runner)
    await call_openapi_agent_async("Get info for pet with ID 1.", runner)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting mock Pet Store HTTP server on {MOCK_SERVER_URL} ...")
    server_thread = threading.Thread(target=run_mock_server, daemon=True)
    server_thread.start()

    # Give the server a moment to start
    import time
    time.sleep(1)
    print("Mock server ready.\n")

    print("Executing OpenAPI + Ollama example...")
    try:
        asyncio.run(run_openapi_example())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Info: Already inside a running event loop (e.g. Jupyter). Use `await run_openapi_example()` instead.")
        else:
            raise
    print("OpenAPI example finished.")
