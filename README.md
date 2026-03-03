# adk-examples
Repo with examples of ADK agents

# Introduction
I gave ADK a try, as I have seen it in production in some middle sized companies, and it does not have so much hype behind lately... so wanted to see what it could do, how it would do it and see if I can use it without using Google products.

The other reason I wanted to test ADK is due to the need to run agents remotely and communicate them over the network; for such A2A was created by Google; even though other agent frameworks support it, I didnt have much luck with them and did not work as I would expect in many occasions, thus the need to try the Google ADK framework.

Some of the agents are meant to be used from the "adk web" web UI, I expect you know how to use ADK before playing with this. Many of the examples are copy/pasted/adapted from the Google's tutorials. Each example is explained from a high level point of view next, and I mention which are the ones that had more changes to work with local Ollama environments.

Some of the projects (those related with A2A are meant to be used together); this is also explained below; for the rest, they are individual examples of agents to run.

I only encountered two points that scared me of being tied to Google: the execution sandbox and the reasoning; those are easily avoided as you will see in the examples.

I leave the examples as the most "template form" as possible and simplest that you can grow your agent from; as it is usually the case that having complex examples make it hard to evolve into other use cases.

# Examples and their use case
- a2a_consumer: agent that uses multiple remote agents using the A2A protocol;such agents need to be running. a version with a single remote agent is also provided.
- a2a_consumer_multiple: same idea as the a2a_consumer but with multiple agents. kind of replicated code, sorry :S
- a2aserver_pydantic: simple example of an a2aserver that runs an agent that is consumed by the projects a2aconsumer mentioned above.
- codeexec: implements a simple python sandbox that is used by the LLM to run code and get the answer from user requests. The sandbox runs on a docker container to avoid damage caused by the agent to your machine.
- devteam: simple example of an agentic workflow, in which data is transformed by multiple chained LLM calls; that attempt to write and review code.
- mcp_client: example of mcp client running on docker. This is meant to be used to call the mcp_server tools exposed repo.
- mcp_server: example of an MCP server implemented with ADK, meant to be consumed by the mcp_client.
- mcp_server_remote: example of an MCP server that is meant to be executed remotely and consumed by MCP clients.
- mcptools: simple example that uses a local MCP server (stdio); that exports tools to do operations in the filesystem.
- multi_agent: example of a multi agent environment
- multi_agent_web: example like the previous mentioned; this time, meant to run from the web ui of the "adk web"
- openapi_tools: example that convers an OpenAPI definition into tools for an agent and makes use of them to resolve user queries.
- react: unfinished agent that should implement the ReACT agentic pattern. sorry :S
- simple_agent: the first example that you would find for an agent use case with simple tool calls; adapted to work on Ollama.
- tool_state_web: meant to be executed using "adk web", it also contains a test suite to run from the cli of ADK. and a command to serve this as an A2A server that is later consumed by other agents. It manages state inside the tools, which is something surprising to me when I tried ADK, I did not store state before in tools, and was more of something to store in a single centralized KeyValue store.
- tool_state_web_guard: same as the previous example, it contains guardrails using llama-guard, all from ollama, all running locally.
- ttc_codeexec: simple example that tries to mimic how "claude code" like agents work; by giving it a file with tasks and giving it tools to edit/read/write files.
