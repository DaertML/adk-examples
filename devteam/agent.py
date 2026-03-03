import os
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models import LiteLlm

# --- 1. Point to your local Ollama instance ---
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

# Use your preferred local model (e.g., mistral, llama3, or codellama)
OLLAMA_MODEL = LiteLlm(model="ollama_chat/mistral-small3.2")

# --- 2. Define Sub-Agents for Each Pipeline Stage ---

# Code Writer Agent
code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model=OLLAMA_MODEL,
    instruction="""
    You are a Python Code Generator.
    Based *only* on the user's request, write Python code that fulfills the requirement.
    Output *only* the complete Python code block, enclosed in triple backticks (```python ... ```).
    Do not add any other text before or after the code block.
    """,
    description="Writes initial Python code based on a specification.",
    output_key="generated_code"
)

# Code Reviewer Agent
code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model=OLLAMA_MODEL,
    instruction="""
    You are an expert Python Code Reviewer.
    Your task is to provide constructive feedback on the provided code.

    **Code to Review:**
    ```python
    {generated_code}
    ```

    **Review Criteria:**
    1. Correctness: Does the code work as intended?
    2. Readability: Follows PEP 8?
    3. Efficiency: Any bottlenecks?
    4. Edge Cases: Handles invalid inputs?

    **Output:**
    Provide your feedback as a concise, bulleted list. 
    If the code is excellent, state: "No major issues found."
    """,
    description="Reviews code and provides feedback.",
    output_key="review_comments"
)

# Code Refactorer Agent
code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model=OLLAMA_MODEL,
    instruction="""
    You are a Python Code Refactoring AI.
    Improve the code based on the feedback.

    **Original Code:**
    ```python
    {generated_code}
    ```

    **Review Comments:**
    {review_comments}

    **Output:**
    Output *only* the final, refactored code block in triple backticks.
    """,
    description="Refactors code based on review comments.",
    output_key="refactored_code"
)

# --- 3. Create the SequentialAgent ---
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
)

root_agent = code_pipeline_agent
