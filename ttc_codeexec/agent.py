"""
Multi-Agent Coding System
=========================
Changes in this version:
  - Full tool-call tracing: every tool call + result is printed with [TOOL] tags.
  - run_shell() executes INSIDE the Docker container (not on the host).
  - pip install runs inside the container via docker exec.
  - Container is pre-checked for pytest on startup; installed if missing.
  - Files are written to WORKSPACE_IN_HOST (volume-mounted → auto-visible in container).
  - Orchestrator classifies errors before each fix call.
"""

import asyncio
import os
import uuid
from functools import wraps
from typing import Callable

import docker
from google.adk.agents import Agent
from google.adk.models import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ─────────────────────────────────────────────
# 0.  Config
# ─────────────────────────────────────────────
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

SANDBOX_IMAGE     = "math-sandbox"
CONTAINER_NAME    = "multi_agent_sandbox"
WORKSPACE_IN_HOST = os.path.abspath("./workspace")
WORKSPACE_IN_CTR  = "/workspace"
MAX_ITERATIONS    = 10
OLLAMA_MODEL      = LiteLlm(model="ollama_chat/llama3.2")

os.makedirs(WORKSPACE_IN_HOST, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  Docker helpers
# ─────────────────────────────────────────────
_docker_client = docker.from_env()


def _get_or_create_container() -> docker.models.containers.Container:
    try:
        c = _docker_client.containers.get(CONTAINER_NAME)
        if c.status != "running":
            c.start()
        return c
    except docker.errors.NotFound:
        print(f"[Docker] Creating container '{CONTAINER_NAME}' from image '{SANDBOX_IMAGE}' …")
        return _docker_client.containers.run(
            SANDBOX_IMAGE,
            detach=True,
            name=CONTAINER_NAME,
            # NOTE: network_mode="none" blocks pip from reaching PyPI.
            # Use "bridge" so the container can install packages,
            # but restrict outbound access at the image/firewall level if needed.
            network_mode="bridge",
            volumes={WORKSPACE_IN_HOST: {"bind": WORKSPACE_IN_CTR, "mode": "rw"}},
            working_dir=WORKSPACE_IN_CTR,
        )


def _exec_in_container(cmd: str) -> tuple[int, str]:
    """
    Execute a bash command INSIDE the running Docker container.
    Returns (exit_code, output).
    """
    container = _get_or_create_container()
    result = container.exec_run(["bash", "-c", cmd], workdir=WORKSPACE_IN_CTR)
    output = result.output.decode("utf-8", errors="replace").strip()
    return result.exit_code, output


def _exec(cmd: str) -> str:
    """Convenience wrapper: run in container, return formatted string."""
    exit_code, output = _exec_in_container(cmd)
    if exit_code != 0:
        return f"[exit {exit_code}]\n{output}"
    return output or "(no output)"


def _ensure_env() -> None:
    """
    Called once at startup.
    Checks that pytest (and other essentials) are available inside the container.
    If not, installs them. This avoids wasting loop iterations on env issues.
    """
    container = _get_or_create_container()
    print("[Orchestrator] Checking container environment …")

    exit_code, out = _exec_in_container("python3 -m pytest --version 2>&1")
    if exit_code != 0:
        print(f"[Orchestrator] pytest not found ({out}). Installing inside container …")
        exit_code2, out2 = _exec_in_container("pip install pytest --quiet 2>&1")
        if exit_code2 != 0:
            print(f"[Orchestrator] ⚠️  pip install failed:\n{out2}")
        else:
            _, ver = _exec_in_container("python3 -m pytest --version 2>&1")
            print(f"[Orchestrator] ✅  pytest installed: {ver}")
    else:
        print(f"[Orchestrator] ✅  pytest already available: {out}")

    # Ensure /workspace/src and /workspace/tests exist inside the container
    _exec_in_container(f"mkdir -p {WORKSPACE_IN_CTR}/src {WORKSPACE_IN_CTR}/tests")
    # Create __init__.py files so imports work
    _exec_in_container(f"touch {WORKSPACE_IN_CTR}/src/__init__.py {WORKSPACE_IN_CTR}/tests/__init__.py")
    print("[Orchestrator] Container environment ready.\n")


# ─────────────────────────────────────────────
# 2.  Tool-call tracer
#     Wraps every tool function so that calls + results are printed.
# ─────────────────────────────────────────────

def _traced(fn: Callable) -> Callable:
    """Decorator: print [TOOL] call + result for every tool invocation."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Build a readable call signature
        arg_parts = [repr(a)[:120] for a in args]
        kwarg_parts = [f"{k}={repr(v)[:120]}" for k, v in kwargs.items()]
        sig = ", ".join(arg_parts + kwarg_parts)
        print(f"\n  [TOOL CALL] {fn.__name__}({sig})")
        result = fn(*args, **kwargs)
        # Truncate long results in the trace
        display = str(result)
        if len(display) > 400:
            display = display[:400] + f"\n  … ({len(str(result))} chars total)"
        print(f"  [TOOL RESULT] {display}\n")
        return result
    return wrapper


# ─────────────────────────────────────────────
# 3.  Tools  (all execute inside the container or on the mounted volume)
# ─────────────────────────────────────────────

@_traced
def run_shell(command: str) -> str:
    """
    Run ANY bash command INSIDE the Docker sandbox container.
    Use for: pip install, environment checks, running scripts, ls, etc.
    The /workspace directory inside the container is synced with local ./workspace.
    Examples:
      run_shell("pip install numpy scipy")
      run_shell("python3 src/main.py")
      run_shell("pip list | grep pytest")
      run_shell("ls -la /workspace/src")
    Args:
        command: bash command string — executed INSIDE the container, not on host
    """
    return _exec(command)


@_traced
def run_tests(test_path: str = "tests") -> str:
    """
    Run pytest INSIDE the Docker container on the given path.
    Args:
        test_path: relative path inside /workspace (default: 'tests')
    """
    full_path = f"{WORKSPACE_IN_CTR}/{test_path.lstrip('/')}"
    return _exec(f"python3 -m pytest {full_path} -v --tb=short 2>&1")


@_traced
def read_file(path: str) -> str:
    """
    Read a file from the workspace (mounted volume).
    Args:
        path: relative path, e.g. 'src/math_utils.py'
    """
    full = os.path.join(WORKSPACE_IN_HOST, path.lstrip("/"))
    if not os.path.exists(full):
        return f"Error: file not found: {path}"
    with open(full) as f:
        return f.read()


@_traced
def write_file(path: str, content: str) -> str:
    """
    Write (overwrite) a file in the workspace.
    File is immediately available inside the container via the volume mount.
    Args:
        path: relative path, e.g. 'src/math_utils.py'
        content: full file content
    """
    full = os.path.join(WORKSPACE_IN_HOST, path.lstrip("/"))
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    # Verify the file is visible inside the container
    exit_code, _ = _exec_in_container(f"test -f {WORKSPACE_IN_CTR}/{path.lstrip('/')}")
    status = "✅ visible in container" if exit_code == 0 else "⚠️ NOT visible in container"
    return f"Written: {path} [{status}]"


@_traced
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """
    Replace the FIRST occurrence of old_str with new_str in a file.
    Args:
        path: relative path
        old_str: exact string to replace
        new_str: replacement string
    """
    full = os.path.join(WORKSPACE_IN_HOST, path.lstrip("/"))
    if not os.path.exists(full):
        return f"Error: file not found: {path}"
    with open(full) as f:
        text = f.read()
    if old_str not in text:
        return f"Error: pattern not found in {path}"
    with open(full, "w") as f:
        f.write(text.replace(old_str, new_str, 1))
    return f"Edited: {path}"


@_traced
def discover_files(subdir: str = ".") -> str:
    """
    List all files in the workspace (recursive).
    Args:
        subdir: relative path to start from (default '.')
    """
    base = os.path.join(WORKSPACE_IN_HOST, subdir.lstrip("/"))
    if not os.path.exists(base):
        return f"Error: directory not found: {subdir}"
    results = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), WORKSPACE_IN_HOST)
            results.append(rel)
    return "\n".join(results) if results else "(empty)"


@_traced
def grep_files(pattern: str, subdir: str = ".", file_glob: str = "*.py") -> str:
    """
    Search for a pattern in workspace files (runs grep inside the container).
    Args:
        pattern: grep regex
        subdir: subdirectory to search (relative)
        file_glob: file pattern (default '*.py')
    """
    ctr_path = f"{WORKSPACE_IN_CTR}/{subdir.lstrip('/')}"
    return _exec(f"grep -rn --include='{file_glob}' '{pattern}' {ctr_path} 2>&1")


@_traced
def docker_python_exec(code: str) -> str:
    """
    Execute a Python snippet INSIDE the container for quick checks.
    Always use print() to see output.
    Args:
        code: Python source code
    """
    escaped = code.replace("'", "'\\''")
    return _exec(f"python3 -c '{escaped}'")


# ─────────────────────────────────────────────
# 4.  Agent definitions
# ─────────────────────────────────────────────

_DEV_TOOLS  = [run_shell, read_file, write_file, edit_file, discover_files, grep_files, docker_python_exec]
_TEST_TOOLS = [run_shell, run_tests, read_file, write_file, edit_file, discover_files, grep_files]

developer_agent = Agent(
    name="developer_agent",
    model=OLLAMA_MODEL,
    instruction="""
You are a senior Python developer. Your workspace is /workspace inside a Docker container.
Files you write are placed in ./workspace/ on the host and are AUTOMATICALLY visible in the
container at /workspace — you do NOT need to copy them manually.

## MANDATORY steps before writing ANY code
1. discover_files() — see the full project structure.
2. If a relevant source file exists, read_file() it.
3. grep_files("<function_name>") — check if the function already exists.
4. Only then: write new code or edit existing code.

## Fixing environment problems
When run_shell() or a test returns "No module named X":
  - run_shell("pip install <package>")  ← this runs INSIDE the container.
  - Then verify: run_shell("python3 -c 'import <package>; print(\"ok\")'")
  - NEVER skip this step and NEVER say "I can't install it". Just do it.

## Coding rules
- Source files go in src/<module>.py.
- Never modify files in tests/.
- After writing/editing, verify with: run_shell("python3 -c 'import src.<module>'")
- State "DEVELOPER DONE: <summary>" when finished.
""",
    tools=_DEV_TOOLS,
)

test_dev_agent = Agent(
    name="test_dev_agent",
    model=OLLAMA_MODEL,
    instruction="""
You are a senior QA engineer. Your workspace is /workspace inside a Docker container.
Files you write are placed in ./workspace/ on the host and are AUTOMATICALLY visible in the
container at /workspace — you do NOT need to copy them manually.

## MANDATORY first steps
1. run_shell("python3 -m pytest --version") — verify pytest is available.
   If missing: run_shell("pip install pytest") then verify again.
2. discover_files() — see existing source and test files.
3. read_file() the relevant source file(s).
4. grep_files() to find all public functions/classes.
5. Check if a test file already exists; extend rather than overwrite.

## Writing tests
- Tests go in tests/test_<module>.py.
- Cover: happy path, edge cases (0, negatives, floats), error conditions.
- Name tests: test_<what>_<condition>().
- Never modify src/ files.
- After writing, run run_tests() and include the output.
- State "TEST DEV DONE: <summary>" when finished.
""",
    tools=_TEST_TOOLS,
)


# ─────────────────────────────────────────────
# 5.  ADK Runner helpers
# ─────────────────────────────────────────────

APP_NAME = "multi_agent_system"
USER_ID  = "orchestrator"

_session_services: dict[str, InMemorySessionService] = {}
_runners: dict[str, Runner] = {}
_session_ids: dict[str, str] = {}


def _ensure_runner(agent: Agent) -> None:
    name = agent.name
    if name not in _runners:
        svc = InMemorySessionService()
        _session_services[name] = svc
        _runners[name] = Runner(agent=agent, app_name=APP_NAME, session_service=svc)
        sid = str(uuid.uuid4())
        _session_ids[name] = sid
        asyncio.get_event_loop().run_until_complete(
            svc.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=sid)
        )


def _reset_session(agent: Agent) -> None:
    """Fresh session between tasks — prevents context bleed."""
    _ensure_runner(agent)
    svc = _session_services[agent.name]
    sid = str(uuid.uuid4())
    _session_ids[agent.name] = sid
    asyncio.get_event_loop().run_until_complete(
        svc.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=sid)
    )


async def _call_agent_async(agent: Agent, message: str) -> str:
    _ensure_runner(agent)
    runner     = _runners[agent.name]
    session_id = _session_ids[agent.name]
    content    = types.Content(role="user", parts=[types.Part(text=message)])
    final_text = "(no response)"
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = "".join(
                    p.text for p in event.content.parts if hasattr(p, "text") and p.text
                )
            break
    return final_text


def _agent_call(agent: Agent, message: str) -> str:
    return asyncio.get_event_loop().run_until_complete(_call_agent_async(agent, message))


# ─────────────────────────────────────────────
# 6.  Task-file helpers
# ─────────────────────────────────────────────

def _parse_pending_tasks(path: str) -> list[str]:
    with open(path) as f:
        lines = f.readlines()
    return [
        l.strip() for l in lines
        if l.strip()
        and not l.strip().startswith("#")
        and not l.strip().startswith("[DONE]")
    ]


def _mark_task_done(path: str, task: str) -> None:
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if line.strip() == task and not line.strip().startswith("[DONE]"):
                f.write(f"[DONE] {line}")
            else:
                f.write(line)


# ─────────────────────────────────────────────
# 7.  Error classifier + prompt builder
# ─────────────────────────────────────────────

def _classify_error(test_output: str) -> str:
    lo = test_output.lower()
    if "no module named" in lo or "modulenotfounderror" in lo:
        return "env"
    if "syntaxerror" in lo or "indentationerror" in lo:
        return "syntax"
    if "assertionerror" in lo or "assert " in lo:
        return "logic"
    if "error" in lo and "test_" in lo:
        return "test_setup"
    return "unknown"


def _build_fix_prompt(error_type: str, task: str, test_output: str) -> str:
    header = f"TASK: {task}\n\npytest output:\n{test_output[:3000]}\n\n"
    if error_type == "env":
        return (
            "## Environment Fix Required\n\n" + header +
            "A Python package is missing inside the Docker container.\n\n"
            "STEPS (use run_shell — it runs INSIDE the container):\n"
            "1. run_shell('pip install <missing_package>')\n"
            "2. run_shell(\"python3 -c 'import <package>; print(\\\"ok\\\")'\") to verify.\n"
            "3. If pip itself fails, run_shell('pip install --upgrade pip') first.\n"
            "4. Do NOT modify any source or test files.\n"
            "State 'DEVELOPER DONE' when the env is fixed."
        )
    if error_type == "syntax":
        return (
            "## Syntax Error in Source Code\n\n" + header +
            "STEPS:\n"
            "1. read_file() the failing source file.\n"
            "2. Fix the SyntaxError/IndentationError.\n"
            "3. Verify: run_shell(\"python3 -c 'import src.<module>'\").\n"
            "State 'DEVELOPER DONE' when fixed."
        )
    if error_type == "logic":
        return (
            "## Logic Error — Tests Asserting Wrong Results\n\n" + header +
            "STEPS:\n"
            "1. read_file() the test file to understand what is expected.\n"
            "2. read_file() the source file.\n"
            "3. Fix the logic in src/ only. Never modify tests/.\n"
            "State 'DEVELOPER DONE' when fixed."
        )
    return (
        "## Tests Still Failing\n\n" + header +
        "STEPS:\n"
        "1. discover_files() then read relevant files.\n"
        "2. If env issue: run_shell('pip install <pkg>') — runs inside container.\n"
        "3. If code issue: fix src/ only, not tests/.\n"
        "State 'DEVELOPER DONE' when fixed."
    )


# ─────────────────────────────────────────────
# 8.  Orchestration loop
# ─────────────────────────────────────────────

def _tests_passing() -> bool:
    output = run_tests("tests")
    return not output.startswith("[exit")


def orchestrate(task_file: str) -> None:
    task_file_abs = os.path.abspath(task_file)
    if not os.path.exists(task_file_abs):
        print(f"[Orchestrator] Task file not found: {task_file_abs}")
        return

    # Ensure the container is up and pytest is installed before we start
    _ensure_env()

    print(f"[Orchestrator] Starting. Task file: {task_file_abs}")
    pending = _parse_pending_tasks(task_file_abs)
    if not pending:
        print("[Orchestrator] No pending tasks found.")
        return

    for task in pending:
        print(f"\n{'='*60}")
        print(f"[Orchestrator] >> Task: {task}")
        print(f"{'='*60}")

        _reset_session(developer_agent)
        _reset_session(test_dev_agent)

        # ── Step 1: Developer checks existing code, then implements ──
        dev_prompt = (
            "Before writing any code:\n"
            "1. discover_files() — inspect the workspace.\n"
            "2. If a relevant .py file exists, read_file() it.\n"
            "3. grep_files() to check if the required function already exists.\n"
            "4. Implement only what is missing or incorrect.\n\n"
            f"TASK: {task}\n\n"
            "State 'DEVELOPER DONE: <summary>' when finished."
        )
        print("\n[Orchestrator] ── Step 1: developer_agent ──")
        dev_response = _agent_call(developer_agent, dev_prompt)
        print(f"[developer_agent final response]\n{dev_response}\n")

        # ── Step 2: Test dev verifies env, writes/extends tests ───────
        test_prompt = (
            "Before writing any tests:\n"
            "1. run_shell('python3 -m pytest --version') — if missing, run_shell('pip install pytest').\n"
            "2. discover_files() — see the workspace layout.\n"
            "3. read_file() the relevant source file(s).\n"
            "4. grep_files() to find all public functions.\n"
            "5. Write or extend tests/test_<module>.py.\n"
            "6. run_tests() and include the output in your summary.\n\n"
            f"TASK: {task}\n"
            f"Developer summary: {dev_response[:400]}\n\n"
            "State 'TEST DEV DONE: <summary>' when finished."
        )
        print("[Orchestrator] ── Step 2: test_dev_agent ──")
        test_response = _agent_call(test_dev_agent, test_prompt)
        print(f"[test_dev_agent final response]\n{test_response}\n")

        # ── Step 3: Fix loop ──────────────────────────────────────────
        for iteration in range(1, MAX_ITERATIONS + 1):
            if _tests_passing():
                print(f"[Orchestrator] ✅  Tests pass after {iteration} iteration(s).")
                break

            test_output = run_tests.__wrapped__("tests")   # bypass tracer for clean output
            error_type  = _classify_error(test_output)

            print(f"\n[Orchestrator] ❌  Failing — iter {iteration}/{MAX_ITERATIONS}, type={error_type}")
            print(f"  pytest output:\n{test_output[:2000]}")

            fix_prompt = _build_fix_prompt(error_type, task, test_output)
            print(f"[Orchestrator] ── Fix #{iteration} → developer_agent (type={error_type}) ──")
            fix_response = _agent_call(developer_agent, fix_prompt)
            print(f"[developer_agent fix #{iteration}]\n{fix_response}\n")

        else:
            print(f"[Orchestrator] ⚠️   Max iterations ({MAX_ITERATIONS}) reached.")

        _mark_task_done(task_file_abs, task)
        print(f"[Orchestrator] Marked [DONE]: {task}")

    print("\n[Orchestrator] All tasks processed. Final test run:")
    print(run_tests("tests"))


# ─────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    task_file = sys.argv[1] if len(sys.argv) > 1 else "tasks.txt"
    orchestrate(task_file)
