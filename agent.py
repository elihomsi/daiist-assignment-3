"""
Assignment 3: the agent (`main.py ask "question"` runs `python agent.py "question"`).

LangChain `create_agent` on Gemini. Its tools come from server.py over MCP and
AGENTS.md is the system prompt.

Safety around the tools:
  1. Approval: add_document never runs unless a human approves it (the person
     typing the question, or the `approve` function that evaluate.py passes in).
  2. Guardrail: add_document only accepts .htm/.html/.txt/.md files that sit
     inside this project folder. The agent cannot read other files on the computer.
  3. Limits: at most MAX_MODEL_CALLS model calls and MAX_TOOL_CALLS tool calls
     per question, so one question cannot loop and burn quota.

evaluate.py imports `ask(question, approve=None)`, which returns the answer and
the tool trace.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
MODEL = "gemini-2.5-flash"
MAX_MODEL_CALLS = 8
MAX_TOOL_CALLS = 6
ALLOWED_SUFFIXES = (".htm", ".html", ".txt", ".md")


def system_prompt() -> str:
    return (ROOT / "AGENTS.md").read_text(encoding="utf-8")


def ask_in_terminal(name: str, args: dict) -> bool:
    """Default approval: show what the agent wants to do and ask a person."""
    if not sys.stdin.isatty():
        print(f"[approval] {name}({args}) denied: no person available to approve.", file=sys.stderr)
        return False
    answer = input(f"\nThe agent wants to run {name} with {json.dumps(args)}. Approve? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def path_problem(path: str) -> str | None:
    """Guardrail for add_document: return why the path is not allowed, or None."""
    file = (ROOT / Path(path).expanduser()).resolve()  # relative paths count from the project folder
    if ROOT not in file.parents:
        return f"Blocked: {path!r} is outside the project folder. Only files inside it can be added."
    if file.suffix.lower() not in ALLOWED_SUFFIXES:
        return f"Blocked: only {', '.join(ALLOWED_SUFFIXES)} files can be added."
    return None


def content_text(content) -> str:
    """Gemini may return a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


async def build_tools(session, approve, trace: list[dict]) -> list[StructuredTool]:
    """Wrap each MCP tool as a LangChain tool, with approval and guardrail on add_document."""
    tools = []
    for mcp_tool in (await session.list_tools()).tools:

        def make(name: str):
            async def run(**kwargs) -> str:
                entry = {"tool": name, "args": kwargs, "result": ""}
                trace.append(entry)
                if name == "add_document":
                    problem = path_problem(kwargs.get("path", ""))
                    if problem:
                        entry["result"] = problem
                        return problem
                    if not await asyncio.to_thread(approve, name, kwargs):
                        entry["result"] = "Not approved by the user. The document was NOT added."
                        return entry["result"]
                result = await session.call_tool(name, kwargs)
                text = "".join(getattr(part, "text", "") for part in result.content)
                entry["result"] = text
                return text

            return run

        tools.append(
            StructuredTool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                args_schema=mcp_tool.input_schema,
                coroutine=make(mcp_tool.name),
            )
        )
    return tools


def make_model():
    return ChatGoogleGenerativeAI(model=MODEL, google_api_key=load_api_key(), temperature=0, max_retries=3)


async def run_agent(question: str, approve=ask_in_terminal, model=None, session=None) -> dict:
    """Answer one question. Returns {answer, tool_trace, seconds, input_tokens, output_tokens}."""

    async def work(session):
        trace: list[dict] = []
        tools = await build_tools(session, approve, trace)
        agent = create_agent(
            model or make_model(),
            tools,
            system_prompt=system_prompt(),
            middleware=[
                ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
                ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="continue"),
            ],
        )
        start = time.time()
        state = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        seconds = time.time() - start
        messages = state["messages"]
        tokens_in = tokens_out = 0
        for message in messages:
            usage = getattr(message, "usage_metadata", None) or {}
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
        answer = content_text(messages[-1].content).strip() if messages else ""
        return {
            "answer": answer or "(no answer)",
            "tool_trace": trace,
            "seconds": round(seconds, 2),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        }

    if session is not None:
        return await work(session)
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server.py")], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            return await work(mcp_session)


def ask(question: str, approve=ask_in_terminal) -> dict:
    """Synchronous helper for evaluate.py."""
    return asyncio.run(run_agent(question, approve))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python main.py ask "your question"')
    result = ask(" ".join(sys.argv[1:]))
    print(result["answer"])
    used = ", ".join(t["tool"] for t in result["tool_trace"]) or "none"
    print(f"\n[tools used: {used} | {result['seconds']}s | "
          f"{result['input_tokens']} in / {result['output_tokens']} out tokens]", file=sys.stderr)


if __name__ == "__main__":
    main()