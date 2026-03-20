import os
import numexpr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

load_dotenv()


def _calculate(expression: str) -> str:
    """Safely evaluate a math expression using numexpr."""
    try:
        result = numexpr.evaluate(expression.strip())
        return str(float(result))
    except Exception as e:
        return f"Error evaluating expression: {e}"


_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this for questions about
    recent events, facts, people, or anything requiring up-to-date data.
    Input should be a concise search query."""
    return _ddg.run(query)


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions. Use this for arithmetic, percentages,
    unit conversions, or any numeric calculation.
    Input must be a valid math expression, e.g. '42 * 1.08' or 'sqrt(144)'."""
    return _calculate(expression)


def _build_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return create_react_agent(llm, [web_search, calculator])


# Lazy singleton — built once on first call
_agent = None


def run(question: str) -> dict:
    """Run the agent on a question and return the answer plus reasoning steps."""
    global _agent
    if _agent is None:
        _agent = _build_agent()

    result = _agent.invoke({"messages": [{"role": "user", "content": question}]})

    messages = result.get("messages", [])

    # Build a map of tool_call_id -> input args from AI messages
    call_inputs = {}
    for msg in messages:
        if msg.type == "ai":
            for tc in getattr(msg, "tool_calls", None) or []:
                call_inputs[tc["id"]] = tc.get("args", {})

    # Each ToolMessage has the output and a tool_call_id linking back to the input
    steps = [
        {
            "tool": msg.name,
            "input": call_inputs.get(msg.tool_call_id, ""),
            "output": msg.content,
        }
        for msg in messages
        if msg.type == "tool"
    ]

    # Final answer is the last AI message without tool calls
    answer = ""
    for msg in reversed(messages):
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            answer = msg.content
            break

    return {
        "answer": answer,
        "steps": steps,
    }
