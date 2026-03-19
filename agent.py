import os
import numexpr
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

load_dotenv()


def _calculate(expression: str) -> str:
    """Safely evaluate a math expression using numexpr."""
    try:
        result = numexpr.evaluate(expression.strip())
        return str(float(result))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def _build_executor() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    search_tool = Tool(
        name="web_search",
        func=DuckDuckGoSearchRun().run,
        description=(
            "Search the web for current information. Use this for questions about "
            "recent events, facts, people, or anything requiring up-to-date data. "
            "Input should be a concise search query."
        ),
    )

    calculator_tool = Tool(
        name="calculator",
        func=_calculate,
        description=(
            "Evaluate mathematical expressions. Use this for arithmetic, percentages, "
            "unit conversions, or any numeric calculation. "
            "Input must be a valid math expression, e.g. '42 * 1.08' or 'sqrt(144)'."
        ),
    )

    tools = [search_tool, calculator_tool]

    # Pull the standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )


# Lazy singleton — built once on first call
_executor: Optional[AgentExecutor] = None


def run(question: str) -> dict:
    """Run the agent on a question and return the answer plus reasoning steps."""
    global _executor
    if _executor is None:
        _executor = _build_executor()

    result = _executor.invoke({"input": question})

    steps = [
        {
            "tool": action.tool,
            "input": action.tool_input,
            "output": observation,
        }
        for action, observation in result.get("intermediate_steps", [])
    ]

    return {
        "answer": result["output"],
        "steps": steps,
    }
