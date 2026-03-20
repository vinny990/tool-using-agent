# Research Assistant — Tool-Using Agent

A web app that shows an LLM **deciding in real time** which tool to reach for: web search, a calculator, or its own knowledge. Built with LangChain, Flask, and DuckDuckGo (no API key required for search).

**Live demo:** https://tool-using-agent.onrender.com

---

## Agent vs Chain — what's the difference?

| | Chain | Agent |
|---|---|---|
| **Control flow** | Fixed sequence of steps defined by the developer | Dynamic — the model decides what to do next |
| **Tools** | Usually none (just prompts/LLMs) | One or more tools the model can invoke |
| **Use case** | Predictable pipelines (summarise → translate → store) | Open-ended tasks where the right steps aren't known in advance |

A **chain** is like a recipe: step 1, step 2, step 3 — always in that order.
An **agent** is like a chef: it reads the fridge, decides what to cook, and may improvise mid-way.

---

## The ReAct Reasoning Loop

This project uses the **ReAct** pattern (**Re**asoning + **Act**ing), which interleaves thinking and tool use:

```
Question: What is the population of Tokyo, and how many times larger is it than Lisbon?

Thought:  I need the population of Tokyo. I'll search the web.
Action:   web_search("Tokyo population 2024")
Observation: Tokyo metropolitan area ~37.4 million …

Thought:  Now I need Lisbon's population.
Action:   web_search("Lisbon population 2024")
Observation: Lisbon ~545,000 …

Thought:  Now I can calculate the ratio.
Action:   calculator(37400000 / 545000)
Observation: 68.62…

Thought:  I have everything I need.
Final Answer: Tokyo has ~37.4 million people; it is about 68.6× larger than Lisbon (~545 000).
```

Each `Thought → Action → Observation` cycle is one **step**. The UI exposes every step so you can watch the agent reason, not just see the final answer.

---

## Quickstart

```bash
# 1. Clone / enter the project
cd tool-using-agent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
cp .env.example .env
# edit .env and paste your key

# 5. Run
python app.py
```

Open http://localhost:5000 and start asking questions.

---

## Example questions to try

- `What is the GDP of Germany in 2024?`  → uses **web search**
- `What is 17% of 4,250?`                → uses **calculator**
- `What is the speed of light in km/h, and how long would it take to reach the Moon?` → uses **both**

---

## Project layout

```
tool-using-agent/
├── agent.py            # LangChain ReAct agent (DuckDuckGo + calculator)
├── app.py              # Flask server — GET / and POST /ask
├── templates/
│   └── index.html      # Chat UI with live reasoning steps
├── requirements.txt
├── .env.example
└── README.md
```

---

## How the agent decides which tool to use

The prompt instructs the model to output a structured `Thought / Action / Action Input / Observation` cycle (defined in the `hwchase17/react` prompt pulled from LangChain Hub). Each tool's `description` field guides when the model should reach for it:

- **web_search** — current events, facts, anything needing up-to-date data
- **calculator** — numeric expressions, unit conversions, arithmetic

If the model already knows the answer confidently, it skips both tools and responds directly — that's an important part of agentic behaviour too.
