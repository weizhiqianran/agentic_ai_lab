# **从规划到执行：使用 LangGraph 构建智能代理**

## 引言

提供的示例展示了一个使用 LangChain 和 LangGraph 实现的**多步骤任务执行工作流程**。该工作流程旨在通过将复杂查询分解为更小、更易管理的步骤，依次执行这些步骤，并根据需要动态重新规划，从而处理复杂的查询。这种方法确保代理能够处理需要多个步骤的任务，例如搜索信息、处理结果并提供最终响应。

### 工作流程的关键步骤：
1. **初始化语言模型（LLM）**：工作流程首先使用 OpenAI 的 GPT-4 或 Anthropic 的 Claude-3 初始化一个语言模型（LLM）。API 密钥通过 `kaggle_secrets` 安全获取。
2. **定义工具和自定义提示**：工作流程定义了工具（例如 `DuckDuckGoSearchRun`）和一个自定义提示，以指导代理的行为。
3. **定义状态和规划逻辑**：定义了状态结构（`PlanExecute`）用于跟踪输入、计划、过去步骤和最终响应。`planner` 生成实现目标的分步计划。
4. **定义重新规划逻辑**：`replanner` 根据之前步骤的结果更新计划。如果不需要进一步步骤，则返回最终响应。
5. **创建图并定义节点**：创建工作流程图，包括用于规划（`plan_node`）、执行（`agent_node`）和重新规划（`replan_node`）的节点。图协调任务的执行并确保在最终响应准备好时终止工作流程。
6. **执行工作流程**：使用示例输入执行工作流程，代理逐步处理查询并提供最终结果。

---

## 准备工作

### 安装所需库
本节安装使用 LangChain、OpenAI 嵌入、Anthropic 模型、DuckDuckGo 搜索和其他实用工具所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型和 API 的集成。
- `langchain-anthropic`：支持与 Anthropic 模型和 API 的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。
- `langgraph`：用于在 LangChain 中构建和可视化基于图的工作流程的库。
- `duckduckgo-search`：实现对 DuckDuckGo 搜索功能的程序化访问。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU duckduckgo-search
```

---

## 步骤 1：使用 API 密钥初始化 LLM
此步骤使用 OpenAI 的 GPT-4 或 Anthropic 的 Claude-3 初始化语言模型（LLM）。API 密钥通过 `kaggle_secrets` 安全获取。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化 LLM
model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
#model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

## 步骤 2：定义工具和自定义提示
此步骤定义了工具（例如 `DuckDuckGoSearchRun`）和代理的自定义提示。提示指导代理逐步执行任务。

```python
import operator
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# 定义工具
tools = [DuckDuckGoSearchRun()]

# 定义自定义提示
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个能够逐步执行任务的助手。")
])

# 定义执行代理
agent_executor = create_react_agent(model, tools, state_modifier=custom_prompt)
```

## 步骤 3：定义状态和规划逻辑
此步骤定义了状态结构（`PlanExecute`）和规划逻辑。`Plan` 类表示实现目标的步骤，`planner` 生成分步计划。

```python
# 定义状态
class PlanExecute(TypedDict):
    """
    表示执行工作流程的状态。
    
    属性：
        input (str): 用户的输入或目标。
        plan (List[str]): 实现目标的步骤列表。
        past_steps (List[Tuple]): 表示已完成步骤及其结果的元组列表。
        response (str): 对用户的最终响应。
    """
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# 规划步骤
class Plan(BaseModel):
    """
    表示由实现目标的步骤组成的计划。
    
    属性：
        steps (List[str]): 要遵循的步骤列表，按所需顺序排序。
    """
    steps: List[str] = Field(description="要遵循的不同步骤，应按顺序排序")

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """为给定的目标制定一个简单的分步计划。\
该计划应涉及单独的任务，如果正确执行，将得出正确答案。不要添加任何多余步骤。\
最后一步的结果应为最终答案。确保每个步骤包含所有必要信息 - 不要跳过步骤。"""),
    ("placeholder", "{messages}")
])

planner = planner_prompt | model.with_structured_output(Plan)
```

## 步骤 4：定义重新规划逻辑
此步骤定义了重新规划逻辑。`replanner` 根据之前步骤的结果更新计划。如果不需要进一步步骤，则返回最终响应。

```python
# 重新规划步骤
class Response(BaseModel):
    """
    表示对用户的响应。
    
    属性：
        response (str): 最终响应消息。
    """
    response: str

class Act(BaseModel):
    """
    表示要执行的操作，可以是响应或新计划。
    
    属性：
        action (Union[Response, Plan]): 要执行的操作。使用 `Response` 回应用户，或使用 `Plan` 继续执行步骤。
    """
    action: Union[Response, Plan] = Field(description="要执行的操作。如果要回应用户，使用 Response。如果需要进一步使用工具获取答案，使用 Plan。")

replanner_prompt = ChatPromptTemplate.from_template(
    """为给定的目标制定一个简单的分步计划。\
该计划应涉及单独的任务，如果正确执行，将得出正确答案。不要添加任何多余步骤。\
最后一步的结果应为最终答案。确保每个步骤包含所有必要信息 - 不要跳过步骤。

你的目标是：
{input}

你原来的计划是：
{plan}

你目前已完成以下步骤：
{past_steps}

相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，则做出回应。否则，填写计划。仅添加仍需完成的步骤到计划中。不要返回已完成的步骤作为计划的一部分。"""
)

replanner = replanner_prompt | model.with_structured_output(Act)
```

## 步骤 5：创建图并定义节点
此步骤创建工作流程图并定义节点（`plan_node`、`agent_node`、`replan_node`）。图协调任务的执行、重新规划和终止。

```python
# 创建图
async def agent_node(state: PlanExecute) -> PlanExecute:
    """
    使用代理执行计划中的第一步。

    参数：
        state (PlanExecute): 当前工作流程状态。

    返回：
        PlanExecute: 更新后的状态，包含执行步骤的结果。
    """
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"对于以下计划：\n{plan_str}\n\n你的任务是执行步骤 {1}，{task}。"
    agent_response = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
    return {"past_steps": [(task, agent_response["messages"][-1].content)]}

async def plan_node(state: PlanExecute) -> PlanExecute:
    """
    根据用户输入生成计划。

    参数：
        state (PlanExecute): 当前工作流程状态。

    返回：
        PlanExecute: 更新后的状态，包含生成的计划。
    """
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

async def replan_node(state: PlanExecute) -> PlanExecute:
    """
    根据之前步骤的结果更新计划。

    参数：
        state (PlanExecute): 当前工作流程状态。

    返回：
        PlanExecute: 更新后的状态，包含新计划或最终响应。
    """
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

def should_end(state: PlanExecute) -> Literal["agent_node", END]:
    """
    判断工作流程是否应结束或继续。

    参数：
        state (PlanExecute): 当前工作流程状态。

    返回：
        Literal["agent_node", END]: 如果工作流程应终止，返回 `END`，否则返回 `"agent_node"`。
    """
    if "response" in state and state["response"]:
        return END
    else:
        return "agent_node"

workflow = StateGraph(PlanExecute)
workflow.add_node("plan_node", plan_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("replan_node", replan_node)

workflow.add_edge(START, "plan_node")
workflow.add_edge("plan_node", "agent_node")
workflow.add_edge("agent_node", "replan_node")
workflow.add_conditional_edges("replan_node", should_end, ["agent_node", END])

# 最后，编译它！
# 这将其编译为 LangChain Runnable，
# 意味着你可以像使用其他可运行对象一样使用它
app = workflow.compile()

from IPython.display import Image, display
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

## 步骤 6：执行工作流程
此步骤使用示例输入执行工作流程。代理处理输入，生成计划，执行任务，并返回最终结果。

```python
# 示例 1：发现 2022 年 FIFA 世界杯冠军的首都
config = {"recursion_limit": 30}
inputs = {"input": "最近一次 FIFA 世界杯冠军国家的首都是哪里？"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

```python
# 示例 2：获得最多奥运金牌国家最大城市的人口
config = {"recursion_limit": 30}
inputs = {"input": "获得最多奥运金牌国家的最大城市人口是多少？"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

```python
# 示例 3：2023 年 GDP 最高国家的货币
config = {"recursion_limit": 30}
inputs = {"input": "2023 年 GDP 最高国家的货币是什么？"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

```python
# 示例 4：创建最受欢迎智能手机操作系统的公司创始人
config = {"recursion_limit": 30}
inputs = {"input": "创建最受欢迎智能手机操作系统的公司的创始人是谁？"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

---

## 结论

该示例展示了如何使用 LangChain 和 LangGraph 构建一个**动态、多步骤任务执行工作流程**。通过将复杂查询分解为较小的步骤，代理能够高效处理需要搜索信息、处理结果并提供最终响应的任务。该工作流程高度灵活，允许根据之前步骤的结果动态重新规划，并且可以适应广泛的用例。

从该示例中得出的主要收获包括：
- **状态管理**在跟踪多步骤任务进度中的重要性。
- 根据中间结果**动态重新规划**的能力，确保工作流程适应新信息。
- 使用诸如 `DuckDuckGoSearchRun` 等**工具**来收集信息并增强代理能力。
- **基于图的工作流程**设计，提供清晰且模块化的任务定义和执行结构。

这种方法特别适用于**问答**、**数据检索**和**自动化决策**等应用场景，这些场景通常需要多个步骤和动态适应。