# 如何使用 LangGraph 和 LangChain 创建支持工具的代理

## **引言**

在快速发展的 AI 和自然语言处理领域，构建能够与工具和外部系统交互的智能代理变得越来越重要。LangGraph 是一个用于创建基于图的工作流的强大框架，它提供了一种无缝的方式，将工具和语言模型集成到基于代理的系统中。本文探讨了如何利用关键组件——`langgraph.prebuilt.ToolNode`、`langchain_community.agent_toolkits.load_tools` 和 `langchain.tools.StructuredTool`——与 `langgraph.prebuilt.create_react_agent` 一起，创建动态的、支持工具的代理。通过理解这些构建模块，开发者可以设计出利用外部工具、管理复杂工作流并提供结构化输出的代理，从而开启智能自动化的新可能性。

### **比较表**

| 功能/类                    | `ToolNode`                          | `load_tools`                        | `StructuredTool.from_function`       |
|----------------------------|-------------------------------------|-------------------------------------|--------------------------------------|
| **用途**                  | 在图节点中执行工具                  | 按名称加载预定义工具                | 将函数转换为工具                    |
| **工具集成**              | 是                                  | 是                                  | 是                                  |
| **自定义工具**            | 支持                                | 不支持                              | 支持                                |
| **错误处理**              | 是                                  | 不支持                              | 不支持                              |
| **语言模型集成**          | 可选                                | 可选                                | 不支持                              |
| **使用场景**              | 在工作流中执行工具                  | 加载预构建工具                      | 创建自定义工具                      |

---

## 准备工作

### 安装所需库

本节将安装使用 LangChain、OpenAI、Anthropic、DuckDuckGo、Arxiv、GraphQL 和其他实用工具所需的 Python 库。这些库包括：

- **`langchain-openai`**：提供与 OpenAI 语言模型和 API 的集成，支持使用 GPT-4 等模型进行自然语言任务。
- **`langchain-anthropic`**：支持与 Anthropic 的模型（如 Claude）集成，用于高级语言处理和推理。
- **`langchain_community`**：包含 LangChain 的社区贡献模块和工具，包括额外的实用工具和集成。
- **`langchain_experimental`**：包含 LangChain 的实验性功能和实用工具，为高级用例提供前沿能力。
- **`langgraph`**：一个用于在 LangChain 中构建和可视化基于图的工作流的库，是创建支持工具的代理和复杂工作流的基础。
- **`duckduckgo-search`**：支持以编程方式访问 DuckDuckGo 的搜索功能，使代理能够从网络检索实时信息。
- **`arxiv`**：提供对 Arxiv API 的访问，使代理能够搜索和检索科学论文和研究文章。
- **`httpx` 和 `gql`**：用于发起 HTTP 请求和与 GraphQL API 交互的库，是集成基于 GraphQL 的工具和服务的关键。

这些库为构建能够与外部工具交互、检索信息并执行复杂工作流的智能代理奠定了基础。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU duckduckgo-search
!pip install -qU arxiv
!pip install -qU httpx gql # graphql
```

### 初始化 OpenAI 和 Anthropic 聊天模型

本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全地获取 OpenAI 和 Anthropic 的 API 密钥，并初始化它们的聊天模型。`ChatOpenAI` 和 `ChatAnthropic` 类用于创建这些模型的实例，可用于自然语言处理任务，如文本生成和对话 AI。

**关键步骤**：
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全地检索 OpenAI 和 Anthropic 的 API 密钥。
2. **初始化聊天模型**：
   - 使用 `gpt-4o-mini` 模型和获取的 OpenAI API 密钥初始化 `ChatOpenAI` 类。
   - 使用 `claude-3-5-sonnet-latest` 模型和获取的 Anthropic API 密钥初始化 `ChatAnthropic` 类。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化语言模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
#model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

```python
import json

def pretty_print(step):
    """
    以格式化的 JSON 形式美观地打印 `step` 输出。
    
    参数：
        step：代理流输出的步骤。
    """
    # 将步骤转换为带有缩进的 JSON 字符串
    step_json = json.dumps(step, indent=4, default=str)
    print(step_json)
```

---

## **工具集成**

### **`ToolNode` 类**

此类表示图中的一个节点，根据最后一条 `AIMessage` 中的工具调用来执行工具。如果请求了多个工具调用，它会并行运行工具。

**主要特性**：
- 根据语言模型的工具调用动态执行工具。
- 优雅地处理工具错误。
- 可集成到 `StateGraph` 中以实现复杂工作流。

```python
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def check_weather(location: str) -> str:
    """调用以检查当前天气。"""
    return f"{location} 的天气总是晴朗的"

tool_node = ToolNode(tools=[check_weather])

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "check_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

result = tool_node.invoke({"messages": [message_with_single_tool_call]})
print(result)
```

### **`load_tools` 函数**

此函数按名称加载预定义工具。这些工具允许代理与外部资源（如 API、数据库和文件系统）交互。

**主要特性**：
- 根据名称动态加载工具。
- 支持某些工具的可选语言模型集成。
- 允许谨慎启用危险工具（例如具有提升权限的工具）。

```python
from langchain_community.agent_toolkits.load_tools import load_tools

# 加载工具
tools = load_tools(["ddg-search", "graphql", "arxiv"], 
                   llm=model, 
                   graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")
print(tools)
```

### **`StructuredTool` 类**

此方法从 Python 函数创建自定义工具。它适用于定义具有特定输入和输出模式的工具。

**主要特性**：
- 将函数转换为具有名称、描述和模式的工具。
- 支持同步和异步函数。
- 可从函数签名和文档字符串推断模式。

```python
from langchain_core.tools.structured import StructuredTool

def add(a: int, b: int) -> int:
    """将两个数字相加。"""
    return a + b

tool = StructuredTool.from_function(add)

# 正确方式：以字典形式传递参数
result = tool.run({"a": 1, "b": 2})
print(result)
```

---

## **使用内置工具**

### **示例 1：使用 DuckDuckGo 和 Arxiv 的基础代理**

此示例展示如何创建一个基础代理，使用内置工具（如 **DuckDuckGo 搜索** 和 **Arxiv**）搜索信息并总结结果。代理使用语言模型 (LLM) 和一个封装工具执行的 `ToolNode` 初始化。

```python
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode, create_react_agent

# 步骤 1：加载工具
tools = load_tools(["ddg-search", "arxiv"], llm=model)

# 步骤 2：创建 ToolNode
tool_node = ToolNode(tools=tools, name="tools")

# 步骤 3：创建 React 代理
agent = create_react_agent(model, tool_node)

# 步骤 4：运行代理
inputs = {"messages": [("user", "搜索量子计算的最新论文并总结排名第一的结果。")]}
for step in agent.stream(inputs):
    pretty_print(step)
```

### **示例 2：中断和检查点**

此示例展示如何为代理工作流添加**中断**和**检查点**。中断允许代理在特定节点（如执行工具之前或之后）暂停执行，而检查点使代理能够在交互中持久化状态。

```python
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 步骤 1：加载工具并创建 ToolNode
tools = load_tools(["ddg-search", "arxiv", "graphql"], 
                   llm=model, 
                   graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")
tool_node = ToolNode(tools=tools, name="tools")

# 步骤 2：创建带有中断和检查点的 React 代理
agent = create_react_agent(
    model,
    tool_node,
    interrupt_before=["tools"],  # 在执行工具前暂停
    checkpointer=MemorySaver()   # 启用检查点以保存聊天记忆
)

# 步骤 3：运行代理
inputs = {"messages": [("user", "搜索量子计算的最新论文并总结排名第一的结果。")]}
config = {"configurable": {"thread_id": "thread-1"}}
for step in agent.stream(inputs, config):
    pretty_print(step)
```

### **示例 3：跨线程记忆**

此示例介绍了使用 `InMemoryStore` 的**跨线程记忆**。该存储允许代理在多个线程或对话中持久化数据，支持用户特定记忆或共享上下文等功能。

```python
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 步骤 1：定义内存存储
store = InMemoryStore()

# 步骤 2：加载工具并创建 ToolNode
tools = load_tools(["ddg-search", "arxiv", "graphql"], 
                   llm=model, 
                   graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")
tool_node = ToolNode(tools=tools, name="tools")

# 步骤 3：创建带有跨线程记忆的 React 代理
agent = create_react_agent(
    model,
    tool_node,
    store=store,
    checkpointer=MemorySaver()
)

# 步骤 4：运行代理
inputs = {"messages": [("user", "搜索量子计算的最新论文并总结排名第一的结果。")]}
config = {"configurable": {"thread_id": "thread-1", "user_id": "123"}}
for step in agent.stream(inputs, config):
    pretty_print(step)
```

### **示例 4：复杂提示和状态修改器**

此示例展示如何使用**状态修改器**来自定义语言模型的输入。使用 `ChatPromptTemplate` 定义复杂提示，状态修改器函数动态为语言模型准备输入。

```python
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

# 步骤 1：定义状态修改器
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个名叫 Fred 的helpful助手。"),
    ("placeholder", "{messages}"),
    ("user", "记得要礼貌！"),
])

def state_modifier(state):
    return prompt.invoke({"messages": state["messages"]})

# 步骤 2：加载工具并创建 ToolNode
tools = load_tools(["ddg-search", "arxiv", "graphql"], 
                   llm=model, 
                   graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")
tool_node = ToolNode(tools=tools, name="tools")

# 步骤 3：创建带有状态修改器的 React 代理
agent = create_react_agent(
    model,
    tool_node,
    state_modifier=state_modifier
)

# 步骤 4：运行代理
inputs = {"messages": [("user", "搜索量子计算的最新论文并总结排名第一的结果。")]}
for step in agent.stream(inputs):
    pretty_print(step)
```

---

## **使用自定义工具**

### **示例 1：带有自定义工具的基础代理**

此示例展示如何使用自定义工具创建基础代理。我们使用 `StructuredTool.from_function` 定义三个自定义工具（`get_current_time`、`calculate_tip` 和 `get_weather`），然后将它们封装在 `ToolNode` 中，并集成到 `create_react_agent` 工作流中。

```python
from langchain_core.tools.structured import StructuredTool
from datetime import datetime
import pytz

# 工具 1：获取指定时区的当前时间
def get_current_time(timezone: str) -> str:
    """返回指定时区的当前时间。"""
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"{timezone} 的当前时间是 {current_time}。"

# 工具 2：计算小费金额
def calculate_tip(bill_total: float, tip_percentage: float) -> str:
    """根据账单总额和小费百分比计算小费金额。"""
    tip_amount = bill_total * (tip_percentage / 100)
    total_amount = bill_total + tip_amount
    return f"小费：${tip_amount:.2f}，总额：${total_amount:.2f}"

# 工具 3：获取某地天气（模拟）
def get_weather(location: str) -> str:
    """模拟获取指定地点的天气。"""
    return f"{location} 的天气是晴天，最高温度 75°F。"

# 创建 StructuredTool 实例
time_tool = StructuredTool.from_function(get_current_time)
tip_tool = StructuredTool.from_function(calculate_tip)
weather_tool = StructuredTool.from_function(get_weather)

# 将工具组合成列表
tools = [time_tool, tip_tool, weather_tool]

# 创建 ToolNode
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools=tools, name="tools")

# 创建 React 代理
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# 创建代理
agent = create_react_agent(model, tool_node)

# 运行代理
inputs = {"messages": [("user", "纽约现在的天气如何？")]}
for step in agent.stream(inputs):
    pretty_print(step)
```

### **示例 2：中断和检查点**

此示例展示如何为代理工作流添加**中断**和**检查点**。中断允许代理在特定节点（如执行工具之前或之后）暂停执行，以便用户确认或额外处理。检查点使用 `MemorySaver` 在交互中持久化代理状态，支持聊天记忆等功能。

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 创建带有中断和检查点的代理
agent = create_react_agent(
    model,
    tool_node,
    interrupt_before=["tools"],  # 在执行工具前暂停
    checkpointer=MemorySaver()   # 启用检查点以保存聊天记忆
)

# 运行代理
inputs = {"messages": [("user", "旧金山的天气如何？")]}
config = {"configurable": {"thread_id": "thread-1"}}
for step in agent.stream(inputs, config):
    pretty_print(step)
```

### **示例 3：跨线程记忆**

此示例介绍了使用 `InMemoryStore` 的**跨线程记忆**。该存储允许代理在多个线程或对话中持久化数据，支持用户特定记忆或共享上下文等功能。

```python
from langgraph.store.memory import InMemoryStore

# 创建内存存储
store = InMemoryStore()

# 创建带有跨线程记忆的 React 代理
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 创建带有跨线程记忆的代理
agent = create_react_agent(
    model,
    tool_node,
    store=store,
    checkpointer=MemorySaver()
)

# 运行代理
inputs = {"messages": [("user", "伦敦现在的天气如何？")]}
config = {"configurable": {"thread_id": "thread-1", "user_id": "123"}}
for step in agent.stream(inputs, config):
    pretty_print(step)
```

### **示例 4：复杂提示和状态修改器**

此示例展示如何使用**状态修改器**来自定义语言模型的输入。使用 `ChatPromptTemplate` 定义复杂提示，状态修改器函数动态为语言模型准备输入。

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个名叫 Fred 的helpful助手。"),
    ("placeholder", "{messages}"),
    ("user", "记得要礼貌！"),
])

# 定义状态修改器
def state_modifier(state):
    return prompt.invoke({"messages": state["messages"]})

# 创建带有状态修改器的 React 代理
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# 创建带有状态修改器的代理
agent = create_react_agent(
    model,
    tool_node,
    state_modifier=state_modifier
)

# 运行代理
inputs = {"messages": [("user", "东京的天气如何？")]}
for step in agent.stream(inputs):
    pretty_print(step)
```

---

## **结论**

通过将 `ToolNode`、`load_tools` 和 `StructuredTool` 与 `create_react_agent` 集成，为构建能够与工具和外部系统交互的智能代理提供了坚实的基础。这些组件使开发者能够创建不仅能理解和生成自然语言，还能执行任务、检索信息并提供结构化响应的代理。无论您是构建对话助手、研究工具还是自动化工作流，理解这些对象如何协同工作，都能赋予您设计能够应对现实世界挑战的复杂、支持工具的代理的能力。随着 AI 的不断进步，掌握这些工具将是释放智能系统全部潜力的关键。