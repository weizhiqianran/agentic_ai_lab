# **使用 PythonREPLTool 和大语言模型（LLMs）自动化数据分析**

## **引言**

在当今数据驱动的世界中，企业和个人都高度依赖数据分析来做出明智的决策。然而，数据分析的过程——加载数据集、执行计算、生成可视化图表——往往耗时且需要专业技能。这时，大型语言模型（LLMs），如 OpenAI 的 GPT-4，彻底改变了我们与技术互动的方式。通过将 LLMs 的自然语言理解能力与自定义工具结合，我们可以自动化复杂的工作流程，让数据分析对每个人来说都更加易于访问。

本示例展示了如何使用 **LangChain**（一个强大的框架，用于构建 LLM 驱动的应用程序）创建一个对话式数据分析助手。该助手将处理一个完整的数据分析任务：加载包含销售数据的 CSV 文件，计算每个产品的总销售额，生成条形图以可视化结果，并提供最终总结——所有这些都通过一个简单的对话界面实现。通过将 LangChain 与自定义 Python 工具集成，我们可以弥合自然语言与数据分析之间的差距，使用户能够以更直观、高效的方式与数据互动。

该工作流程设计灵活且可扩展，适用于广泛的数据分析任务。无论您是希望自动化重复性任务的业务分析师，还是探索 LLMs 潜力的开发者，本示例都为构建智能、工具增强的对话代理提供了实用基础。让我们深入了解创建此助手的逐步过程，看看它如何改变我们处理数据的方式。

---

## **准备工作**

### **安装所需库**
本节将安装使用 LangChain、OpenAI 嵌入、Anthropic 模型、DuckDuckGo 搜索和其他实用工具所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型和 API 的集成。
- `langchain-anthropic`：支持与 Anthropic 模型和 API 的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。
- `langgraph`：用于在 LangChain 中构建和可视化基于图的工作流程的库。
- `duckduckgo-search`：支持编程访问 DuckDuckGo 的搜索功能。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU duckduckgo-search
```

### **创建包含销售数据的 CSV 文件**
此代码块将创建一个名为 `sales_data.csv` 的 CSV 文件，其中包含三个产品在五天内的销售数据。

```python
import pandas as pd

# 定义数据
data = {
    "Date": [
        "2023-10-01", "2023-10-01", "2023-10-01",
        "2023-10-02", "2023-10-02", "2023-10-02",
        "2023-10-03", "2023-10-03", "2023-10-03",
        "2023-10-04", "2023-10-04", "2023-10-04",
        "2023-10-05", "2023-10-05", "2023-10-05",
    ],
    "Product": [
        "Product A", "Product B", "Product C",
        "Product A", "Product B", "Product C",
        "Product A", "Product B", "Product C",
        "Product A", "Product B", "Product C",
        "Product A", "Product B", "Product C",
    ],
    "Sales": [
        1000, 500, 1500,
        1200, 600, 1800,
        900, 400, 1300,
        1100, 550, 1600,
        1300, 700, 2000,
    ],
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 保存为 CSV 文件
df.to_csv("sales_data.csv", index=False)

print("sales_data.csv 创建成功！")
```

## **步骤 1：定义自定义工具**
使用 `@tool` 装饰器定义用于数据分析的自定义工具。这些工具将处理特定任务，如加载 CSV 文件和生成图表。

```python
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt

@tool
def csv_loader(file_path: str) -> str:
    """将 CSV 文件加载到 pandas DataFrame 中并以 JSON 格式返回。"""
    return pd.read_csv(file_path).to_json(orient="split")  # 使用 "split" 方向以获得更好的兼容性

@tool
def plot(data_json: str, plot_type: str, x: str, y: str) -> str:
    """从 pandas DataFrame 生成图表并将其保存为图像。"""
    try:
        df = pd.read_json(data_json, orient="split")  # 使用 "split" 方向以获得更好的兼容性
        if plot_type == "line":
            df.plot(x=x, y=y, kind="line")
        elif plot_type == "bar":
            df.plot(x=x, y=y, kind="bar")
        plt.savefig("plot.png")
        return "图表已保存为 plot.png"
    except Exception as e:
        return f"生成图表时出错：{str(e)}"
```

## **步骤 2：初始化 LLM**
使用 `gpt-4o-mini` 模型初始化 `ChatOpenAI`，并将温度设置为 `0` 以获得确定性响应。通过 `kaggle_secrets` 安全获取 API 密钥。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
#model = ChatAnthropic(model="claude-3-5-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

## **步骤 3：初始化工具**
初始化工具，包括内置的 `PythonREPLTool` 和自定义工具（`csv_loader` 和 `plot`）。

```python
from langchain_experimental.tools.python.tool import PythonREPLTool

python_tool = PythonREPLTool()
csv_loader_tool = csv_loader
plot_tool = plot
```

## **步骤 4：将工具绑定到 LLM**
使用 `bind_tools()` 方法将工具绑定到 `ChatOpenAI` 模型。这使模型能够根据用户请求动态调用这些工具。

```python
model_with_tools = model.bind_tools([python_tool, csv_loader_tool, plot_tool])
```

## **步骤 5：定义用户提示**
定义用户提示，要求助手执行以下操作：
1. 加载 CSV 文件。
2. 计算每个产品的总销售额。
3. 生成按产品显示总销售额的条形图。
4. 显示每个产品的总销售额并展示图表。

```python
user_prompt = """
我有一个名为 'sales_data.csv' 的 CSV 文件，包含以下列：'Date'、'Product'、'Sales'。
1. 加载 CSV 文件。
2. 计算每个产品的总销售额。
3. 生成按产品显示总销售额的条形图。
4. 告诉我每个产品的总销售额并展示图表。
"""
```

## **步骤 6：创建对话历史**
通过 `SystemMessage` 设置助手的角色，并使用包含用户提示的 `HumanMessage` 创建对话历史。

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

messages = [
    SystemMessage(content="你是一个有用的数据分析助手。"),
    HumanMessage(content=user_prompt),
]
```

## **步骤 7：迭代处理工具调用**
迭代处理工具调用，直到 LLM 生成最终响应。这包括：
1. 将对话历史发送给 LLM。
2. 检查响应中是否有工具调用。
3. 调用相应的工具并更新对话历史。
4. 当不再生成工具调用时退出循环。

```python
max_iterations = 5  # 防止无限循环
for _ in range(max_iterations):
    response = model_with_tools.invoke(messages)

    # 检查响应中是否包含工具调用
    if not response.additional_kwargs.get("tool_calls"):
        # 没有更多的工具调用，退出循环
        break

    # 处理工具调用
    print("正在处理工具调用...")  # 调试信息
    for tool_call in response.additional_kwargs.get("tool_calls", []):
        tool_name = tool_call["function"]["name"]
        tool_args = eval(tool_call["function"]["arguments"])

        if tool_name == "csv_loader":
            # 加载 CSV 文件
            tool_result = csv_loader_tool.invoke(tool_args["file_path"])
            messages.append(AIMessage(content="", additional_kwargs={"tool_calls": [tool_call]}))
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        elif tool_name == "Python_REPL":
            # 执行 Python 代码（例如，计算总销售额）
            tool_result = python_tool.run(tool_args["query"])
            messages.append(AIMessage(content="", additional_kwargs={"tool_calls": [tool_call]}))
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        elif tool_name == "plot":
            # 生成图表
            tool_result = plot_tool.invoke(tool_args)  # 将 tool_args 作为输入传递
            messages.append(AIMessage(content="", additional_kwargs={"tool_calls": [tool_call]}))
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
```

## **步骤 8：打印最终响应**
打印 LLM 的最终响应，其中包括每个产品的总销售额以及图表已保存的确认信息。

```python
print("最终答案：\n", response.content)
```

---

## **结论**

本示例展示了将大型语言模型与自定义工具集成以自动化和简化数据分析任务的变革潜力。通过利用 LangChain，我们构建了一个对话式助手，能够动态加载数据、执行计算并生成可视化图表——所有这些都通过自然语言交互实现。该工作流程不仅高效，而且高度适应性强，可扩展到更复杂的任务或集成到更大的系统中。

使用自然语言与数据交互的能力为可访问性和生产力开辟了新的可能性。用户无需编写复杂代码或操作专业软件即可分析数据；他们只需用简单的英语描述需求，助手就会处理其余部分。随着 LLMs 的不断发展，其与特定领域工具的集成将解锁更多创新机会，使高级数据分析能力惠及更广泛的受众。

无论是分析销售数据、可视化趋势还是自动化重复性任务，这种方法都提供了一种强大且用户友好的数据交互方式。通过结合 LLMs 和自定义工具的优势，我们可以创建智能系统，帮助用户轻松做出数据驱动的决策。数据分析的未来是对话式的，而这个示例只是一个开始。