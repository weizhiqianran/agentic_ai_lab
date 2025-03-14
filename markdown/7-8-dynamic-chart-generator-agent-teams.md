# 多代理网络架构处理复杂任务

## 引言

在本教程中，我们将探索一种通过利用**多代理网络架构**来处理复杂任务的强大方法。其核心思想是采用“分而治之”的策略，为特定任务或领域创建专门的代理，并将任务路由到合适的“专家”代理。这种方法通过将复杂任务分解成较小的、可管理的子任务，每个子任务由具备相应专长的代理处理，从而实现高效的问题解决。

本教程将展示如何使用**LangGraph**（一个用于构建多代理工作流的框架）来实现这样的系统。我们将定义专门的代理，例如用于检索数据的研究代理和用于创建可视化图表的图表生成代理，并将它们连接成一个协作网络。通过本教程的学习，您将了解如何设计和部署一个多代理系统，以有效应对复杂的现实世界问题。

## 安装与设置

在开始之前，我们需要安装必要的库，包括LangChain、LangGraph以及代理运行所需的其它依赖项。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU duckduckgo_search
```

## 定义工具和代理

在本节中，我们将定义多代理网络中使用的工具和代理。工具包括DuckDuckGo搜索工具和用于执行代码的Python REPL工具。我们还将定义代理的系统提示。

代码首先导入必要的库，包括用于网页搜索的`DuckDuckGoSearchRun`和用于执行Python代码的`PythonREPL`。然后定义了两个工具：

1. **DuckDuckGo搜索工具**：
   - 该工具使用DuckDuckGo执行网页搜索。
   - 它接受一个`query`（查询）和一个可选的`max_results`参数（默认为5），并以字符串形式返回搜索结果。

2. **Python REPL工具**：
   - 该工具使用Python REPL在本地执行Python代码。
   - 它接受一个`code`（代码）字符串作为输入，执行代码并返回结果，如果执行失败则返回错误信息。
   - 包含警告，指出在未正确沙箱化的情况下本地执行代码可能不安全。

此外，`make_system_prompt`函数生成AI助手的系统提示。该提示提供协作上下文，指示助手使用提供的工具，并在任务完成时以`"FINAL ANSWER"`为前缀回复。该函数将附加指令（`suffix`）添加到基础提示中以实现定制。

这些工具和提示构成了多代理网络的基础，使代理能够在有效协作的同时检索数据和执行代码。

```python
from typing import Annotated
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# 定义DuckDuckGo搜索工具
@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """使用DuckDuckGo执行搜索并返回结果。
    
    参数：
        query (str): 要执行的搜索查询。
        max_results (int, 可选): 返回的最大结果数。默认为5。
    
    返回：
        str: 以字符串形式返回的搜索结果。
    """
    search = DuckDuckGoSearchRun()  # 初始化DuckDuckGoSearchRun
    results = search.run(query)     # 使用run方法进行搜索
    return str(results)

# 警告：本地执行代码在未沙箱化时可能不安全
repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "用于生成图表的Python代码"]):
    """使用Python REPL（读取-求值-打印循环）执行Python代码。
    
    参数：
        code (str): 要执行的Python代码。
    
    返回：
        str: 执行代码的结果，或执行失败时的错误信息。
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"执行失败。错误：{repr(e)}"
    result_str = f"成功执行：\n```python\n{code}\n```\n标准输出：{result}"
    return (
        result_str + "\n\n如果所有任务已完成，请以FINAL ANSWER回复。"
    )

# 创建图并定义代理节点
def make_system_prompt(suffix: str) -> str:
    """为AI助手生成系统提示。
    
    参数：
        suffix (str): 添加到基础系统提示中的额外上下文或指令。
    
    返回：
        str: 完整的系统提示。
    """
    return (
        "您是一个乐于助人的AI助手，与其他助手协作。"
        "使用提供的工具推进问题的解答。"
        "如果无法完全回答，没关系，另一个拥有不同工具的助手会在您停下的地方继续帮助。"
        "尽可能执行您能做的以取得进展。"
        "如果您或其他助手获得了最终答案或交付物，"
        "请以FINAL ANSWER作为回复前缀，以便团队知道可以停止。"
        f"\n{suffix}"
    )
```

## 创建多代理网络

在本节中，我们将使用LangGraph创建多代理网络。我们定义研究代理和图表生成代理，然后创建连接这些代理的图。

代码首先导入必要的库，包括用于消息处理的`langchain_core.messages`、用于语言模型的`langchain_openai`和`langchain_anthropic`，以及用于构建多代理图的`langgraph`。加载Anthropic API密钥以验证语言模型（`claude-3-5-sonnet-latest`），该模型为代理提供动力。

接下来，我们使用`get_next_node`函数定义节点之间的转换逻辑。该函数检查最后一条消息是否包含`"FINAL ANSWER"`，以决定是终止图还是继续前往下一个节点。

然后创建**研究代理**，专门使用DuckDuckGo搜索工具检索数据。它被赋予特定任务（`research_task`）和系统提示，仅专注于研究。`research_node`函数执行此代理，处理搜索结果，并转换到`chart_node`进行进一步处理。

**图表生成代理**负责使用Python REPL工具创建可视化内容。它遵循详细指令（`chart_task`），使用`seaborn`和`plotly`等库生成清晰、视觉上吸引人的图表。`chart_node`函数执行此代理，处理图表生成，如果需要额外研究，则转换回`research_node`。

最后，使用`StateGraph`和`MessagesState`构建图以管理对话状态。添加两个节点：用于数据检索的`research_node`和用于图表生成的`chart_node`。图从`research_node`开始，在研究完成后转换到`chart_node`。然后将图编译为可执行工作流。可选地，可以使用Mermaid.js可视化图，但此步骤需要额外依赖。

这种模块化方法通过协作的专门代理在结构化工作流中高效处理复杂任务。

```python
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from kaggle_secrets import UserSecretsClient

# 加载Anthropic API密钥
my_api_key = UserSecretsClient().get_secret("my-anthropic-api-key")

#llm = ChatOpenAI(model="gpt-o1-mini", api_key=my_api_key)
llm = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=my_api_key)

def get_next_node(last_message: BaseMessage, goto: str):
    """根据最后一条消息确定下一个要转换的节点。
    
    参数：
        last_message (BaseMessage): 对话中的最后一条消息。
        goto (str): 如果未找到最终答案，默认要转换到的节点。
    
    返回：
        str: 要转换到的下一个节点，如果找到最终答案则返回END。
    """
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

# 研究代理和节点
research_task = "您只能进行研究。您正在与一位图表生成同事合作。"
research_agent = create_react_agent(llm, tools=[duckduckgo_search], state_modifier=make_system_prompt(research_task))

def research_node(state: MessagesState) -> Command[Literal["chart_node", END]]:
    """执行研究节点，使用DuckDuckGo搜索工具进行研究。
    
    参数：
        state (MessagesState): 当前对话状态。
    
    返回：
        Command: 包含更新状态和下一个要转换的节点的命令对象。
    """
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_node")
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="research_node"
    )
    return Command(update={"messages": result["messages"]}, goto=goto)

# 图表生成代理和节点
chart_task = """使用seaborn和plotly创建清晰且视觉上吸引人的图表。遵循以下规则：
1. 添加标题、带标签的轴（包括单位），并在需要时添加图例。
2. 使用`sns.set_context("notebook")`确保文本可读，并使用`sns.set_theme()`或`sns.set_style("whitegrid")`等主题。
3. 使用易于访问的颜色调色板，如`sns.color_palette("husl")`。
4. 选择合适的图表类型：`sns.lineplot()`、`sns.barplot()`或`sns.heatmap()`。
5. 为关键点添加注释（例如“2020年高峰”）以提高清晰度。
6. 确保图表的宽度和显示分辨率不超过1000像素。
7. 使用`plt.show()`显示图表。
目标：生成准确、引人入胜且易于理解的图表。"""
chart_agent = create_react_agent(llm, [python_repl_tool], state_modifier=make_system_prompt(chart_task))

def chart_node(state: MessagesState) -> Command[Literal["research_node", END]]:
    """执行图表节点，使用Python REPL工具生成图表。
    
    参数：
        state (MessagesState): 当前对话状态。
    
    返回：
        Command: 包含更新状态和下一个要转换的节点的命令对象。
    """
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "research_node")
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_node"
    )
    return Command(update={"messages": result["messages"]}, goto=goto)

# 定义图
from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("research_node", research_node)
workflow.add_node("chart_node", chart_node)

workflow.add_edge(START, "research_node")
graph = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # 此步骤需要额外依赖，是可选的
    pass
```

## 美化打印事件消息以进行调试

`print_pretty`函数旨在以可读且结构化的方式格式化和显示事件中的消息，便于调试或记录多代理网络中的交互。其工作原理如下：

1. **输入**：函数接受一个包含代理生成消息（例如`research_node`或`chart_node`）的`event`字典作为输入。
2. **节点键检查**：检查事件是否包含来自特定节点（`research_node`或`chart_node`）的消息。如果找到，则继续处理消息。
3. **消息提取**：
   - 对事件中的每条消息，提取消息类型（例如`HumanMessage`、`AIMessage`）使用`message.__class__.__name__`。
   - 提取消息的`content`并进行格式化。如果内容是列表，则作为项目列表处理；如果内容是字符串，则用引号包裹以提高可读性。
4. **附加字段**：
   - 函数还提取并打印附加元数据，如`additional_kwargs`、`response_metadata`和`message.id`，以提供更多消息上下文。
5. **格式化输出**：
   - 每条消息以结构化格式打印，显示其类型、内容、元数据和ID。
   - 输出缩进并格式化以提高清晰度，便于阅读和分析。
6. **分隔符**：
   - 在每个节点的消息后打印分隔线（`"-" * 120`），以在视觉上区分不同节点。
7. **回退**：
   - 如果事件中未找到消息，函数打印“未在事件中找到消息”。

```python
def print_pretty(event):
    """美化打印事件消息以用于调试或日志记录。
    
    参数：
        event (dict): 包含研究或图表节点消息的事件。
    """
    # 检查事件是否包含'research_node'或'chart_node'
    for node_key in ["research_node", "chart_node"]:
        if node_key in event:
            messages = event[node_key].get("messages", [])
            print(f"{node_key}: [")
            for message in messages:
                # 提取消息类型（HumanMessage、AIMessage等）
                message_type = message.__class__.__name__

                # 提取消息内容
                content = message.content
                if isinstance(content, list):
                    content = [item for item in content]  # 处理列表内容（例如使用工具的AIMessage）
                elif isinstance(content, str):
                    content = f'"{content}"'  # 将字符串内容用引号包裹

                # 提取附加字段
                additional_kwargs = message.additional_kwargs
                response_metadata = message.response_metadata
                message_id = message.id

                # 以所需格式打印消息
                print(f"    {message_type}(")
                print(f"        content={content},")
                print(f"        additional_kwargs={additional_kwargs},")
                print(f"        response_metadata={response_metadata},")
                print(f"        id='{message_id}'")
                print( "    ),")
            print("]")
            print("-" * 120)
            return

    print("未在事件中找到消息。")
```

## 示例1：美国人口随时间增长

在此示例中，我们使用多代理网络检索过去50年的美国人口数据，并生成带有经济衰退等重大事件注释的折线图。

```python
# 调用图
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="首先，获取过去50年的美国人口数据。"
                "然后，创建带有经济衰退等重大事件注释的折线图。"
                "使用numpy.polyfit添加趋势线。"
                "完成图表后结束。"
            )
        ],
    },
    {"recursion_limit": 150},
)

# 使用print_pretty打印事件
for event in events:
    print_pretty(event)
```

## 示例2：全球各国CO2排放

在此示例中，我们检索过去20年十大排放国的CO2排放数据，并使用Plotly创建交互式堆叠面积图。

```python
# 调用图
prompt_message = ("""首先，检索过去20年十大排放国的CO2排放数据。
然后，使用Seaborn创建堆叠面积图。
通过加粗线条并添加注释突出每年排放最高的国家。
确保图表包含标题、带标签的轴和图例。
使用自定义颜色调色板以获得更好的可视化效果。
图表创建完成后，使用`plt.show()`显示，并确保宽度不超过1000像素。""")

# 调用图
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content=prompt_message
            )
        ],
    },
    {"recursion_limit": 150},
)

# 使用print_pretty打印事件
for event in events:
    print_pretty(event)
```

## 示例3：天气数据可视化

在此示例中，我们检索过去5年纽约市的温度和降水数据，并创建带有极端天气事件（如热浪和大雨）阴影区域的双轴图表。

```python
# 调用图
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="首先，获取过去5年纽约市的温度和降水数据。"
                "然后，创建双轴图表，左侧y轴为温度，右侧y轴为降水。"
                "为极端天气事件（如热浪和大雨）添加阴影区域。"
                "完成图表后结束。"
            )
        ],
    },
    {"recursion_limit": 150},
)

# 使用print_pretty打印事件
for event in events:
    print_pretty(event)
```

## 结论

本笔记本展示了如何使用`LangGraph`实现`多代理网络架构`。通过将任务分配给专门的代理，我们可以有效处理需要多个专业领域知识的复杂任务。这种方法高度灵活，可适应各种用例，从数据检索到可视化。