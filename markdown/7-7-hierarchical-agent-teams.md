# 层级代理团队

## 引言

随着任务的复杂性或规模增加，使用单一监督节点进行管理可能会变得低效。单一监督者可以在工作者之间路由任务，但当单个任务需要复杂的子任务或工作者数量变得过多时会发生什么？

在这种情况下，层级结构成为一个强大的解决方案。通过将任务分解为子任务，并将工作者组织成拥有各自监督者的团队，我们可以创建一个健壮且可扩展的系统。每个团队处理特定领域的任务，中层监督者协调努力，然后向顶级监督者报告。

这种方法能够实现高效的任务分配、更好的资源管理和复杂工作流程的改进可扩展性。

**在本笔记本中，我们将：**

- 定义代理访问网络数据和管理文件的工具
- 实现实用工具以简化任务工作流的创建
- 开发专注于网络研究和文档撰写的团队
- 将这些组件组成一个由监督者和工作者构成的层级系统

## 安装所需包

首先，我们需要安装构建层级代理系统所需的必要包。这些包包括 LangChain 的各种组件、LangGraph、ChromaDB、DuckDuckGo 搜索和 Wikipedia 集成。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU chromadb
!pip install -qU duckduckgo_search
!pip install -qU wikipedia
```

## 导入模块并设置环境

在本节中，我们导入所有必要的模块并设置环境变量。我们还定义了代理将使用的工具，例如网页抓取、文档撰写和 Python REPL 执行。

```python
import os

# 为 HTTP 请求设置自定义用户代理
os.environ["USER_AGENT"] = "MyApp/1.0 (https://myapp.com; contact@myapp.com)"

from typing import Annotated, List, Dict, Optional, Literal
from pathlib import Path
from tempfile import TemporaryDirectory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
from IPython.display import Image, display
```

## 为研究团队创建工具

在这里，我们定义研究团队将使用的工具，包括使用 DuckDuckGo 的搜索工具和从提供 URL 中提取信息的网页抓取工具。

```python
# --------------------------------------------------------------------------------------------------------
# 创建工具
# --------------------------------------------------------------------------------------------------------

# 研究团队工具
# 使用 DuckDuckGo 的搜索工具
search_tool = DuckDuckGoSearchRun()

@tool
def scrape_webpages(urls: List[str]) -> str:
    """
    使用 WebBaseLoader 抓取提供的网页以获取详细信息。
    参数：
        urls (List[str]): 要抓取的 URL 列表。
    返回：
        str: 以结构化格式包含抓取内容的字符串。
    """
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
```

## 为文档撰写团队创建工具

文档撰写团队需要工具来创建大纲、读取、撰写和编辑文档。我们设置这些工具并为文件操作创建一个临时目录。以下工具在临时目录中实现：

1. **`create_outline`**：  
   从一系列要点创建大纲并保存到指定文件中。  
   示例：`create_outline(["引言", "方法论"], "outline.txt")`。

2. **`read_document`**：  
   从文件中读取内容，可选择指定起始和结束行以进行部分读取。  
   示例：`read_document("outline.txt", start=0, end=2)`。

3. **`write_document`**：  
   将提供的内容写入指定文件。  
   示例：`write_document("这是内容。", "document.txt")`。

4. **`edit_document`**：  
   通过在特定行号（从 1 开始计数）插入文本来编辑文档。  
   示例：`edit_document("document.txt", {2: "这是插入的一行。"})`。

每个工具在临时目录中操作，并返回带有文件路径或读取内容的确认消息。这些工具使文档撰写团队能够高效执行基于文件的操作。

```python
# 文档撰写团队工具
# 为文件操作创建一个临时目录
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

@tool
def create_outline(
    points: Annotated[List[str], "主要要点或部分的列表。"],
    file_name: Annotated[str, "保存大纲的文件路径。"],
) -> Annotated[str, "保存的大纲文件路径。"]:
    """
    创建并将大纲保存到文件中。

    result = create_outline(["引言", "方法论", "结果", "结论"], "outline.txt")

    参数：
        points (List[str]): 主要要点或部分的列表。
        file_name (str): 保存大纲的文件路径。
    返回：
        str: 带有保存大纲文件路径的确认消息。
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"大纲已保存到 {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "读取文档的文件路径。"],
    start: Annotated[Optional[int], "起始行。默认值为 0"] = None,
    end: Annotated[Optional[int], "结束行。默认值为 None"] = None,
) -> str:
    """
    从文件中读取指定文档。

    content = read_document("outline.txt", start=0, end=2)

    参数：
        file_name (str): 读取文档的文件路径。
        start (Optional[int]): 起始行。默认值为 0。
        end (Optional[int]): 结束行。默认值为 None。
    返回：
        str: 包含文档中指定行的字符串。
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "要写入文档的文本内容。"],
    file_name: Annotated[str, "保存文档的文件路径。"],
) -> Annotated[str, "保存的文档文件路径。"]:
    """
    将内容写入文件。
    参数：
        content (str): 要写入的文本内容。
        file_name (str): 保存文档的文件路径。
    返回：
        str: 带有保存文档文件路径的确认消息。
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"文档已保存到 {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "要编辑的文档路径。"],
    inserts: Annotated[
        Dict[int, str],
        "键为行号（从 1 开始计数），值为要在该行插入的文本的字典。",
    ],
) -> Annotated[str, "编辑后的文档文件路径。"]:
    """
    通过在特定行号插入文本来编辑文档。

    result = edit_document("document.txt", {2: "这是插入的一行。"})

    参数：
        file_name (str): 要编辑的文档文件路径。
        inserts (Dict[int, str]): 键为行号（从 1 开始计数），值为要插入文本的字典。
    返回：
        str: 带有编辑后文档文件路径的确认消息。
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"错误：行号 {line_number} 超出范围。"
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"文档已编辑并保存到 {file_name}"
```

## 设置 Python REPL 工具

Python REPL 工具允许在代理中执行 Python 代码。这对于根据抓取的数据生成图表或执行计算非常有用。

```python
# 警告：这会在本地执行代码，如果没有沙盒隔离可能不安全
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "要执行以生成图表的 Python 代码。"],
):
    """
    使用 REPL（读取-求值-打印循环）执行 Python 代码。
    参数：
        code (str): 要执行的 Python 代码。
    返回：
        str: 包含执行结果或错误消息的字符串。
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"执行失败。错误：{repr(e)}"
    return f"成功执行：\n```python\n{code}\n```\n标准输出：{result}"
```

## 定义辅助实用工具

我们定义了辅助函数和实用工具，以帮助创建监督节点并为我们的代理设置语言模型（LLM）。

```python
# --------------------------------------------------------------------------------------------------------
# 辅助实用工具
# --------------------------------------------------------------------------------------------------------

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    """
    创建一个将任务路由给工作者的监督节点。

    supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

    参数：
        llm (BaseChatModel): 语言模型（例如 ChatOpenAI）。
        members (list[str]): 工作者名称列表。
    返回：
        str: 一个将任务路由到适当工作者的函数（supervisor_node）。
    """
    options = ["FINISH"] + members
    system_prompt = (
        "您是一个负责管理以下工作者之间对话的监督者："
        f"{members}。根据以下用户请求，"
        "回复接下来行动的工作者。每个工作者将执行一个"
        "任务并返回其结果和状态。完成后，"
        "回复 FINISH。"
    )

    class Router(TypedDict):
        """接下来要路由的工作者。如果不需要工作者，则路由到 FINISH。"""
        next: str # Literal[ ["FINISH"] + members ]

    def supervisor_node(state: MessagesState) -> Command[str]:  # 更改返回类型
        """
        基于 LLM 的路由器。
        参数：
            state (MessagesState): 当前消息状态。
        返回：
            Command[str]: 指示接下来要路由的工作者的命令。
        """
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        return Command(goto=goto)

    return supervisor_node
```

## 获取 API 密钥

此单元格从 Kaggle Secrets 中检索存储的 OpenAI API 密钥，并使用获取的密钥初始化 `ChatOpenAI` 模型。使用 `UserSecretsClient` 安全地获取密钥，确保 API 密钥保持私密。

```python
from kaggle_secrets import UserSecretsClient

# 从 Kaggle Secrets 获取 OpenAI API 密钥
my_api_key = UserSecretsClient().get_secret("my-openai-api-key")

# 使用获取的 API 密钥初始化 ChatOpenAI 模型
llm = ChatOpenAI(model="gpt-4o-mini", api_key=my_api_key)
```

## 设置研究团队

研究团队负责搜索网络并抓取相关信息。我们为搜索和网页抓取创建代理，定义它们各自的节点，并将它们编译成状态图。

```python
# --------------------------------------------------------------------------------------------------------
# 研究团队
# --------------------------------------------------------------------------------------------------------

# 使用 LLM 和搜索工具创建搜索代理
search_agent = create_react_agent(llm, tools=[search_tool])

def search_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    执行搜索代理并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    result = search_agent.invoke(state)
    print(">>> search_node >>>")
    print(result["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< search_node <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # 完成任务后始终路由回监督者
        goto="supervisor",
    )

# 使用 LLM 和 scrape_webpages 工具创建网页抓取代理
web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])

def web_scraper_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    执行网页抓取代理并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    result = web_scraper_agent.invoke(state)
    print(">>> web_scraper_node >>>")
    print(result["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< web_scraper_node <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # 完成任务后始终路由回监督者
        goto="supervisor",
    )

# 为研究团队创建监督节点
research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

# 构建研究团队状态图
research_builder = StateGraph(MessagesState)
research_builder.add_node("supervisor", research_supervisor_node)  # 添加监督节点
research_builder.add_node("search", search_node)  # 添加搜索节点
research_builder.add_node("web_scraper", web_scraper_node)  # 添加网页抓取节点

research_builder.add_edge(START, "supervisor")  # 从监督者开始
research_graph = research_builder.compile()  # 编译图

# 以 Mermaid 图表显示研究团队工作流程
display(Image(research_graph.get_graph().draw_mermaid_png()))

# 使用用户查询流式传输研究团队工作流程
for s in research_graph.stream(
    {"messages": [("user", "泰勒·斯威夫特的下一次巡演是什么时候？")]},
    {"recursion_limit": 100},  # 限制递归以防止无限循环
):
    print(s)  # 打印每一步的状态
    print("-" * 60)  # 可读性分隔符
```

## 设置文档撰写团队

文档撰写团队负责根据大纲创建、读取、编辑和撰写文档。此外，它还可以通过执行 Python 代码生成图表。我们为文档撰写、笔记记录和图表生成创建代理，定义它们的节点，并将它们编译成状态图。

### **文档撰写团队设置**
此块定义了 **文档撰写代理**，负责根据笔记记录者提供的大纲读取、撰写和编辑文档。它使用 LLM（大型语言模型）和工具如 `write_document`、`edit_document` 和 `read_document`。`doc_writing_node` 函数执行代理，打印结果以进行调试，并将输出路由回监督者。

```python
# --------------------------------------------------------------------------------------------------------
# 文档撰写团队
# --------------------------------------------------------------------------------------------------------

# 使用 LLM 和文档相关工具创建文档撰写代理
doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    state_modifier=(
        "您可以根据笔记记录者的大纲读取、撰写和编辑文档。"
        "不要提出后续问题。"
    ),
)

def doc_writing_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    执行文档撰写代理并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    result = doc_writer_agent.invoke(state)
    print(">>> doc_writing_node >>>")
    print(result["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< doc_writing_node <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ]
        },
        # 完成任务后始终路由回监督者
        goto="supervisor",
    )
```

### **笔记记录代理设置**
此块定义了 **笔记记录代理**，负责读取文档并为文档撰写者创建大纲。它使用工具如 `create_outline` 和 `read_document`。`note_taking_node` 函数执行代理，打印结果以进行调试，并将输出路由回监督者。

```python
# 使用 LLM 和大纲相关工具创建笔记记录代理
note_taking_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    state_modifier=(
        "您可以读取文档并为文档撰写者创建大纲。"
        "不要提出后续问题。"
    ),
)

def note_taking_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    执行笔记记录代理并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    result = note_taking_agent.invoke(state)
    print(">>> note_taking_node >>>")
    print(result["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< note_taking_node <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ]
        },
        # 完成任务后始终路由回监督者
        goto="supervisor",
    )
```

### **图表生成代理设置**
此块定义了 **图表生成代理**，负责使用 LLM 和 Python REPL 工具生成图表。它还可以读取文档。`chart_generating_node` 函数执行代理，打印结果以进行调试，并将输出路由回监督者。此外，还创建了一个 **监督节点** 来管理文档撰写者、笔记记录者和图表生成者之间的交互。

```python
# 使用 LLM 和 Python REPL 工具创建图表生成代理
chart_generating_agent = create_react_agent(
    llm, tools=[read_document, python_repl_tool]
)

def chart_generating_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    执行图表生成代理并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    result = chart_generating_agent.invoke(state)
    print(">>> chart_generating_node >>>")
    print(result["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< chart_generating_node <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        # 完成任务后始终路由回监督者
        goto="supervisor",
    )

# 为文档撰写团队创建监督节点
doc_writing_supervisor_node = make_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)
```

### **状态图构建和执行**
此块构建了文档撰写团队工作流程的 **状态图**。它为监督者、文档撰写者、笔记记录者和图表生成者添加节点。工作流程从监督者开始，并将任务路由到适当的代理。图表被编译，并显示 Mermaid 图表以可视化工作流程。最后，使用用户查询（例如，“为关于猫的诗写一个大纲，然后将诗写入磁盘”）执行工作流程，并在每一步打印状态以进行调试。

```python
# 构建文档撰写团队状态图
paper_writing_builder = StateGraph(MessagesState)
paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)  # 添加监督节点
paper_writing_builder.add_node("doc_writer", doc_writing_node)  # 添加文档撰写节点
paper_writing_builder.add_node("note_taker", note_taking_node)  # 添加笔记记录节点
paper_writing_builder.add_node("chart_generator", chart_generating_node)  # 添加图表生成节点

paper_writing_builder.add_edge(START, "supervisor")  # 从监督者开始
paper_writing_graph = paper_writing_builder.compile()  # 编译图

# 以 Mermaid 图表显示文档撰写团队工作流程
display(Image(paper_writing_graph.get_graph().draw_mermaid_png()))
```

```python
# 使用用户查询流式传输文档撰写团队工作流程
for s in paper_writing_graph.stream(
    {
        "messages": [
            (
                "user",
                "为关于猫的诗写一个大纲，然后将诗写入磁盘。",
            )
        ]
    },
    {"recursion_limit": 100},  # 限制递归以防止无限循环
):
    print(s)  # 打印每一步的状态
    print("-" * 60)  # 可读性分隔符
```

## 将团队组成层级结构

为了有效管理多个团队，我们创建了一个顶级监督者，负责监督研究团队和文档撰写团队。这种层级结构允许更好的任务分配和管理。

```python
# --------------------------------------------------------------------------------------------------------
# 添加层级
# --------------------------------------------------------------------------------------------------------

# 为组合团队创建监督节点
teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

def call_research_team(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    调用研究团队并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    response = research_graph.invoke({"messages": state["messages"][-1]})
    print(">>> call_research_team >>>")
    print(response["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< call_research_team <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ]
        },
        goto="supervisor",
    )

def call_paper_writing_team(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    调用文档撰写团队并将结果返回给监督者。
    参数：
        state (MessagesState): 当前消息状态。
    返回：
        Command[Literal["supervisor"]]: 更新状态并路由到监督者的命令。
    """
    response = paper_writing_graph.invoke({"messages": state["messages"][-1]})
    print(">>> call_paper_writing_team >>>")
    print(response["messages"][-1].content)  # 打印最新消息内容以进行调试
    print("<<< call_paper_writing_team <<<")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"
                )
            ]
        },
        goto="supervisor",
    )

# 构建组合团队状态图
super_builder = StateGraph(MessagesState)
super_builder.add_node("supervisor", teams_supervisor_node)  # 添加监督节点
super_builder.add_node("research_team", call_research_team)  # 添加研究团队节点
super_builder.add_node("writing_team", call_paper_writing_team)  # 添加撰写团队节点

super_builder.add_edge(START, "supervisor")  # 从监督者开始
super_graph = super_builder.compile()  # 编译图

# 以 Mermaid 图表显示组合团队工作流程
display(Image(super_graph.get_graph().draw_mermaid_png()))

# 使用用户查询流式传输组合团队工作流程
for s in super_graph.stream(
    {
        "messages": [
            ("user", "研究 AI 代理并撰写一份简短报告。")
        ],
    },
    {"recursion_limit": 150},  # 限制递归以防止无限循环
):
    print(s)  # 打印每一步的状态
    print("-" * 60)  # 可读性分隔符
```

## 结论

通过将代理组织成具有不同层级监督者的层级团队，我们可以高效地管理复杂任务和大量工作者。这种结构增强了可扩展性和可维护性，使 AI 驱动的应用程序能够实现更复杂和有组织的工作流程。