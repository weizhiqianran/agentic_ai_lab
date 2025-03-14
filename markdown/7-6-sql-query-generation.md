# 使用LangGraph进行SQL查询生成与执行

## 简介

本笔记本展示了如何使用LangChain、OpenAI和SQLDatabaseToolkit根据自然语言问题生成并执行SQL查询。工作流程包括初始化SQL数据库、定义查询和模式检索工具，以及设置一个有状态图来管理查询生成与执行过程。本笔记本还包括错误处理和结果格式化，以确保流畅的用户体验。

## 安装与设置

首先，我们需要安装必要的库并导入所需的模块。

这些安装为以下功能设置了必要环境：
- 与**OpenAI**和**Anthropic**语言模型交互。
- 使用**LangChain工具**进行数据库交互和工作流管理。
- 使用`langgraph`构建**有状态工作流**。
- 可选使用如ChromaDB的**向量数据库**进行高级任务。

这确保了笔记本正常运行所需的所有依赖项都可用。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langgraph
!pip install -qU chromadb
```

```python
# 导入必要的模块
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict, Any
```

## 初始化数据库

在本教程中，我们将创建一个SQLite数据库。SQLite是一种轻量级数据库，易于设置和使用。我们将加载chinook数据库，这是一个表示数字媒体商店的样本数据库。有关数据库的更多信息，请参见[此处](https://www.sqlitetutorial.net/sqlite-sample-database/)。

```python
import requests

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
response = requests.get(url)

if response.status_code == 200:
    # 以二进制写入模式打开本地文件
    with open("Chinook.db", "wb") as file:
        # 将响应的内容（文件）写入本地文件
        file.write(response.content)
    print("文件已下载并保存为Chinook.db")
else:
    print(f"文件下载失败。状态码：{response.status_code}")

# 初始化数据库
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"数据库类型：{db.dialect}")
print(f"表名：{db.get_usable_table_names()}")
print(f"艺术家：{db.run('SELECT * FROM Artist LIMIT 5;')}")
```

## 为代理定义工具

接下来，我们定义代理将用来与数据库交互的工具。这些工具包括查询数据库、检索模式和列出表名。

此代码：
1. 安全加载OpenAI API密钥。
2. 初始化一个带有与SQL数据库交互所需工具的`SQLDatabaseToolkit`。
3. 提取用于查询、模式检索和表列举的特定工具。
4. 将语言模型绑定到模式工具以处理与模式相关的任务。

此设置使代理能够：
- 执行SQL查询。
- 检索数据库模式和表信息。
- 检查SQL查询的正确性。
- 使用OpenAI的语言模型协助数据库交互。

```python
from kaggle_secrets import UserSecretsClient

# 加载OpenAI API密钥
my_api_key = UserSecretsClient().get_secret("my-openai-api-key")

# 为代理定义工具
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o-mini", api_key=my_api_key))
tools = toolkit.get_tools()

# 从工具包中提取特定工具
db_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
db_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
db_list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
db_query_checker_tool = next(tool for tool in tools if tool.name == "sql_db_query_checker")

# 将模式工具绑定到模型
db_schema_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key).bind_tools([db_schema_tool])
```

### 定义工作流状态

我们定义工作流的状态，用于跟踪查询生成和执行过程中交换的消息。

```python
# 定义工作流状态
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### 定义初始工具节点

初始工具节点负责通过列出数据库中的表来启动工作流。

```python
# 定义初始工具节点
def initial_tool_node(state: State) -> dict[str, list[AIMessage]]:
    """
    通过创建列出数据库表的工具调用来初始化工作流。
    """
    tool_call_id = "tool_abcd123"  # 硬编码用于调试
    print(f"--- 第一个工具调用节点 ---\n工具调用ID：{tool_call_id}")
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": tool_call_id,
                    }
                ],
            )
        ]
    }
```

### 定义列出表节点

此节点列出数据库中的所有表，并将结果作为ToolMessage返回。

```python
# 定义列出表节点
def list_tables_node(state: State) -> dict[str, list[AIMessage]]:
    """
    列出数据库中的所有表，并将结果作为ToolMessage返回。
    """
    print("--- 列出表节点 ---")
    result = db_list_tables_tool.invoke({})
    print("数据库中的表：", result)

    # 从上一条消息中获取tool_call_id
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    print(f"工具调用ID：{tool_call_id}")
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}
```

### 定义模型获取模式节点

此节点使用模型根据当前状态生成模式请求。

```python
# 定义模型获取模式节点
def model_get_schema_node(state: State) -> dict[str, list[AIMessage]]:
    """
    使用模型根据当前状态生成模式请求。
    """
    print("--- 模型获取模式节点 ---")
    return {"messages": [db_schema_model.invoke(state["messages"])]}
```

### 定义检索模式节点

此节点检索特定表的模式并将其作为ToolMessage返回。

```python
# 定义检索模式节点
def retrieve_schema_node(state: State) -> dict[str, list[AIMessage]]:
    """
    检索特定表的模式并将其作为ToolMessage返回。
    """
    print("--- 检索模式节点 ---")
    table_name = state["messages"][-1].tool_calls[0]["args"]["table_names"]
    result = db_schema_tool.invoke(table_name)
    print(f"表'{table_name}'的模式：\n{result}")

    # 从上一条消息中获取tool_call_id
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    print(f"工具调用ID：{tool_call_id}")

    # 返回带有相同tool_call_id的ToolMessage
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}
```

### 定义SubmitFinalAnswer类

此类表示要提交给用户的最终答案。

```python
# 定义SubmitFinalAnswer类
class SubmitFinalAnswer(BaseModel):
    """
    表示要提交给用户的最终答案的Pydantic模型。
    """
    final_answer: str = Field(..., description="用户的最终答案")
```

## 定义查询生成系统提示

此提示指导模型根据输入问题生成SQL查询。

此代码设置了一个**查询生成管道**：
1. 模型作为SQL专家，根据输入问题生成SQL查询。
2. 查询生成遵循严格的指导方针，以确保正确性、相关性和安全性。
3. 使用`SubmitFinalAnswer`工具提交最终答案。

此系统确保：
- SQL查询**准确且优化**。
- 结果**有限且相关**。
- 错误和边缘情况**处理得当**。
- 工作流遵循**最佳实践**（例如，不使用DML语句）。

```python
# 定义查询生成系统提示
query_gen_system = """你是一个注重细节的SQL专家。

给定一个输入问题，输出一个语法正确的SQLite查询来运行，然后查看查询结果并返回答案。

除了使用SubmitFinalAnswer提交最终答案外，不要调用任何其他工具。

生成查询时：

输出回答输入问题的SQL查询，不使用工具调用。

除非用户指定希望获取的具体示例数量，否则始终将查询限制为最多5个结果。
你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
永远不要查询特定表的所有列，只询问与问题相关的列。

如果执行查询时出现错误，请重写查询并再次尝试。

如果得到空结果集，应尝试重写查询以获得非空结果集。
如果没有足够的信息回答查询，绝不要编造内容……只需说没有足够的信息。

如果有足够的信息回答输入问题，只需调用适当的工具向用户提交最终答案。

不要对数据库执行任何DML语句（INSERT、UPDATE、DELETE、DROP等）。"""
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen_chain = query_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key).bind_tools([SubmitFinalAnswer])
```

### 定义查询生成节点

此节点根据当前状态生成SQL查询并返回结果。

1. 验证输入状态以确保包含`ToolMessage`。
2. 使用`query_gen_chain`生成SQL查询。
3. 如果在查询生成期间调用了错误的工具，则处理错误。
4. 将生成的查询或错误消息作为工作流状态的一部分返回。

此节点在工作流中起着关键作用，确保SQL查询生成正确且错误处理得当。

```python
# 定义查询生成节点
def query_gen_node(state: State):
    """
    根据当前状态生成SQL查询并返回结果。
    """
    print("--- 查询生成节点 ---")
    # 确保最后一条消息是ToolMessage
    if isinstance(state["messages"][-1], ToolMessage):
        tool_call_id = state["messages"][-1].tool_call_id
        print(f"上一条消息的工具调用ID：{tool_call_id}")
    else:
        raise ValueError("预期最后一条消息是ToolMessage。")

    # 生成查询
    message = query_gen_chain.invoke(state)
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"错误：调用了错误的工具：{tc['name']}。请修正你的错误。记住只能调用SubmitFinalAnswer提交最终答案。生成的查询应不带工具调用输出。",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}
```

### 定义是否继续函数

此函数根据当前状态确定工作流中的下一步。

```python
# 定义是否继续函数
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    """
    根据当前状态确定工作流中的下一步。
    """
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"
```

### 定义数据库语句执行工具

此工具针对数据库执行SQL查询并返回结果。

```python
# 定义db_stmt_exec_tool函数
@tool
def db_stmt_exec_tool(query: str) -> str:
    """
    针对数据库执行SQL查询并返回结果。
    如果查询失败，返回错误消息。
    """
    result = db.run_no_throw(query)
    if not result:
        return "错误：查询失败。请重写你的查询并再次尝试。"
    return result
```

### 定义查询检查系统提示

此提示指导模型检查SQL查询中的常见错误。

1. 模型作为SQL专家，检查查询中的常见错误。
2. 如果发现错误，则重写查询；否则使用原始查询。
3. 将验证后的查询传递给`db_stmt_exec_tool`执行。

这确保只有**正确且安全的SQL查询**在数据库上执行，减少错误或意外行为的风险。

```python
# 定义查询检查系统提示
query_check_system = """你是一个注重细节的SQL专家。
仔细检查SQLite查询中的常见错误，包括：
- 对NULL值使用NOT IN
- 使用UNION时应使用UNION ALL
- 对独占范围使用BETWEEN
- 谓词中的数据类型不匹配
- 正确引用标识符
- 函数使用正确的参数数量
- 转换为正确的数据类型
- 使用正确的列进行联接

如果存在上述任何错误，请重写查询。如果没有错误，只需重现原始查询。

在运行此检查后，你将调用适当的工具来执行查询。"""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check_chain = query_check_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key).bind_tools([db_stmt_exec_tool], tool_choice="required")
```

### 定义纠正查询节点

此节点在必要时纠正SQL查询并返回纠正后的查询。

```python
# 定义纠正查询节点
def correct_query_node(state: State) -> dict[str, list[AIMessage]]:
    """
    在必要时纠正SQL查询并返回纠正后的查询。
    """
    print("--- 纠正查询节点 ---")
    return {"messages": [query_check_chain.invoke({"messages": [state["messages"][-1]]})]}
```

### 定义执行查询节点

此节点执行SQL查询并将结果作为ToolMessage返回。

```python
# 定义执行查询节点
def execute_query_node(state: State) -> dict[str, list[AIMessage]]:
    """
    执行SQL查询并将结果作为ToolMessage返回。
    """
    print("--- 执行查询节点 ---")
    try:
        query = state["messages"][-1].tool_calls[0]["args"]["query"]
        result = db_stmt_exec_tool.invoke(query)
        print(f"查询结果：\n{result}")
    except Exception as e:
        result = f"错误：{str(e)}"
        print(result)

    # 从上一条消息中获取tool_call_id
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    print(f"工具调用ID：{tool_call_id}")
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}
```

## 定义工作流

我们通过向状态图添加节点和边来定义工作流。此代码定义了一个用于生成和执行SQL查询的**有状态、逐步工作流**。工作流：
1. 从列出数据库表开始。
2. 检索特定表的模式。
3. 生成和纠正SQL查询。
4. 执行查询并处理错误或纠正。
5. 将整个工作流可视化为图表以便更好地理解。

工作流设计为模块化，每个节点处理特定任务，条件边确保根据流程状态的正确流动。

```python
# 定义工作流
workflow = StateGraph(State)

# 添加重新设计的节点名称
workflow.add_node("initial_tool_node", initial_tool_node)
workflow.add_node("list_tables_node", list_tables_node)
workflow.add_node("model_get_schema_node", model_get_schema_node)
workflow.add_node("retrieve_schema_node", retrieve_schema_node)
workflow.add_node("query_gen_node", query_gen_node)
workflow.add_node("correct_query_node", correct_query_node)
workflow.add_node("execute_query_node", execute_query_node)

# 添加更新后的节点名称的边
workflow.add_edge(START, "initial_tool_node")
workflow.add_edge("initial_tool_node", "list_tables_node")
workflow.add_edge("list_tables_node", "model_get_schema_node")
workflow.add_edge("model_get_schema_node", "retrieve_schema_node")
workflow.add_edge("retrieve_schema_node", "query_gen_node")
workflow.add_conditional_edges("query_gen_node", should_continue, [END, "correct_query_node", "query_gen_node"])
workflow.add_edge("correct_query_node", "execute_query_node")
workflow.add_edge("execute_query_node", "query_gen_node")

app = workflow.compile()

# 可视化图表
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
```

## 定义查询执行的辅助函数

我们定义辅助函数来提取、执行和格式化SQL查询及其结果。

### 关键工作流：

- 代码设计用于与从自然语言问题生成SQL查询的工作流配合使用。
- 它提取查询，在数据库上执行，并格式化结果以便于解释。
- 包括错误处理以确保查询执行和结果格式化期间的鲁棒性。

这组函数通常与更大的系统（如笔记本中的系统）结合使用，以根据用户输入自动生成和执行SQL查询。

```python
from openai import BadRequestError

def extract_sql_query(final_answer: str) -> str:
    """
    从final_answer字符串中提取SQL查询。
    
    参数：
        final_answer (str)：可能包含SQL查询的最终答案字符串。
    
    返回：
        str：如果找到，则返回提取的SQL查询，否则返回None。
    """
    if "```sql" in final_answer:
        return final_answer.split("```sql")[1].split("```")[0].strip()
    return None

def execute_sql_query(sql_query: str):
    """
    执行SQL查询并返回结果。
    
    参数：
        sql_query (str)：要执行的SQL查询。
    
    返回：
        Any：SQL查询执行的结果，如果发生错误则返回None。
    """
    try:
        results = db.run(sql_query)
        return results
    except Exception as e:
        print("执行SQL查询时出错：", e)
        return None

def format_results(results) -> str:
    """
    将查询结果格式化为人类可读的字符串。
    
    参数：
        results (Any)：SQL查询执行的结果。
    
    返回：
        str：表示查询结果的格式化字符串。
    """
    if isinstance(results, str):
        # 如果结果是字符串，则按原样返回
        return results

    formatted_results = "每个国家的总销售额为：\n"
    try:
        for row in results:
            # 处理行是元组或列表的情况
            if isinstance(row, (tuple, list)) and len(row) >= 2:
                formatted_results += f"- {row[0]}：${row[1]:.2f}\n"
            else:
                # 处理意外的行格式
                formatted_results += f"- {row}\n"
    except Exception as e:
        print("格式化结果时出错：", e)
        return str(results)  # 回退：将结果作为字符串返回

    return formatted_results

def process_event(event):
    """
    处理事件以提取、执行并打印最终答案。
    
    参数：
        event (dict)：包含工作流状态的事件。
    """
    if "query_gen_node" not in event:
        return

    query_gen_state = event["query_gen_node"]
    if "messages" not in query_gen_state:
        return

    last_message = query_gen_state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return

    final_answer = last_message.tool_calls[0]["args"]["final_answer"]
    sql_query = extract_sql_query(final_answer)

    if not sql_query:
        print("未在最终答案中找到SQL查询。代理提供了以下响应：")
        print(final_answer)
        return

    print(f"提取的SQL查询：\n{sql_query}\n")
    results = execute_sql_query(sql_query)

    if results:
        formatted_results = format_results(results)
        print("最终答案：", formatted_results)
```

## 根据用户问题执行查询

最后，我们针对不同用户问题执行工作流并打印结果。

```python
try:
    question = "每个国家的总销售额是多少？"
    for event in app.stream({"messages": [("user", question)]}):
        process_event(event)
except BadRequestError as e:
    print(f"处理问题时出错：{question}")
    print(f"错误详情：{e}")
```

```python
try:
    question = "每种类型的总销售额是多少？"
    for event in app.stream({"messages": [("user", question)]}):
        process_event(event)
except BadRequestError as e:
    print(f"处理问题时出错：{question}")
    print(f"错误详情：{e}")
```

```python
try:
    question = "每个播放列表中有多少首曲目？"
    for event in app.stream({"messages": [("user", question)]}):
        process_event(event)
except BadRequestError as e:
    print(f"处理问题时出错：{question}")
    print(f"错误详情：{e}")
```

```python
try:
    question = "数据库中拥有最多曲目的5位艺术家是谁？"
    for event in app.stream({"messages": [("user", question)]}):
        process_event(event)
except BadRequestError as e:
    print(f"处理问题时出错：{question}")
    print(f"错误详情：{e}")
```

```python
try:
    question = "按总消费额排名的前3名客户是谁？"
    for event in app.stream({"messages": [("user", question)]}):
        process_event(event)
except BadRequestError as e:
    print(f"处理问题时出错：{question}")
    print(f"错误详情：{e}")
```

## 结论

本笔记本展示了一个基于自然语言问题生成和执行SQL查询的强大工作流。通过利用LangChain、OpenAI和SQLDatabaseToolkit，我们可以创建一个处理复杂查询、纠正常见错误并格式化结果以便于解释的健壮系统。这种方法可以扩展到各种其他用例，使其成为数据分析和数据库管理的宝贵工具。

此工作流可应用于各种现实场景，例如：
- **商业智能**：自动从数据库生成报告和洞察。
- **数据探索**：使非技术用户能够使用自然语言查询数据库。
- **客户支持**：根据数据库数据提供客户查询的自动回答。
- **教育**：通过将自然语言问题翻译成查询来教授SQL概念。