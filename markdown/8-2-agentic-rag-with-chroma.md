# 使用 LangGraph 的 Agentic RAG

## 引言

Agentic RAG（检索增强生成）是一种结合检索方法和生成模型的强大方法，旨在提升响应的质量和相关性。在本例中，我们使用 LangGraph 实现了一个**检索代理**，它允许大语言模型（LLM）决定是否从索引数据集中检索信息。通过为 LLM 提供检索工具，我们使其能够动态决定何时检索额外上下文，何时直接生成响应。

本例展示了如何：
1. 将博客文章加载并索引到向量存储中。
2. 创建一个检索工具，用于搜索索引数据。
3. 使用 LangGraph 定义代理状态和工作流程。
4. 实现用于评估文档相关性、重写查询和生成响应的节点。
5. 可视化并执行基于图的工作流程。

## 第 0 步：安装所需库

安装项目所需的 Python 库，包括用于操作 OpenAI、Anthropic、LangChain、LangGraph 和 ChromaDB 的库。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU chromadb
```

## 第 1 步：设置和文档加载

从指定 URL 加载博客文章，将其分割成较小的块，并将其存储到向量数据库中以供检索。

## 第 2 步：检索工具创建

这一步涉及创建检索工具，使代理能够从索引的博客文章中搜索和检索相关信息。检索工具使用包含已处理和分割文档块的向量存储构建。该工具旨在帮助代理高效查询并获取存储博客文章中的特定信息。

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from kaggle_secrets import UserSecretsClient

# 获取 LLM API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("my-openai-api-key")

# 使用 OpenAI 嵌入进行向量化
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 定义加载博客文章的 URL
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 从指定 URL 加载文档
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]  # 展平文档列表

# 将文档分割成较小的块进行处理
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,   # 每个块的大小
    chunk_overlap=50  # 块之间的重叠以保持上下文
)
doc_splits = text_splitter.split_documents(docs_list)

# 将分割后的文档添加到向量存储以供检索
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedding,           
)
retriever = vectorstore.as_retriever()
```

这一步涉及创建检索工具，允许代理搜索并检索存储的博客文章信息。该工具被添加到代理可用的工具列表中。

```python
from langchain.tools.retriever import create_retriever_tool

# 创建检索工具以搜索和检索博客文章信息
retriever_tool = create_retriever_tool(retriever, "retrieve_blog_posts",
    "搜索并返回关于 Lilian Weng 博客文章中有关 LLM 代理、提示工程和 LLM 对抗性攻击的信息。",
)

# 代理可用的工具列表
tools = [retriever_tool]
```

## 第 3 步：代理状态定义

定义代理的状态，它由一系列消息组成。该状态在图的节点之间传递。

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 定义代理的状态，即一系列消息
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # 消息会被附加到状态中
```

## 第 4 步：节点和边定义

在这一步中，我们定义了构成基于图的工作流程的**节点**和**边**。每个节点代表一个特定任务或决策点，而边定义了这些节点之间的逻辑流。节点包括：

1. **`grade_documents`**：判断检索到的文档与用户问题是否相关。如果文档相关，工作流程将继续生成响应；如果不相关，则重写查询以提升其清晰度或相关性。
2. **`agent_node`**：核心决策节点。它决定是使用检索工具获取更多信息，还是在无需进一步检索时结束工作流程。
3. **`rewrite_node`**：重写用户问题以更好地匹配潜在的语义意图。当检索到的文档不相关时，此节点可优化查询以获得更好的结果。
4. **`generate_node`**：使用检索到的文档生成用户问题的最终响应。此节点仅在文档被认为相关时调用。

```python
from typing import Annotated, Literal, Sequence, Dict, Any
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition

def grade_documents(state: AgentState) -> Literal["generate_node", "rewrite_node"]:
    """
    评估检索到的文档与用户问题的相关性。
    如果文档相关，返回 "generate_node"，否则返回 "rewrite_node"。

    参数：
        state (AgentState)：包含消息的当前状态。

    返回：
        Literal["generate_node", "rewrite_node"]：基于文档相关性的决策。
    """
    print("---检查相关性---")

    # 定义用于评估相关性的 Pydantic 模型
    class Grade(BaseModel):
        binary_score: str = Field(description="相关性得分 'yes' 或 'no'")

    # 初始化用于评估的 LLM
    model = ChatOpenAI(model_name="gpt-4o-mini", api_key=my_api_key, temperature=0, streaming=True)
    llm_with_tool = model.with_structured_output(Grade)

    # 定义相关性评估的提示
    prompt = PromptTemplate(
        template="""你是评估检索到的文档与用户问题相关性的评分者。\n 
        这是检索到的文档：\n\n {context} \n\n
        这是用户的问题：{question} \n
        如果文档包含与用户问题相关的关键词或语义含义，则将其评为相关。\n
        给出 'yes' 或 'no' 的二元得分，表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    # 创建处理评分的链
    chain = prompt | llm_with_tool

    # 从状态中提取最后一条消息和问题
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    # 调用链来评分文档
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    # 根据得分返回决策
    if score == "yes":
        print("---决策：文档相关---")
        return "generate_node"
    else:
        print("---决策：文档不相关---")
        print(score)
        return "rewrite_node"

def agent_node(state: AgentState) -> Dict[str, Sequence[BaseMessage]]:
    """
    调用代理生成响应或决定检索信息。

    参数：
        state (AgentState)：包含消息的当前状态。

    返回：
        Dict[str, Sequence[BaseMessage]]：更新状态，代理的响应被附加到消息中。
    """
    print("---调用代理---")
    messages = state["messages"]
    model = ChatOpenAI(model_name="gpt-4o-mini", api_key=my_api_key, temperature=0, streaming=True)
    llm_with_tool = model.bind_tools(tools)    # 将可用工具绑定到模型
    response = llm_with_tool.invoke(messages)  # 生成响应
    return {"messages": [response]}

def rewrite_node(state: AgentState) -> Dict[str, Sequence[BaseMessage]]:
    """
    重写用户问题以提高清晰度或相关性。

    参数：
        state (AgentState)：包含消息的当前状态。

    返回：
        Dict[str, Sequence[BaseMessage]]：更新状态，重写后的问题被附加到消息中。
    """
    print("---转换查询---")
    messages = state["messages"]
    question = messages[0].content

    # 创建消息以请求重写问题
    msg = [
        HumanMessage(
            content=f""" \n 
    查看输入并尝试理解其潜在的语义意图或含义。\n 
    这是初始问题：
    \n -------------------------------------------------------- \n
    {question} 
    \n -------------------------------------------------------- \n
    制定一个改进后的问题：""",
        )
    ]

    # 调用 LLM 重写问题
    model = ChatOpenAI(model_name="gpt-4o-mini", api_key=my_api_key, temperature=0, streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate_node(state: AgentState) -> Dict[str, Sequence[BaseMessage]]:
    """
    使用检索到的文档生成用户问题的响应。

    参数：
        state (AgentState)：包含消息的当前状态。

    返回：
        Dict[str, Sequence[BaseMessage]]：更新状态，生成的响应被附加到消息中。
    """
    print("---生成---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content  # 检索到的文档

    # 从 hub 中拉取 RAG 提示
    prompt = hub.pull("rlm/rag-prompt")

    # 初始化用于生成响应的 LLM
    model = ChatOpenAI(model_name="gpt-4o-mini", api_key=my_api_key, temperature=0, streaming=True)

    # 创建 RAG 链以生成响应
    rag_chain = prompt | model | StrOutputParser()

    # 调用链生成响应
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
```

```python
# 显示提示的样子
prompt = hub.pull("rlm/rag-prompt").pretty_print()
```

## 第 5 步：图构建

通过定义节点（代理、检索、重写、生成）和边（条件逻辑）来构建图。该图决定了工作流程的流向。

```python
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# 用于检索文档的节点
retrieve_node = ToolNode([retriever_tool])

# 定义工作流程图
workflow = StateGraph(AgentState)

# 将节点添加到图中
workflow.add_node("agent_node", agent_node)        # 用于决策的代理节点
workflow.add_node("retrieve_node", retrieve_node)  # 用于检索文档的节点
workflow.add_node("rewrite_node", rewrite_node)    # 用于重写问题的节点
workflow.add_node("generate_node", generate_node)  # 用于生成最终响应的节点

# 定义节点之间的边
workflow.add_edge(START, "agent_node")             # 从代理节点开始

# 在条件边中使用 tools_condition，若最后一条消息有工具调用，则路由到 ToolNode。
# 否则，路由到结束。
workflow.add_conditional_edges(
    "agent_node",
    tools_condition,               # 决定是否检索或结束的条件
    {
        "tools": "retrieve_node",  # 如果需要工具，转到检索节点
        END: END,                  # 否则，结束工作流程
    },
)
workflow.add_conditional_edges("retrieve_node", grade_documents, ["generate_node", "rewrite_node"])
workflow.add_edge("generate_node", END)            # 生成响应后结束
workflow.add_edge("rewrite_node", "agent_node")    # 重写后返回代理节点

# 编译图
graph = workflow.compile()
```

## 第 6 步：可视化

可视化图以理解其结构。此步骤为可选步骤，需要额外的依赖项。

```python
from IPython.display import Image, display

# 可视化图（可选，需要额外依赖项）
try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass
```

## 第 7 步：执行

使用用户问题执行图，并打印每个节点的输出。代码抑制了与 LangSmith API 密钥相关的警告（如果未使用），并包含一个辅助函数，将不可序列化的对象转换为字典以进行 JSON 序列化。该示例展示了使用关于提示工程的问题查询图，并流式输出以供显示。

```python
# 使用输入执行图
import json
import warnings

# 抑制 LangSmith API 密钥警告（如果未使用 LangSmith）
warnings.filterwarnings("ignore", category=UserWarning, message="API key must be provided when using hosted LangSmith API")

# 将不可序列化对象转换为字典的辅助函数
def convert_to_serializable(obj):
    if hasattr(obj, "dict"):  # 检查对象是否有 .dict() 方法
        return obj.dict()
    elif isinstance(obj, (list, tuple)):  # 处理列表和元组
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):  # 处理字典
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:  # 如果对象已可序列化，则直接返回
        return obj

# 示例 1：关于提示工程的问题
inputs = {
    "messages": [
        ("user", "Lilian Weng 说了什么关于代理记忆类型的内容？"),
    ]
}

# 流式执行图并打印输出
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 将序列化值打印为 JSON 字符串
        print(json.dumps(serializable_value, indent=2))
    print("="*80)
```

```python
# 示例 2：关于 LLM 对抗性攻击的问题
inputs = {
    "messages": [
        ("user", "如何在大语言模型中缓解对抗性攻击？"),
    ]
}

# 流式执行图并打印输出
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 将序列化值打印为 JSON 字符串
        print(json.dumps(serializable_value, indent=2))
    print("="*80)
```

```python
# 示例 3：关于代理规划的问题
inputs = {
    "messages": [
        ("user", "Lilian Weng 讨论的代理规划的主要组成部分是什么？"),
    ]
}

# 流式执行图并打印输出
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 将序列化值打印为 JSON 字符串
        print(json.dumps(serializable_value, indent=2))
    print("="*80)
```

## 结论

使用 LangGraph 的 Agentic RAG 提供了一个灵活且模块化的框架，用于构建检索增强系统。通过将决策能力融入工作流程，系统可以动态适应用户查询，确保响应既准确又具有上下文相关性。这种方法特别适用于问答等应用，其中检索和处理外部信息的能力至关重要。LangGraph 的模块化设计使其易于扩展和定制工作流程以适应特定用例，使其成为构建高级 AI 系统的宝贵工具。