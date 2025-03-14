# 使用 LangGraph 的自适应 RAG

## 引言

自适应 RAG（Retrieval-Augmented Generation，检索增强生成）是一种先进的策略，它结合了 **查询分析** 和 **主动/自我纠正 RAG**，能够根据用户查询的性质动态调整检索和生成过程。这种方法确保系统能够有效处理从简单事实性查询到复杂多步推理任务的各种问题。

在本次实现中，我们使用 **LangGraph** 构建工作流程，将查询分为两个主要路径：
1. **网络搜索**：适用于近期事件或索引文档未涵盖主题的问题。
2. **自我纠正 RAG**：适用于与索引文档相关的问题，系统检索相关信息，生成答案，并迭代优化以确保准确性和相关性。

通过利用 LangGraph 的图工作流程，我们创建了一个灵活且自适应的 RAG 管道，能够动态切换检索策略，评估生成答案的质量，并在必要时进行自我纠正。这种方法反映了近期研究中提出的原则，即通过查询分析将查询路由到不同检索策略，例如 **无检索**、**单次 RAG** 和 **迭代 RAG**。

## 步骤 0：安装命令

以下代码安装了构建和操作语言模型、代理及其相关工具常用的几个 Python 库。以下是各库的简要说明：

1. **`langchain-openai`**：将 OpenAI 的模型（如 GPT）与 LangChain 框架集成，用于构建语言模型应用。
2. **`langchain-anthropic`**：将 Anthropic 的模型（如 Claude）与 LangChain 集成。
3. **`langchain_community`**：提供 LangChain 的社区贡献工具、集成和实用程序。
4. **`langchain_experimental`**：包含 LangChain 的实验性功能和工具，仍在开发中。
5. **`langgraph`**：用于构建和可视化基于图的工作流程的库，常与 LangChain 配合使用。
6. **`tiktoken`**：OpenAI 模型的分词器，用于统计和管理文本中的标记（token）。
7. **`chromadb`**：用于存储和查询嵌入的向量数据库，常用于语义搜索和检索增强生成（RAG）管道。
8. **`duckduckgo_search`**：DuckDuckGo 搜索引擎的 Python 封装，用于从网络检索实时信息。

这些库对于构建高级语言模型应用（包括聊天机器人、代理和检索系统）至关重要。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU tiktoken
!pip install -qU chromadb
!pip install -qU duckduckgo_search
```

## 步骤 1：构建索引

此步骤通过加载基于网络的文档，将其分割成小块，并使用 OpenAI 嵌入将其存储在向量存储（Chroma）中来设置文档索引。索引用于在查询处理期间高效检索相关文档。

```python
# 第一阶段：构建索引
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("my-openai-api-key")

# 初始化 OpenAI 嵌入
embd = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 要索引的文档 URL
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 从 URL 加载文档
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 将文档分割为小块以高效处理
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 将文档存储在 Chroma 向量存储中
vectorstore = Chroma.from_documents(
    documents=doc_splits, collection_name="rag-chroma", embedding=embd
)
retriever = vectorstore.as_retriever()
```

## 步骤 2：路由器

路由器决定用户查询应使用向量存储（针对特定领域问题）还是网络搜索（针对一般问题）来回答。它使用结构化的语言模型（GPT-4o-mini）对查询进行分类。

```python
# 第二阶段：路由器
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 定义路由决策的 Pydantic 模型
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源。"""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题，选择将其路由到网络搜索还是向量存储。",
    )

# 初始化用于路由的语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_router = llm.with_structured_output(RouteQuery)

# 定义路由提示
system = """你是一个擅长将用户问题路由到向量存储或网络搜索的专家。
向量存储包含与代理、提示工程和对抗性攻击相关的文档。
对于这些主题的问题使用向量存储，否则使用网络搜索。"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 创建问题路由链
question_router = route_prompt | structured_llm_router

# 使用示例问题测试路由器
print(question_router.invoke({"question": "熊队会在 NFL 选秀中首先选择谁？"}))
print(question_router.invoke({"question": "代理的记忆类型有哪些？"}))
```

## 步骤 3：检索评分

此步骤评估检索到的文档与用户查询的相关性。它使用二元评分系统（是/否）过滤掉无关文档，确保仅使用上下文适当的内容来生成答案。

```python
# 第三阶段：检索评分
class GradeDocuments(BaseModel):
    """对检索文档相关性检查的二元评分。"""

    binary_score: str = Field(description="文档与问题相关，'是'或'否'")

# 初始化用于评分的语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 定义评分提示
system = """你是一个评估检索文档与用户问题相关性的评分者。
如果文档包含与用户问题相关的关键词或语义含义，则将其评为相关。
不需要严格测试，目标是过滤掉错误的检索结果。
给出二元评分 '是' 或 '否'，表示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n {document} \n\n 用户问题：{question}"),
    ]
)

# 创建检索评分链
retrieval_grader = grade_prompt | structured_llm_grader

# 测试检索评分器
question = "代理记忆"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content

result = retrieval_grader.invoke({"question": question, "document": doc_txt})
print(result)
```

## 步骤 4：生成

生成阶段构建 RAG 链，基于检索到的文档和用户查询生成答案。它使用预定义提示和 GPT-4o-mini 生成连贯且上下文准确的回答。

```python
# 第四阶段：生成
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# 从 hub 中提取 RAG 提示
prompt = hub.pull("rlm/rag-prompt")

# 初始化用于生成的语言模型
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 创建 RAG 链
rag_chain = prompt | llm | StrOutputParser()

# 使用 RAG 链生成答案
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)
```

```python
# 显示提示的样子
prompt = hub.pull("rlm/rag-prompt").pretty_print()
```

## 步骤 5：幻觉评分

此步骤检查生成的答案是否基于检索到的文档，确保答案有事实依据而非幻觉，使用二元评分系统。

```python
# 第五阶段：幻觉评分
class GradeHallucinations(BaseModel):
    """对生成答案中是否存在幻觉的二元评分。"""

    binary_score: str = Field(description="答案基于事实，'是'或'否'")

# 初始化用于幻觉评分的语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 定义幻觉评分提示
system = """你是一个评估语言模型生成内容是否基于检索事实的评分者。
给出二元评分 '是' 或 '否'。'是' 表示答案基于/受事实支持。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集合：\n\n {documents} \n\n 语言模型生成：{generation}"),
    ]
)

# 创建幻觉评分链
hallucination_grader = hallucination_prompt | structured_llm_grader

# 测试幻觉评分器
result = hallucination_grader.invoke({"documents": docs, "generation": generation})
print(result)
```

## 步骤 6：答案评分

答案评分器评估生成的答案是否完全回答了用户查询，确保回答相关且有效解决问题。

```python
# 第六阶段：答案评分
class GradeAnswer(BaseModel):
    """评估答案是否解决问题的二元评分。"""

    binary_score: str = Field(description="答案解决了问题，'是'或'否'")

# 初始化用于答案评分的语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 定义答案评分提示
system = """你是一个评估答案是否解决问题的评分者。
给出二元评分 '是' 或 '否'。'是' 表示答案解决了问题。"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题：\n\n {question} \n\n 语言模型生成：{generation}"),
    ]
)

# 创建答案评分链
answer_grader = answer_prompt | structured_llm_grader

# 测试答案评分器
result = answer_grader.invoke({"question": question, "generation": generation})
print(result)
```

## 步骤 7：问题重写

此步骤重写用户查询以优化向量存储检索。它改进查询的语义理解，提高检索文档的相关性。

```python
# 第七阶段：问题重写
# 初始化用于问题重写的语言模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 定义问题重写提示
system = """你是一个问题重写器，将输入问题转换为更适合向量存储检索的优化版本。
查看输入并推理其底层语义意图/含义。"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "这里是初始问题：\n\n {question} \n 制定一个改进的问题。",
        ),
    ]
)

# 创建问题重写链
question_rewriter = re_write_prompt | llm | StrOutputParser()

# 测试问题重写器
result = question_rewriter.invoke({"question": question})
print(result)
```

## 步骤 8：搜索

搜索阶段集成了 DuckDuckGo 搜索，用于基于网络的查询。当路由器确定问题更适合使用外部来源回答时，它会检索网络结果。

```python
# 第八阶段：搜索
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化 DuckDuckGo 搜索工具
web_search_tool = DuckDuckGoSearchRun()
```

## 步骤 9：定义图状态和流程

此步骤定义了检索增强生成（RAG）管道的 **状态** 和 **流程**，将其作为 **基于图的工作流程**。图由相互连接的 **节点** 组成，每个节点代表管道中的特定任务，例如文档检索、问题路由、答案生成和评估。图的 **状态** 在整个工作流程中得以维持，确保每个节点都能访问必要信息（例如用户的问题、检索到的文档、生成的答案）。

```python
# 第九阶段：定义图状态和流程
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

class GraphState(TypedDict):
    """
    表示图的状态。

    属性：
        question：问题
        generation：语言模型生成
        documents：文档列表
    """

    question: str
    generation: str
    documents: List[str]

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    根据用户问题检索文档。

    参数：
        state (GraphState)：当前图状态。

    返回：
        Dict[str, Any]：更新后的状态，包含检索到的文档。
    """
    print("---检索---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    使用 RAG 链生成答案。

    参数：
        state (GraphState)：当前图状态。

    返回：
        Dict[str, Any]：更新后的状态，包含生成的答案。
    """
    print("---生成---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    评估检索到的文档与问题的相关性。

    参数：
        state (GraphState)：当前图状态。

    返回：
        Dict[str, Any]：更新后的状态，包含筛选后的相关文档。
    """
    print("---检查文档与问题的相关性---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---评分：文档相关---")
            filtered_docs.append(d)
        else:
            print("---评分：文档不相关---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query_node(state: GraphState) -> Dict[str, Any]:
    """
    重写用户问题以优化检索。

    参数：
        state (GraphState)：当前图状态。

    返回：
        Dict[str, Any]：更新后的状态，包含重新表述的问题。
    """
    print("---转换查询---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    使用 DuckDuckGo 执行网络搜索。

    参数：
        state (GraphState)：当前图状态。

    返回：
        Dict[str, Any]：更新后的状态，包含网络搜索结果。
    """
    print("---网络搜索---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    
    # 处理 docs 是字符串列表或单个字符串的情况
    if isinstance(docs, str):
        web_results = docs  # 如果 docs 是单个字符串，直接使用
    elif isinstance(docs, list):
        web_results = "\n".join(docs)  # 如果 docs 是字符串列表，连接它们
    else:
        raise ValueError(f"网络搜索结果的类型意外：{type(docs)}")
    
    web_results = Document(page_content=web_results)
    return {"documents": [web_results], "question": question}

def route_search_store(state: GraphState) -> Literal["web_search_node", "retrieve_node"]:
    """
    将问题路由到网络搜索或 RAG。

    参数：
        state (GraphState)：当前图状态。

    返回：
        str：下一个调用的节点（"web_search_node"，"retrieve_node"）。
    """
    print("---路由问题---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---将问题路由到网络搜索---")
        return "web_search_node"
    elif source.datasource == "vectorstore":
        print("---将问题路由到 RAG---")
        return "retrieve_node"

def decide_to_generate(state: GraphState) -> Literal["transform_query_node", "generate_node"]:
    """
    决定是生成答案还是重新生成问题。

    参数：
        state (GraphState)：当前图状态。

    返回：
        str：下一个调用的节点（"transform_query_node"，"generate_node"）。
    """
    print("---评估筛选后的文档---")
    filtered_documents = state["documents"]
    if not filtered_documents:
        print("---决定：所有文档与问题无关，转换查询---")
        return "transform_query_node"
    else:
        print("---决定：生成---")
        return "generate_node"

def route_generate_transform(state: GraphState) -> Literal["generate_node", "transform_query_node", END]:
    """
    根据文档和问题评估生成内容。

    参数：
        state (GraphState)：当前图状态。

    返回：
        str：下一个节点的决定（"generate_node"，"transform_query_node"，END）。
    """
    print("---检查幻觉---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    if grade == "yes":
        print("---决定：生成内容基于文档---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---决定：生成内容解决了问题---")
            return END
        else:
            print("---决定：生成内容未解决问题---")
            return "transform_query_node"
    else:
        print("---决定：生成内容未基于文档，重新尝试---")
        return "generate_node"
```

## 步骤 10：编译和使用图

图被编译成一个可执行的工作流程。它协调节点之间的交互，确保管道高效处理查询并产生准确的结果。

1. **图状态（`GraphState`）：**
   - 一个 `TypedDict`，表示工作流程中任意点的图状态。
   - 包含：
     - **`question`**：用户输入的问题。
     - **`generation`**：语言模型生成的答案。
     - **`documents`**：与问题相关的检索文档列表。

2. **节点：**
   - 每个节点是一个执行特定任务并更新图状态的函数。
   - 关键节点包括：
     - **`retrieve_node`**：根据用户问题从向量存储中检索文档。
     - **`generate_node`**：使用 RAG 链生成答案。
     - **`grade_documents_node`**：评估检索文档与问题的相关性。
     - **`transform_query_node`**：重写用户问题以优化检索。
     - **`web_search_node`**：使用 DuckDuckGo 执行网络搜索以获取外部信息。
     - **`route_search_store`**：将问题路由到网络搜索或向量存储。
     - **`decide_to_generate`**：决定是生成答案还是重新表述问题。
     - **`route_generate_transform`**：评估生成的答案并决定下一步（例如结束工作流程、重新表述问题或重新生成答案）。

3. **条件边：**
   - 图使用 **条件边** 根据每个节点的结果动态路由工作流程。
   - 例如：
     - 如果检索到的文档不相关，工作流程路由到 `transform_query_node` 以重新表述问题。
     - 如果生成的答案未基于文档，工作流程路由回 `generate_node` 以重新生成答案。

4. **流程逻辑：**
   - 工作流程从用户问题开始，根据相关性将其路由到向量存储或网络搜索。
   - 检索到的文档被评分以确定相关性，无关文档被过滤掉。
   - RAG 链生成答案，然后评估其是否存在幻觉以及是否与问题相关。
   - 如果答案令人满意，工作流程结束。否则，它会循环回退以重新表述问题或重新生成答案。

```python
# 第十阶段：编译和使用图
# 初始化工作流程
workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("transform_query_node", transform_query_node)

# 构建图
workflow.add_conditional_edges(START, route_search_store, ["web_search_node", "retrieve_node"])
workflow.add_edge("web_search_node", "generate_node")
workflow.add_edge("retrieve_node", "grade_documents_node")
workflow.add_conditional_edges("grade_documents_node", decide_to_generate, ["transform_query_node", "generate_node"])
workflow.add_edge("transform_query_node", "retrieve_node")
workflow.add_conditional_edges("generate_node", route_generate_transform, ["generate_node", "transform_query_node", END])

# 编译工作流程
app = workflow.compile()
```

## 步骤 11：可视化

此可选阶段使用 Mermaid 和 IPython 可视化图结构。它提供了管道流程和决策点的图形表示。

```python
# 第十一阶段：可视化
from IPython.display import Image, display

# 可视化图（可选，需要额外依赖）
try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass
```

## 步骤 12：执行图

最后一步使用用户查询执行图。它通过管道处理查询，以 JSON 格式打印中间输出，并显示最终生成的答案。不可序列化的对象被转换为字典以确保输出整洁。

```python
# 第十二阶段：使用输入执行图
import json
import warnings

# 抑制 LangSmith API 密钥警告（如果未使用 LangSmith）
warnings.filterwarnings("ignore", category=UserWarning, message="API key must be provided when using hosted LangSmith API")

# 将不可序列化对象转换为字典的辅助函数
def convert_to_serializable(obj):
    if hasattr(obj, "dict"):  # 检查对象是否具有 .dict() 方法
        return obj.dict()
    elif isinstance(obj, (list, tuple)):  # 处理列表和元组
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):  # 处理字典
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:  # 如果对象已是可序列化的，直接返回
        return obj

# 使用输入问题运行图
inputs = {"question": "代理的记忆类型有哪些？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以 JSON 字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)

# 打印最终生成的答案
print(f"最终生成：\n{value['generation']}\n")
```

```python
# 使用与代理相关的问题运行图
inputs = {"question": "AI 代理的关键组件是什么？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以 JSON 字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)

# 打印最终生成的答案
print(f"最终生成：\n{value['generation']}\n")
```

```python
# 使用与代理无关的问题运行图
inputs = {"question": "法国的首都是什么？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以 JSON 字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)

# 打印最终生成的答案
print(f"最终生成：\n{value['generation']}\n")
```

## 结论

使用 LangGraph 构建的自适应 RAG 管道展示了结合 **查询分析** 和 **自我纠正机制** 的强大功能，创建了一个健壮且灵活的问答系统。通过动态地在 **网络搜索** 和 **自我纠正 RAG** 之间路由查询，系统确保用户无论问题的复杂性或时效性如何，都能获得准确且上下文相关的答案。

此实现突出了现代 RAG 系统中模块化和适应性的重要性。使用 LangGraph 的基于图的工作流程允许无缝集成多种检索策略，实时评估生成答案，并通过迭代优化提高响应质量。随着 RAG 系统的不断发展，像自适应 RAG 这样的策略将在提升其处理多样化和挑战性查询的能力方面发挥关键作用。

总之，使用 LangGraph 的自适应 RAG 代表了构建智能、自我纠正问答系统的重要进步，这些系统能够适应用户需求和查询性质的变化。