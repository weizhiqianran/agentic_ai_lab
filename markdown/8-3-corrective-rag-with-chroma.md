# 实现纠正性RAG：提升语言模型中文档相关性

## 引言

在自然语言处理快速发展的领域中，**检索增强生成（RAG）**已成为一种强大的框架，通过整合外部知识源来提升语言模型的能力。传统的RAG系统通过检索相关文档来为生成响应提供信息并改进其质量，但它们往往缺乏评估检索信息质量和相关性的机制。这种局限性可能导致生成不准确或无关的答案，尤其是在检索到的文档无法充分回答用户查询时。

为了应对这一挑战，引入了**纠正性RAG（CRAG）**的概念。`CRAG`通过引入自我反思和自我评分机制来评估检索文档的相关性，从而增强了RAG框架。通过评估每篇文档与用户问题的相关性，`CRAG`确保只使用最相关的信息来生成响应。这种方法不仅提高了生成答案的准确性，还减少了对额外数据源的依赖。

在此示例中，我们使用LangGraph和LangChain实现了一个简化的`CRAG`策略。虽然完整的`CRAG`方法包括知识提炼和将文档分区为“知识条”等步骤，但此实现专注于评分文档相关性的核心思想，并在必要时通过网络搜索补充检索。通过利用DuckDuckGo进行网络搜索并结合查询重写以优化搜索，此工作流程展示了如何有效地应用`CRAG`原则来增强RAG系统。

## 安装所需包

以下代码块将必要的Python包静默安装（`-q`）并在已安装的情况下进行升级（`-U`）。这些包包括`LangChain`的各种组件、`LangGraph`、`ChromaDB`和`DuckDuckGo`搜索工具，这些是构建和运行检索增强生成（RAG）工作流程所必需的。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU chromadb
!pip install -qU duckduckgo_search
```

## 加载和准备文档

此代码块导入必要的模块并设置环境，以从指定URL加载、分裂和向量化文档。它从Kaggle密钥中安全检索OpenAI API密钥，使用OpenAI嵌入进行向量化，从提供的URL加载文档，将其分割为较小的块以便高效处理，并将其存储在Chroma向量数据库中以供后续检索。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from kaggle_secrets import UserSecretsClient

# 从Kaggle密钥中安全加载OpenAI API密钥
my_api_key = UserSecretsClient().get_secret("my-openai-api-key")

# 使用OpenAI嵌入进行向量化
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 要索引的博客文章URL
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 从URL加载文档
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 将文档分割为较小的块以便处理
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# 将文档添加到Chroma向量数据库
vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embed)
retriever = vectorstore.as_retriever()
```

## 评分文档相关性

此部分定义了一个数据模型，用于评分检索到的文档与用户问题的相关性。它初始化了一个具有函数调用能力的语言模型（LLM）来执行评分。构建了一个提示，指导LLM通过分配二元得分（“是”或“否”）来评估每篇文档的相关性。然后创建并测试了一个检索评分链，以演示其功能。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 评分文档相关性的数据模型
class GradeDocuments(BaseModel):
    """对检索文档相关性检查的二元得分。"""

    binary_score: str = Field(description="文档与问题相关，‘是’或‘否’")

# 用于评分文档的LLM，带有函数调用
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 评分文档相关性的提示
system = """你是评估检索文档与用户问题相关性的评分者。\n 
    如果文档包含与问题相关的关键字或语义意义，则将其评为相关。\n
    给出一个二元得分‘是’或‘否’，以指示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n {document} \n\n 用户问题：{question}"),
    ]
)

# 评分文档相关性的链
retrieval_grader = grade_prompt | structured_llm_grader

# 使用示例问题测试检索评分器
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content

result = retrieval_grader.invoke({"question": question, "document": doc_txt})
print(result)
```

## 设置RAG链以生成答案

此代码块设置了负责根据检索到的文档生成答案的检索增强生成（RAG）链。它从LangChain Hub中提取预定义的RAG提示，初始化一个用于生成响应的LLM，并构建一个处理上下文和问题以生成最终答案的链。然后通过RAG链运行一个示例问题，以展示其操作。

```python
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# 从Hub中提取RAG提示
prompt = hub.pull("rlm/rag-prompt")

# 用于生成答案的LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 使用RAG生成答案的链
rag_chain = prompt | llm | StrOutputParser()

# 使用示例问题运行RAG链
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)
```

```python
# 显示提示的样子
prompt = hub.pull("rlm/rag-prompt").pretty_print()
```

## 实现问题重写器

此部分创建了一个机制，以改进用户问题以获得更好的搜索结果。它定义了一个LLM，通过理解潜在的语义意图来重新表述输入问题，使其更适合网络搜索。编写了一个提示，指导LLM转换问题，并建立了一个链来处理和重新表述问题。

```python
# 问题重写器
# 用于重写问题的LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 重写问题的提示
system = """你是一个问题重写器，将输入问题转换为更适合网络搜索的优化版本。\n 
     查看输入并尝试推理其潜在的语义意图/含义。"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "这是初始问题：\n\n {question} \n 制定一个改进的问题。"),
    ]
)

# 重写问题的链
question_rewriter = re_write_prompt | llm | StrOutputParser()
result = question_rewriter.invoke({"question": question})
print(result)
```

## 初始化网络搜索工具

此代码块设置了来自LangChain社区工具的DuckDuckGo搜索工具。该工具用于在最初检索到的文档被认为不相关时执行网络搜索，确保系统可以从网络中获取额外信息以有效回答用户查询。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化DuckDuckGo搜索工具
web_search_tool = DuckDuckGoSearchRun()
```

## 定义图状态和工作流程节点

在此定义了图状态和工作流程节点，以管理RAG系统中的操作流程。`GraphState` TypedDict概述了必要的属性，如问题、生成的答案、网络搜索标志和检索到的文档。实现了工作流程中每个节点的功能——检索文档、生成答案、评分文档相关性、转换查询和执行网络搜索。此外，还实现了一个决策函数，根据检索文档的相关性确定下一步。

```python
from typing import List, Literal, Dict, Any
from typing_extensions import TypedDict
from langchain.schema import Document

class GraphState(TypedDict):
    """
    表示我们图的状态。

    属性：
        question: 当前问题。
        generation: LLM生成（答案）。
        web_search: 是否执行网络搜索。（“是”或“否”）
        documents: 检索到的文档列表。
    """
    question: str
    generation: str
    web_search: Literal["Yes", "No"]
    documents: List[Document]

def retrieve_node(state: GraphState) -> GraphState:
    """
    检索与问题相关的文档。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Dict[str, Any]: 更新后的状态，包含检索到的文档。
    """
    print("---检索---")
    question = state["question"]

    # 使用检索器检索文档
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate_node(state: GraphState) -> GraphState:
    """
    使用RAG链生成答案。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Dict[str, Any]: 更新后的状态，包含生成的答案。
    """
    print("---生成---")
    question = state["question"]
    documents = state["documents"]

    # 使用RAG链生成答案
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents_node(state: GraphState) -> GraphState:
    """
    评分检索到的文档与问题的相关性。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Dict[str, Any]: 更新后的状态，包含过滤后的相关文档。
    """
    print("---检查文档与问题的相关性---")
    question = state["question"]
    documents = state["documents"]

    # 为每篇文档评分相关性
    filtered_docs = []
    web_search: Literal["Yes", "No"] = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---评分：文档相关---")
            filtered_docs.append(d)
        else:
            print("---评分：文档不相关---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query_node(state: GraphState) -> GraphState:
    """
    转换查询以生成更适合网络搜索的版本。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Dict[str, Any]: 更新后的状态，包含重新表述的问题。
    """
    print("---转换查询---")
    question = state["question"]
    documents = state["documents"]

    # 重写问题以获得更好的搜索结果
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search_node(state: GraphState) -> GraphState:
    """
    根据重新表述的问题执行网络搜索。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Dict[str, Any]: 更新后的状态，附加了网络搜索结果。
    """
    print("---网络搜索---")
    question = state["question"]
    documents = state["documents"]

    # 使用DuckDuckGo执行网络搜索
    search_results = web_search_tool.run(question)
    web_results = Document(page_content=search_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

def decide_to_generate(state: GraphState) -> Literal["transform_query_node", "generate_node"]:
    """
    决定是生成答案还是重新生成问题。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Literal["transform_query_node", "generate_node"]: 下一个要调用的节点的决策。
    """
    print("---评估评分文档---")
    web_search = state["web_search"]

    if web_search == "Yes":
        # 如果文档不相关，重新生成问题
        print("---决策：所有文档与问题不相关，转换查询---")
        return "transform_query_node"
    else:
        # 如果文档相关，生成答案
        print("---决策：生成---")
        return "generate_node"
```

## 编译和可视化工作流程图

此代码块通过添加定义的节点并建立它们之间的连接（边）来构造工作流程图。它使用LangGraph的`StateGraph`来管理工作流程，指定起点和终点。此外，它尝试使用Mermaid语法可视化图表，提供工作流程的图形表示。可视化是可选的，可能需要额外的依赖项。

```python
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display

# 初始化工作流程图
workflow = StateGraph(GraphState)

# 定义工作流程中的节点
workflow.add_node("retrieve_node", retrieve_node)                # 检索文档
workflow.add_node("grade_documents_node", grade_documents_node)  # 评分文档
workflow.add_node("generate_node", generate_node)                # 生成答案
workflow.add_node("transform_query_node", transform_query_node)  # 转换查询
workflow.add_node("web_search_node", web_search_node)            # 网络搜索

# 构建图的边
workflow.add_edge(START, "retrieve_node")
workflow.add_edge("retrieve_node", "grade_documents_node")
workflow.add_conditional_edges("grade_documents_node", decide_to_generate, ["transform_query_node", "generate_node"])
workflow.add_edge("transform_query_node", "web_search_node")
workflow.add_edge("web_search_node", "generate_node")
workflow.add_edge("generate_node", END)

# 编译工作流程
app = workflow.compile()

# 可视化图表（可选，需要额外依赖项）
try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass
```

## 使用示例执行工作流程

此部分展示了如何通过提供输入问题来执行编译后的工作流程。它包括一个辅助函数，将不可序列化的对象转换为JSON可序列化的格式，以便更容易可视化输出。使用示例问题运行工作流程，并以结构化的JSON格式打印每个节点的输出。

```python
import json
import warnings

# 抑制LangSmith API密钥警告（如果不使用LangSmith）
warnings.filterwarnings("ignore", category=UserWarning, message="API key must be provided when using hosted LangSmith API")

# 将不可序列化对象转换为字典的辅助函数
def convert_to_serializable(obj):
    if hasattr(obj, "dict"):              # 检查对象是否有.dict()方法
        return obj.dict()
    elif isinstance(obj, (list, tuple)):  # 处理列表和元组
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):           # 处理字典
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:                                 # 如果对象已可序列化，则按原样返回
        return obj
```

```python
# 示例1：代理记忆的类型
inputs = {"question": "代理记忆的类型有哪些？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点'{key}'的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以JSON字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)
```

```python
# 示例2：提示工程技术
inputs = {"question": "有哪些提示工程技术？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点'{key}'的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以JSON字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)
```

```python
# 示例3：对大型语言模型的对抗性攻击
inputs = {"question": "对大型语言模型的对抗性攻击是什么？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点'{key}'的输出：")
        print("-"*80)
        # 将不可序列化对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以JSON字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)
```

## 结论

本文展示的实现提供了一种基础方法，将**纠正性RAG（CRAG）**原则整合到使用LangGraph和LangChain的检索增强生成框架中。通过引入评分机制来评估检索文档的相关性，系统确保只有相关信息用于生成响应，从而提高了输出的整体准确性和可靠性。

虽然此示例为简化起见省略了知识提炼阶段，但它为更复杂的增强奠定了基础，例如将文档分区为知识条和实现更深入的自我反思能力。此外，通过DuckDuckGo整合网络搜索补充功能展示了系统在初始检索不足时动态寻求额外信息的能力，确保对用户查询提供全面且准确的答案。

未来，进一步的改进可能包括整合先进的知识提炼技术、扩展数据源范围以及增强评分标准以捕捉更细微的相关性方面。这些增强将提升系统性能，使其成为需要精确且上下文适当的语言生成的广泛应用的强大工具。