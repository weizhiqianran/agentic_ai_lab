# 使用 LangGraph 实现 Self RAG

## 引言

**Self-RAG** 是一种创新的 **检索增强生成（RAG）** 策略，它集成了 `自我反思` 和 `自我评分` 机制，以提升检索文档和生成响应的准确性和相关性。该方法利用大型语言模型（LLM）的能力，自主评估和改进信息检索与生成过程的质量，确保最终输出既可靠又与用户查询高度相关。

在 **Self-RAG** 框架中，系统性地做出几个关键决策，以优化检索与生成之间的交互：

1. **检索决策**：根据初始问题或当前生成输出，决定是否需要检索额外的文档片段。这一决策有助于管理信息范围，确保仅考虑相关数据。

2. **检索段落的相关性评估**：评估每个检索到的文档片段，确定其对用户问题的有用性。通过将文档分类为相关或无关，系统过滤掉噪音，聚焦于高质量信息源。

3. **验证 LLM 生成内容与检索片段的一致性**：评估 LLM 生成的语句是否完全受检索文档支持、部分支持或缺乏支持。这一步骤对于识别和减少幻觉（hallucination）至关重要，从而保持生成响应的 factual 准确性。

4. **生成响应的有用性评估**：衡量 LLM 生成的整体有用性，判断其是否有效解决了用户的问题。通过对响应评分，系统确保最终答案不仅准确，还能有效满足用户的意图。

通过这些自我调节步骤，**Self-RAG** 通过直接在检索和生成流程中嵌入质量控制，增强了传统 RAG 方法。这使得生成的响应更加可信且上下文适当，使 Self-RAG 成为需要高信息准确性和可靠性的应用的强大解决方案。

## 软件包安装

此代码块使用 `pip` 安装几个 Python 库，这些库通常用于构建和处理语言模型及 AI 应用：

1. **langchain-openai**：用于将 OpenAI 的语言模型与 LangChain 框架集成的库。
2. **langchain-anthropic**：用于将 Anthropic 的语言模型与 LangChain 集成的库。
3. **langchain_community**：社区驱动的库，为 LangChain 提供额外的工具和集成。
4. **langchain_experimental**：包含 LangChain 实验性功能和扩展的库。
5. **langgraph**：用于创建和管理语言模型交互图的库。
6. **chromadb**：用于存储和查询嵌入向量的矢量数据库，常用于 AI 应用。
7. **duckduckgo_search**：使用 DuckDuckGo 进行网络搜索的库。

这些库对于构建高级 AI 应用至关重要，特别是涉及自然语言处理、检索增强生成（RAG）和基于代理的系统。`-qU` 标志确保安装过程安静（非冗长）并在必要时升级现有安装。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU chromadb
!pip install -qU duckduckgo_search
```

## 导入库并设置向量存储

此代码块导入了工作流程所需的库和模块。它使用从 Kaggle 密钥中安全加载的 API 密钥设置 OpenAI 嵌入。然后定义要索引的 URL 列表，从这些 URL 加载文档，将其分割成较小的片段以便更好地检索，并将其添加到 Chroma 向量存储中以实现高效搜索。

```python
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import END, StateGraph, START

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from kaggle_secrets import UserSecretsClient

# 从 Kaggle 密钥安全加载 OpenAI API 密钥
my_api_key = UserSecretsClient().get_secret("my-openai-api-key")

# 使用 OpenAI 嵌入进行向量化
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 检索器
# 定义要索引的博客文章 URL
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 从 URL 加载文档
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 将文档分割成较小的片段以便更好地检索
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# 将文档添加到向量数据库（Chroma）
vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embed)
retriever = vectorstore.as_retriever()
```

## 检索评分器

此代码块使用 Pydantic 模型定义了一个检索评分器，以评估检索到的文档与用户问题的相关性。它初始化一个 OpenAI LLM，根据系统提示进行评分，该提示指示模型提供一个二进制分数来表示相关性。

```python
# 检索评分器
# 定义一个 Pydantic 模型来评分文档相关性
class GradeDocuments(BaseModel):
    """对检索文档相关性检查的二进制分数。"""

    binary_score: str = Field(description="文档与问题相关，‘yes’或‘no’")

# 初始化用于评分的 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 定义用于评分相关性的系统提示
system = """你是一个评分者，负责评估检索到的文档与用户问题的相关性。\n 
    不需要严格的测试。目标是过滤掉错误的检索结果。\n
    如果文档包含与用户问题相关的关键字或语义意义，则将其评为相关。\n
    给出一个二进制分数‘yes’或‘no’，以指示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n {document} \n\n 用户问题：{question}"),
    ]
)

# 创建用于评分文档相关性的链
retrieval_grader = grade_prompt | structured_llm_grader

# 使用示例问题测试检索评分器
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content

result = retrieval_grader.invoke({"question": question, "document": doc_txt})
print(result)
```

## 使用检索增强生成（RAG）链进行生成

此代码块设置了使用检索增强生成（RAG）链的生成阶段。它从 LangChain 的中心拉取一个 RAG 提示，初始化一个用于生成的 OpenAI LLM，并创建一个基于检索文档和用户问题生成答案的链。

```python
# 生成
# 从中心拉取一个 RAG 提示
prompt = hub.pull("rlm/rag-prompt")

# 初始化用于生成的 LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 创建用于生成答案的 RAG 链
rag_chain = prompt | llm | StrOutputParser()
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)
```

```python
# 显示提示的样子
prompt = hub.pull("rlm/rag-prompt").pretty_print()
```

## 幻觉评分器

此代码块定义了一个幻觉评分器，以评估生成的答案是否基于检索到的文档。它使用 Pydantic 模型进行二进制评分，并初始化一个 OpenAI LLM，系统提示指示模型判断生成内容是否受提供的文档支持。

```python
# 幻觉评分器
# 定义一个 Pydantic 模型来评分生成答案中的幻觉
class GradeHallucinations(BaseModel):
    """生成答案中是否存在幻觉的二进制分数。"""

    binary_score: str = Field(description="答案基于事实，‘yes’或‘no’")

# 初始化用于幻觉评分的 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 定义用于幻觉评分的系统提示
system = """你是一个评分者，负责评估 LLM 生成内容是否基于/受一组检索事实支持。\n 
     给出一个二进制分数‘yes’或‘no’。‘Yes’表示答案基于/受事实集支持。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集：\n\n {documents} \n\n LLM 生成内容：{generation}"),
    ]
)

# 创建用于幻觉评分的链
hallucination_grader = hallucination_prompt | structured_llm_grader
result = hallucination_grader.invoke({"documents": docs, "generation": generation})
print(result)
```

## 答案评分器

此代码块定义了一个答案评分器，以评估生成的答案是否充分回答了用户的问题。它利用 Pydantic 模型进行二进制评分，并设置一个 OpenAI LLM，系统提示指导模型判断答案是否解决了问题。

```python
# 答案评分器
# 定义一个 Pydantic 模型来评分答案是否解决了问题
class GradeAnswer(BaseModel):
    """评估答案是否解决问题的二进制分数。"""

    binary_score: str = Field(description="答案解决了问题，‘yes’或‘no’")

# 初始化用于答案评分的 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 定义用于答案评分的系统提示
system = """你是一个评分者，负责评估答案是否解决了问题。\n 
     给出一个二进制分数‘yes’或‘no’。‘Yes’表示答案解决了问题。"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题：\n\n {question} \n\n LLM 生成内容：{generation}"),
    ]
)

# 创建用于答案评分的链
answer_grader = answer_prompt | structured_llm_grader
result = answer_grader.invoke({"question": question, "generation": generation})
print(result)
```

## 问题重写器

此代码块设置了一个问题重写器，以改进用户的输入问题，从而优化从向量存储中的检索。它初始化一个 OpenAI LLM，系统提示指示模型重新表述问题，以更好地捕捉其语义意图。

```python
# 问题重写器
# 初始化用于问题重写的 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 定义用于问题重写的系统提示
system = """你是一个问题重写器，将输入问题转换为优化版本，\n 
     以便更好地从向量存储中检索。查看输入并尝试推理其底层的语义意图/含义。"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "这是初始问题：\n\n {question} \n 制定一个改进的问题。",
        ),
    ]
)

# 创建用于问题重写的链
question_rewriter = re_write_prompt | llm | StrOutputParser()
result = question_rewriter.invoke({"question": question})
print(result)
```

## 定义图状态和工作流节点

此代码块使用 LangGraph 定义了状态图工作流的状态和节点。它指定了图状态的结构，实现了每个节点的功能（检索、生成、评分文档、转换查询），并根据评分结果定义了节点之间的决策逻辑。

```python
from typing import List, Dict, Literal
from typing_extensions import TypedDict
from langchain.schema import Document
from IPython.display import Image, display

# 图状态
# 将图的状态定义为 TypedDict
class GraphState(TypedDict):
    """
    表示我们图的状态。

    属性：
        question: 用户的问题。
        generation: LLM 生成的答案。
        documents: 检索到的文档列表。
    """
    question: str
    generation: str
    documents: List[Document]

# 节点
def retrieve_node(state: GraphState) -> GraphState:
    """
    检索与用户问题相关的文档。

    参数：
        state (GraphState): 当前图状态。

    返回：
        GraphState: 更新了检索文档的状态。
    """
    print("---RETRIEVE---")
    question = state["question"]

    # 使用检索器检索文档
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate_node(state: GraphState) -> GraphState:
    """
    使用 RAG 链生成答案。

    参数：
        state (GraphState): 当前图状态。

    返回：
        GraphState: 更新了生成答案的状态。
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # 使用 RAG 链生成答案
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents_node(state: GraphState) -> GraphState:
    """
    评分检索到的文档与问题的相关性。

    参数：
        state (GraphState): 当前图状态。

    返回：
        GraphState: 更新了过滤后的相关文档的状态。
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # 根据相关性过滤文档
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query_node(state: GraphState) -> GraphState:
    """
    将用户的问题转换为更好的版本以便检索。

    参数：
        state (GraphState): 当前图状态。

    返回：
        GraphState: 更新了重新表述的问题的状态。
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # 重写问题以便更好地检索
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

# 边
def decide_to_generate(state: GraphState) -> Literal["transform_query_node", "generate_node"]:
    """
    决定是生成答案还是重新表述问题。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Literal["transform_query_node", "generate_node"]: 下一个节点的决策。
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 如果没有相关文档，重新表述问题
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query_node"
    else:
        # 如果存在相关文档，生成答案
        print("---DECISION: GENERATE---")
        return "generate_node"

def decide_generation_useful(state: GraphState) -> Literal["generate_node", "transform_query_node", END]:
    """
    决定生成的答案是否有用或需要重新生成。

    参数：
        state (GraphState): 当前图状态。

    返回：
        Literal["generate_node", "transform_query_node", END]: 下一个节点的决策。
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 检查生成内容是否基于文档
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # 检查生成内容是否解决了问题
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return END
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "transform_query_node"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate_node"
```

## 构建工作流图

此代码块使用之前定义的节点和决策函数构建工作流图。它通过添加节点并定义它们之间的边来建立工作流的流程。此外，如果必要的依赖可用，它会尝试可视化图表。

```python
# 构建图
workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("retrieve_node", retrieve_node)                # 检索文档
workflow.add_node("grade_documents_node", grade_documents_node)  # 评分文档相关性
workflow.add_node("generate_node", generate_node)                # 生成答案
workflow.add_node("transform_query_node", transform_query_node)  # 转换查询

# 构建图的边
workflow.add_edge(START, "retrieve_node")
workflow.add_edge("retrieve_node", "grade_documents_node")
workflow.add_conditional_edges("grade_documents_node", decide_to_generate, ["transform_query_node", "generate_node"])
workflow.add_edge("transform_query_node", "retrieve_node")
workflow.add_conditional_edges("generate_node", decide_generation_useful, ["generate_node", "transform_query_node", END])

# 编译工作流
app = workflow.compile()

# 可视化图（可选，需要额外的依赖）
try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass
```

## 序列化辅助函数

此代码块包含辅助函数，用于抑制特定警告并将不可序列化的对象转换为可序列化的格式（例如字典）。这确保工作流节点的输出可以轻松序列化并以 JSON 格式显示。

```python
import json
import warnings

# 抑制 LangSmith API 密钥警告（如果不使用 LangSmith）
warnings.filterwarnings("ignore", category=UserWarning, message="API key must be provided when using hosted LangSmith API")

# 将不可序列化的对象转换为字典的辅助函数
def convert_to_serializable(obj):
    if hasattr(obj, "dict"):              # 检查对象是否具有 .dict() 方法
        return obj.dict()
    elif isinstance(obj, (list, tuple)):  # 处理列表和元组
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):           # 处理字典
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:                                 # 如果对象已可序列化，则直接返回
        return obj
```

```python
# 示例 1：关于代理记忆的问题
inputs = {"question": "解释不同类型的代理记忆是如何工作的？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化的对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以 JSON 字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)

# 打印最终生成内容
print(f"最终生成内容：\n{value['generation']}\n")
```

```python
# 示例 2：关于对 LLM 的对抗性攻击的问题
inputs = {"question": "对抗性攻击如何影响大型语言模型？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化的对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以 JSON 字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)

# 打印最终生成内容
print(f"最终生成内容：\n{value['generation']}\n")
```

```python
# 示例 3：关于代理型代理的问题
inputs = {"question": "代理型代理的关键组成部分是什么？"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"来自节点 '{key}' 的输出：")
        print("-"*80)
        # 将不可序列化的对象转换为字典
        serializable_value = convert_to_serializable(value)
        # 以 JSON 字符串形式打印序列化值
        print(json.dumps(serializable_value, indent=2))
    print("="*80)

# 打印最终生成内容
print(f"最终生成内容：\n{value['generation']}\n")
```

## 结论

提供的代码设置了一个使用检索增强生成（RAG）方法处理用户问题的全面工作流。它涉及安装必要的软件包、加载和索引文档、评分检索文档的相关性、生成答案、检查幻觉，并确保最终答案解决了用户的问题。工作流被结构化为状态图，允许灵活的决策和对查询及生成过程的迭代改进。