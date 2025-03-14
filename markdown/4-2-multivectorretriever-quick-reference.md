# **LangChain `MultiVectorRetriever` 快速参考**

## **简介**

`MultiVectorRetriever` 类允许基于多组嵌入向量检索文档。这对于以下应用场景非常有益：

- **分块（Chunking）**：将文档分割成较小的片段并为每个片段生成嵌入向量，有助于在保持上下文的同时捕捉语义含义。
- **摘要（Summarization）**：创建文档摘要并为其生成嵌入向量，与原始内容一起存储，便于更快地搜索和检索。
- **假设性问题（Hypothetical Questions）**：为文档生成相关的假设性问题并嵌入，提升检索器匹配查询与合适文档的能力。

### 核心功能

- **并行调用**：`MultiVectorRetriever` 可以通过异步方法并行运行检索过程，提升处理大数据集时的性能。
- **自定义搜索类型**：用户可以指定不同的搜索类型，例如相似性搜索或最大边际相关性（MMR），以根据需求定制检索过程。
- **灵活的存储选项**：检索器支持多种存储后端，用于存储父文档及其嵌入向量，提供更大的实现灵活性。

### 使用场景

1. **增强文档检索**：通过使用多个向量，检索器可以基于文档的不同表示形式返回更相关的结果。
2. **高效信息检索**：适用于需要快速访问大量信息的应用，例如需要深入理解用户查询的聊天机器人或搜索引擎。

---

## **准备工作**

### 安装所需的库
本节安装使用 LangChain、OpenAI 嵌入模型和 Chroma 向量存储所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和工具。
- `langchain-chroma`：支持与 Chroma 向量数据库的集成。
- `chromadb`：Chroma 向量数据库的核心库。

```python
!pip install -qU langchain-openai
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langchain-chroma>=0.1.2
!pip install -qU chromadb
```

### 初始化 OpenAI 嵌入模型
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全地获取 OpenAI API 密钥并初始化 OpenAI 嵌入模型。`OpenAIEmbeddings` 类用于创建嵌入模型实例，将文本转换为数值嵌入向量。

主要步骤：
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全地检索 OpenAI API 密钥。
2. **初始化嵌入模型**：使用 `text-embedding-3-small` 模型和获取的 API 密钥初始化 `OpenAIEmbeddings` 类。

此设置确保嵌入模型可用于后续任务，例如缓存嵌入或创建向量存储。

```python
from langchain_openai import OpenAIEmbeddings
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入模型
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
```

---

## **1. 文档检索**

### **基本文档检索**
此示例展示如何使用查询检索相关文档。它初始化一个 `Chroma` 向量存储和一个 `InMemoryByteStore` 用于存储父文档。文档被添加到向量存储和字节存储中，并使用查询检索相关文档。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain_core.load.dump import dumps  # 用于序列化文档

# 初始化向量存储、字节存储和嵌入模型
vectorstore = Chroma(embedding_function=embed)
byte_store = InMemoryByteStore()  # MultiVectorRetriever 必需
retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=byte_store)

# 向向量存储和字节存储添加文档
documents = [
    Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。", metadata={"doc_id": "1"})
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 将父文档添加到字节存储（正确序列化）
for doc in documents:
    serialized_doc = dumps(doc)  # 序列化 Document 对象
    byte_store.mset([(doc.metadata["doc_id"], serialized_doc.encode("utf-8"))])

# 检索相关文档
query = "LangChain 是什么？"
results = retriever.invoke(query)
print(results)
```

### **带元数据过滤的检索**
此示例展示如何在检索文档时根据元数据进行过滤。带有特定元数据（例如 `language: Python`）的文档被添加到向量存储和字节存储中。检索器随后用于获取匹配元数据过滤条件的文档。

```python
# 添加带元数据的文档
documents = [
    Document(page_content="LangChain 支持 Python。", metadata={"doc_id": "2", "language": "Python"}),
    Document(page_content="LangChain 还支持 JavaScript。", metadata={"doc_id": "3", "language": "JavaScript"}),
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 将父文档添加到字节存储（正确序列化）
for doc in documents:
    serialized_doc = dumps(doc)  # 序列化 Document 对象
    byte_store.mset([(doc.metadata["doc_id"], serialized_doc.encode("utf-8"))])

# 使用元数据过滤检索文档
query = "LangChain 支持哪些语言？"
results = retriever.invoke(query, search_kwargs={"filter": {"language": "Python"}})
print(results)
```

---

## **2. 批处理**

### **批次检索**
此示例展示如何为多个查询进行批次检索文档。文档被添加到向量存储和字节存储中，一组查询在单个批次中处理，所有查询的结果一起返回。

```python
# 将文档添加到向量存储
documents = [
    Document(page_content="LangChain 是一个用于 LLM 应用的框架。", metadata={"doc_id": "4"}),
    Document(page_content="OpenAI 提供强大的语言模型。", metadata={"doc_id": "5"}),
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 将父文档添加到字节存储（正确序列化）
for doc in documents:
    serialized_doc = dumps(doc)  # 序列化 Document 对象
    byte_store.mset([(doc.metadata["doc_id"], serialized_doc.encode("utf-8"))])

# 批次检索
queries = ["LangChain 是什么？", "OpenAI 提供什么？"]
results = retriever.batch(queries)
print(results)
```

### **带自定义配置的批次检索**
此示例展示如何为批次检索使用自定义配置。`max_concurrency` 参数用于控制并行检索操作的数量。这在处理大量查询时有助于优化性能。

```python
# 带自定义配置的批次检索
queries = ["LangChain 是什么？", "OpenAI 提供什么？"]
results = retriever.batch(queries, config={"max_concurrency": 2})
print(results)
```

---

## **3. 流式处理**

### **流式检索结果**
此示例展示如何实时流式传输检索结果。文档被添加到向量存储和字节存储中，检索器在检索结果时将其流式输出。这对于处理大数据集或实时应用非常有用。

```python
# 将文档添加到向量存储
documents = [
    Document(page_content="LangChain 是一个用于 LLM 应用的框架。", metadata={"doc_id": "6"}),
    Document(page_content="OpenAI 提供强大的语言模型。", metadata={"doc_id": "7"}),
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 将父文档添加到字节存储（正确序列化）
for doc in documents:
    serialized_doc = dumps(doc)  # 序列化 Document 对象
    byte_store.mset([(doc.metadata["doc_id"], serialized_doc.encode("utf-8"))])

# 流式检索结果
query = "LangChain 是什么？"
for result in retriever.stream(query):
    print(result)
```

### **带元数据的流式处理**
此示例展示如何在流式输出结果时包含元数据。`include_metadata` 参数用于确保流式输出中包含元数据。这在需要为每个检索文档提供额外上下文时很有用。

```python
# 带元数据的流式检索结果
query = "OpenAI 提供什么？"
for result in retriever.stream(query, search_kwargs={"include_metadata": True}):
    print(result)
```

---

## **4. 配置和自定义**

### **绑定参数到检索器**
此示例展示如何为检索器绑定额外参数。`search_kwargs` 参数用于自定义检索过程，例如限制结果数量（`k`）。这允许灵活配置检索器。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain_core.load.dump import dumps  # 用于序列化文档

# 初始化向量存储、字节存储
vectorstore = Chroma(embedding_function=embed)
byte_store = InMemoryByteStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=byte_store)

# 向向量存储和字节存储添加文档
documents = [
    Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。", metadata={"doc_id": "1"})
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 将父文档添加到字节存储（正确序列化）
for doc in documents:
    serialized_doc = dumps(doc)  # 序列化 Document 对象
    byte_store.mset([(doc.metadata["doc_id"], serialized_doc.encode("utf-8"))])

# 绑定额外参数
custom_retriever = retriever.bind(search_kwargs={"k": 3})  # 检索前 3 个结果

# 检索相关文档
query = "LangChain 是什么？"
results = custom_retriever.invoke(query)
print(results)
```

### **可配置的替代方案**
此示例展示如何在运行时配置替代检索器。使用 `ConfigurableField` 定义默认检索器和替代检索器。可以通过配置参数在运行时切换检索器。这对于测试不同的检索策略非常有用。

```python
from langchain_core.runnables.utils import ConfigurableField

# 创建可配置的检索器
configurable_retriever = retriever.configurable_alternatives(
    ConfigurableField(id="retriever"),
    default_key="default",
    alternative_retriever=MultiVectorRetriever(vectorstore=Chroma(embedding_function=embed), byte_store=InMemoryByteStore())
)

# 使用默认检索器
print("使用默认检索器：")
results = configurable_retriever.invoke("LangChain 是什么？")
print(results)

# 使用替代检索器
print("\n使用替代检索器：")
results = configurable_retriever.with_config(configurable={"retriever": "alternative_retriever"}).invoke("LangChain 是什么？")
print(results)
```

---

## **5. 事件处理和错误处理**

### **添加生命周期监听器**
此示例展示如何为检索器添加同步生命周期监听器。`on_start` 和 `on_end` 监听器用于跟踪检索操作的开始和结束。这对于日志记录或监控检索过程非常有用。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain_core.load.dump import dumps

# 初始化向量存储、字节存储和嵌入模型
vectorstore = Chroma(embedding_function=embed)
byte_store = InMemoryByteStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=byte_store)

# 向向量存储和字节存储添加文档
documents = [
    Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。", metadata={"doc_id": "1"})
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 将父文档添加到字节存储（正确序列化）
for doc in documents:
    serialized_doc = dumps(doc)  # 序列化 Document 对象
    byte_store.mset([(doc.metadata["doc_id"], serialized_doc.encode("utf-8"))])

# 定义生命周期监听器
def on_start(run):
    print(f"检索开始，输入为：{run.input}")

def on_end(run):
    print(f"检索结束，输出为：{run.output}")

# 为检索器添加监听器
listener_retriever = retriever.with_listeners(on_start=on_start, on_end=on_end)

# 调用带监听器的检索器
results = listener_retriever.invoke("LangChain 是什么？")
print(results)
```

### **失败时重试**
此示例展示如何添加重试逻辑处理失败。`with_retry` 方法用于指定重试次数和处理的异常类型。这确保临时故障不会中断检索过程。

```python
# 添加重试逻辑
retry_retriever = retriever.with_retry(stop_after_attempt=3, retry_if_exception_type=(Exception,))

# 调用带重试逻辑的检索器
results = retry_retriever.invoke("LangChain 是什么？")
print(results)
```

### **备用检索器**
此示例展示如何在失败时添加备用检索器。定义并将备用检索器添加到主检索器。如果主检索器失败，则使用备用检索器作为备份。这提供了冗余并提高了可靠性。

```python
from langchain.retrievers import MultiVectorRetriever

# 创建备用检索器
fallback_retriever = MultiVectorRetriever(vectorstore=Chroma(embedding_function=embed), byte_store=InMemoryByteStore())

# 为检索器添加备用
fallback_enabled_retriever = retriever.with_fallbacks([fallback_retriever])
results = fallback_enabled_retriever.invoke("LangChain 是什么？")
print(results)
```

---

## **6. 最佳实践**

### **示例 1：将摘要与文档关联以进行检索**

1. **摘要链**：
   - 使用 LLM（如 `ChatOpenAI`）创建链以总结文档。
   - 该链接受文档内容，生成摘要，并将其作为字符串输出。
2. **批次摘要**：
   - 对一组文档（`docs`）应用摘要链，设置并发限制为 5。
3. **向量存储和文档存储**：
   - 初始化 `Chroma` 向量存储以存储摘要。
   - 使用 `InMemoryByteStore` 存储原始文档。
4. **检索器初始化**：
   - 初始化 `MultiVectorRetriever`，将摘要（存储在向量存储中）与原始文档（存储在文档存储中）关联。
5. **查询**：
   - 使用搜索词（`"justice breyer"`）查询检索器，返回相关的父文档。

```python
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import uuid

# 初始化 OpenAI 嵌入模型和 LLM
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
model = ChatOpenAI(model="gpt-4o-mini", api_key=my_api_key)

# 定义要总结和检索的文档列表
docs = [
    Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。", metadata={"title": "LangChain 概述"}),
    Document(page_content="OpenAI 提供强大的语言模型，如 GPT-4。", metadata={"title": "OpenAI 模型"}),
    Document(page_content="Chroma 是一个基于嵌入的检索向量存储。", metadata={"title": "Chroma 向量存储"}),
]

# 定义摘要链
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("总结以下文档：\n\n{doc}")
    | model
    | StrOutputParser()
)

# 为文档生成摘要
summaries = chain.batch(docs, {"max_concurrency": 5})

# 初始化向量存储和文档存储
vectorstore = Chroma(collection_name="summaries", embedding_function=embed)
store = InMemoryByteStore()
id_key = "doc_id"

# 初始化检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# 为文档生成唯一 ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 创建摘要文档
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# 将摘要添加到向量存储，原始文档添加到文档存储
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 查询检索器
retrieved_docs = retriever.invoke("LangChain")
print(retrieved_docs)
```

### **示例 2：假设性问题以改进检索**

1. **假设性问题链**：
   - 使用 LLM（如 `ChatOpenAI`）创建链为文档生成假设性问题。
   - 该链使用结构化输出（`HypotheticalQuestions`）确保输出为问题列表。
2. **批次问题生成**：
   - 对一组文档（`docs`）应用该链，设置并发限制为 5。
3. **向量存储和文档存储**：
   - 初始化 `Chroma` 向量存储以存储假设性问题。
   - 使用 `InMemoryByteStore` 存储原始文档。
4. **检索器初始化**：
   - 初始化 `MultiVectorRetriever`，将假设性问题（存储在向量存储中）与原始文档（存储在文档存储中）关联。
5. **查询**：
   - 使用搜索词（`"justice breyer"`）查询检索器，返回相关的父文档。

```python
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import uuid

# 定义假设性问题的 Pydantic 模型
class HypotheticalQuestions(BaseModel):
    """生成假设性问题。"""
    questions: List[str] = Field(..., description="问题列表")

# 初始化 OpenAI 嵌入模型和 LLM
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
model = ChatOpenAI(model="gpt-4o-mini", api_key=my_api_key)

# 定义生成假设性问题的链
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
        "为以下文档生成正好 3 个假设性问题列表，这些问题是文档可以回答的：\n\n{doc}"
    )
    | model.with_structured_output(HypotheticalQuestions)
    | (lambda x: x.questions)
)

# 为文档生成假设性问题
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})

# 初始化向量存储和文档存储
vectorstore = Chroma(collection_name="hypo-questions", embedding_function=embed)
store = InMemoryByteStore()
id_key = "doc_id"

# 初始化检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# 为文档生成唯一 ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 创建问题文档
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )

# 将问题添加到向量存储，原始文档添加到文档存储
retriever.vectorstore.add_documents(question_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 查询检索器
retrieved_docs = retriever.invoke("justice breyer")
print(retrieved_docs)
```

## **结论**

`MultiVectorRetriever` 是高级文档检索任务的通用且强大的解决方案。通过利用文档的多种表示形式（如摘要、分块或假设性问题），它显著提高了搜索结果的准确性和相关性。它与向量存储和文档存储的集成实现了高效的索引和检索，而其可定制特性使其适应广泛的使用场景。无论您是构建检索增强生成（RAG）系统、语义搜索引擎还是文档摘要工具，`MultiVectorRetriever` 都提供了交付高质量结果所需的灵活性和能力。凭借其处理复杂检索场景的能力，它成为现代自然语言处理工作流程中的关键组件。