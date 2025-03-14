# **LangChain `ParentDocumentRetriever` 快速参考**

## 简介

`ParentDocumentRetriever` 的工作原理是首先将文档分割成较小的片段，然后存储和索引这些片段。在检索时，它会根据查询检索这些小片段，随后查找它们的父文档 ID，返回较大的原始文档或预定义的较大片段。这种方法在保持足够小的片段以获得准确嵌入的同时，也能保留足够的上下文以实现有意义的检索。

### 主要特点

- **分块策略**：检索器允许同时进行小片段检索和父文档查找，确保嵌入准确反映内容的含义，同时保留上下文。
- **动态检索**：它可以根据检索到的片段动态获取父文档，提升返回结果的相关性。
- **元数据处理**：支持元数据字段，用户可以在检索时保留与子文档相关的有用信息。

### 使用场景

1. **上下文检索**：适用于需要理解特定信息周围上下文的场景，例如问答系统。
2. **高效文档管理**：在需要快速管理和访问大型文档而不丢失重要上下文信息的场景中非常有用。

因此，`ParentDocumentRetriever` 是 LangChain 中的一个强大工具，能够增强用户检索和交互大型文本数据的能力，同时保持上下文的完整性。

---

## 准备工作

### 安装所需库
本节将安装使用 LangChain、OpenAI 嵌入和 Chroma 向量存储所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。
- `langchain-chroma`：支持与 Chroma 向量数据库的集成。
- `chromadb`：Chroma 向量数据库的核心库。

```python
!pip install -qU langchain-openai
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langchain-chroma>=0.1.2
!pip install -qU chromadb
```

### 初始化 OpenAI 嵌入
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI API 密钥并初始化 OpenAI 嵌入模型。`OpenAIEmbeddings` 类用于创建嵌入模型实例，将文本转换为数字嵌入。

主要步骤：
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全检索 OpenAI API 密钥。
2. **初始化嵌入**：使用 `text-embedding-3-small` 模型和获取的 API 密钥初始化 `OpenAIEmbeddings` 类。

此设置确保嵌入模型可用于下游任务，例如缓存嵌入或创建向量存储。

```python
from langchain_openai import OpenAIEmbeddings
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
```

---

## 1. 文档检索与管理

### **添加和检索文档**

此示例展示如何向 `ParentDocumentRetriever` 添加文档并根据查询检索它们，展示了索引和检索文档的基本工作流程。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="doc_retrieval_add_retrieve")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 示例文档（创建 Document 对象）
documents = [
    Document(page_content="文档 1 的内容在此。", metadata={"source": "doc1"}),
    Document(page_content="文档 2 的内容在此。", metadata={"source": "doc2"}),
]

# 将文档添加到检索器
retriever.add_documents(documents)

# 根据查询检索相关文档
query = "内容在此"
relevant_docs = retriever.invoke(query)

print("检索到的文档：")
for doc in relevant_docs:
    print(f"来源：{doc.metadata['source']}，内容：{doc.page_content}")
```

### **根据元数据过滤检索到的文档**

此示例展示如何在检索文档时根据特定的元数据条件进行过滤。这在希望将搜索结果缩小到特定来源或具有特定属性的文档时非常有用。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="doc_retrieval_filter_metadata")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever 并指定子文档元数据字段
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    child_metadata_fields=["source"],
)

# 示例文档（使用 Document 对象）
documents = [
    Document(page_content="文档 1 的内容在此。", metadata={"source": "内部"}),
    Document(page_content="文档 2 的内容在此。", metadata={"source": "外部"}),
]

# 将文档添加到检索器
retriever.add_documents(documents)

# 使用元数据过滤器检索相关文档
query = "内容在此"
metadata_filter = {"source": "内部"}
relevant_docs = retriever.invoke(query, metadata=metadata_filter)

print("检索到的来源为 '内部' 的文档：")
for doc in relevant_docs:
    print(f"来源：{doc.metadata['source']}，内容：{doc.page_content}")
```

### **更新检索器中的文档**

虽然 `ParentDocumentRetriever` 未提供直接更新文档的方法，但可以通过移除现有文档并添加更新版本来管理更新。此示例展示如何执行此类更新。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="doc_retrieval_update_docs")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 示例文档（使用 Document 对象）
documents = [
    Document(page_content="文档 1 的原始内容。", metadata={"doc_id": "doc1"}),
    Document(page_content="文档 2 的原始内容。", metadata={"doc_id": "doc2"}),
]

# 将文档添加到检索器
retriever.add_documents(documents)

# 更新后的文档（使用 Document 对象）
updated_document = Document(page_content="文档 1 的更新内容。", metadata={"doc_id": "doc1"})

# 移除旧文档并重新初始化检索器（假设没有直接删除方法）
retriever = ParentDocumentRetriever(
    vectorstore=Chroma(embedding_function=embed, collection_name="doc_retrieval_update_docs"),
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加更新后的文档
retriever.add_documents([updated_document])

# 检索更新后的文档
query = "更新内容"
relevant_docs = retriever.invoke(query)

print("检索到的更新文档：")
for doc in relevant_docs:
    print(f"文档 ID：{doc.metadata['doc_id']}，内容：{doc.page_content}")
```

---

## 2. 批处理

### **批量添加多个文档**

此示例展示如何通过单次批处理操作向 `ParentDocumentRetriever` 添加多个文档。批处理在处理大量数据时可以提高效率。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="batch_processing_add")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 批量文档（使用 Document 对象）
documents = [
    Document(page_content="文档 1 的内容。", metadata={"source": "doc1"}),
    Document(page_content="文档 2 的内容。", metadata={"source": "doc2"}),
    Document(page_content="文档 3 的内容。", metadata={"source": "doc3"}),
]

# 批量添加文档
retriever.add_documents(documents)

# 通过检索验证添加
query = "文档 2 的内容。"
relevant_docs = retriever.invoke(query)

print("检索到的文档：")
for doc in relevant_docs:
    print(f"来源：{doc.metadata['source']}，内容：{doc.page_content}")
```

### **批量检索多个查询的文档**

此示例展示如何同时对多个查询执行批量检索。批量检索在处理多个搜索请求时可以显著加快速度。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="batch_processing_retrieve_queries")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="Python 是一种多功能的编程语言。", metadata={"doc_id": "doc1"}),
    Document(page_content="Java 广泛用于企业应用程序。", metadata={"doc_id": "doc2"}),
    Document(page_content="JavaScript 为网络提供动力。", metadata={"doc_id": "doc3"}),
]
retriever.add_documents(documents)

# 查询列表
queries = [
    "编程语言",
    "企业应用程序",
    "网页开发",
]

# 批量检索所有查询的文档
results = retriever.batch(queries)

for i, docs in enumerate(results):
    print(f"查询 {i+1} 的结果：")
    for doc in docs:
        print(f"文档 ID：{doc.metadata['doc_id']}，内容：{doc.page_content}")
    print("---")
```

### **使用配置进行批处理**

此示例展示如何为每次批量调用使用不同的配置。这种灵活性允许根据特定需求定制检索行为。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="batch_processing_no_runnableconfig")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="机器学习使计算机能够从数据中学习。", metadata={"doc_id": "doc1"}),
    Document(page_content="深度学习是机器学习的一个子集。", metadata={"doc_id": "doc2"}),
    Document(page_content="人工智能包括机器学习和深度学习。", metadata={"doc_id": "doc3"}),
]
retriever.add_documents(documents)

# 定义批量输入
inputs = ["机器学习", "深度学习"]

# 批量检索文档
results = retriever.batch(inputs)

for i, docs in enumerate(results):
    print(f"查询 '{inputs[i]}' 的结果：")
    for doc in docs:
        print(f"文档 ID：{doc.metadata['doc_id']}，内容：{doc.page_content}")
    print("---")
```

---

## 3. 流式处理

### **流式检索文档**

此示例展示如何根据查询流式检索文档。流式处理允许在结果可用时逐步处理它们，这对实时应用非常有益。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="streaming_retrieval")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="流处理允许实时处理数据。", metadata={"doc_id": "doc1"}),
    Document(page_content="批处理一次性处理大量数据。", metadata={"doc_id": "doc2"}),
    Document(page_content="实时分析需要高效的流处理。", metadata={"doc_id": "doc3"}),
]
retriever.add_documents(documents)

# 流式检索文档
query = "实时数据处理"

# 处理流方法返回的每个块
for chunk in retriever.stream(query):
    for doc in chunk:  # 遍历块中的文档
        print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
```

### **检索过程中的流式事件**

此示例展示如何生成和处理与检索过程相关的事件流。事件流式处理可以提供检索器内部操作和进展的洞察。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="streaming_events_retrieval")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="事件驱动架构对事件做出响应。", metadata={"doc_id": "doc1"}),
    Document(page_content="流式数据支持实时处理。", metadata={"doc_id": "doc2"}),
    Document(page_content="异步事件提升系统响应性。", metadata={"doc_id": "doc3"}),
]
retriever.add_documents(documents)

# 检索过程中的流式事件
query = "实时处理"

# 遍历流方法返回的块
for chunk in retriever.stream(query):  # 每个块是文档对象列表
    for doc in chunk:  # 遍历块中的单独文档
        print(f"事件：检索到的文档 ID {doc.metadata['doc_id']}，内容：{doc.page_content}")
```

### **结合流式处理与监听器**

虽然 `ParentDocumentRetriever` 主要支持同步流式处理，但可以通过集成对流数据的反应监听器来增强检索过程。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 定义监听器函数
def on_start(run_obj):
    print("检索过程已开始。")

def on_end(run_obj):
    print("检索过程已完成。")

def on_error(run_obj):
    print("检索过程中发生错误。")

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="streaming_with_listeners")
store = InMemoryStore()

# 使用监听器初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
).with_listeners(
    on_start=on_start,
    on_end=on_end,
    on_error=on_error
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="监听器函数可以对检索事件做出反应。", metadata={"doc_id": "doc1"}),
    Document(page_content="事件监听器增强检索器的功能。", metadata={"doc_id": "doc2"}),
    Document(page_content="适当的错误处理确保系统健壮性。", metadata={"doc_id": "doc3"}),
]
retriever.add_documents(documents)

# 使用监听器进行流式检索
query = "检索事件"

for chunk in retriever.stream(query):  # 处理每个块（文档对象列表）
    for doc in chunk:  # 处理块中的每个文档对象
        print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
```

---

## 4. 事件处理

### **绑定同步生命周期监听器**

此示例展示如何将同步生命周期监听器（`on_start` 和 `on_end`）绑定到 `ParentDocumentRetriever`。这些监听器在检索过程的不同阶段执行自定义函数。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 定义监听器函数
def on_start(run_obj):
    print("检索已开始。")

def on_end(run_obj):
    print("检索已结束。")

def on_error(run_obj):
    print("检索过程中发生错误。")

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="event_handling_bind_listeners")
store = InMemoryStore()

# 使用监听器初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
).with_listeners(
    on_start=on_start,
    on_end=on_end,
    on_error=on_error
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="事件监听器允许在检索期间执行自定义操作。", metadata={"doc_id": "doc1"}),
    Document(page_content="它们可用于记录检索活动。", metadata={"doc_id": "doc2"}),
]
retriever.add_documents(documents)

# 调用检索以触发监听器
query = "事件监听器"

retrieved_docs = retriever.invoke(query)

for doc in retrieved_docs:
    print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
```

### **分发自定义事件**

虽然 `ParentDocumentRetriever` 未直接提供分发自定义事件的方法，但可以在应用逻辑中集成自定义事件分发。此示例展示如何在检索过程中模拟自定义事件处理。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 定义自定义事件分发器
def dispatch_custom_event(event_name, data):
    print(f"自定义事件：{event_name}，数据：{data}")

# 定义带有自定义事件分发的监听器函数
def on_start(run_obj):
    dispatch_custom_event("检索开始", {"query": run_obj.input})

def on_end(run_obj):
    dispatch_custom_event("检索完成", {"num_documents": len(run_obj.output)})

def on_error(run_obj):
    dispatch_custom_event("检索错误", {"error": str(run_obj.error)})

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="event_handling_custom_events")
store = InMemoryStore()

# 使用自定义事件监听器初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
).with_listeners(
    on_start=on_start,
    on_end=on_end,
    on_error=on_error
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="自定义事件为处理检索过程提供了灵活性。", metadata={"doc_id": "doc1"}),
    Document(page_content="它们可以根据特定应用需求进行定制。", metadata={"doc_id": "doc2"}),
]
retriever.add_documents(documents)

# 调用检索以触发自定义事件
query = "自定义事件"

try:
    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
except Exception as e:
    print(f"检索过程中发生异常：{e}")
```

### **使用自定义回调与监听器**

此示例展示如何将自定义回调函数与检索器的生命周期监听器集成，以在检索过程中执行额外操作，例如日志记录或数据转换。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 定义自定义回调函数
def log_start(run_obj):
    print(f"[日志] 针对查询 '{run_obj.input}' 的检索已开始")

def log_end(run_obj):
    print(f"[日志] 检索已结束。检索到的文档数量：{len(run_obj.output)}")

def log_error(run_obj):
    print(f"[日志] 检索失败，错误：{run_obj.error}")

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="event_handling_custom_callbacks")
store = InMemoryStore()

# 使用自定义回调初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
).with_listeners(
    on_start=log_start,
    on_end=log_end,
    on_error=log_error
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="回调增强了检索过程的功能。", metadata={"doc_id": "doc1"}),
    Document(page_content="它们允许在检索期间执行自定义操作。", metadata={"doc_id": "doc2"}),
]
retriever.add_documents(documents)

# 调用检索以触发自定义回调
query = "检索中的回调"

retrieved_docs = retriever.invoke(query)

for doc in retrieved_docs:
    print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
```

---

## 5. 错误处理

### **使用 `with_retry` 实现重试逻辑**

此示例展示如何使用 `with_retry` 方法为 `ParentDocumentRetriever` 添加重试逻辑。检索器将在遇到指定异常时尝试重新执行检索操作。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="error_handling_retry_logic")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="可靠的检索对应用程序至关重要。", metadata={"doc_id": "doc1"}),
]
retriever.add_documents(documents)

# 为检索器应用重试逻辑
retriever_with_retry = retriever.with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(ValueError,),
    wait_exponential_jitter=True
)

# 使用重试逻辑调用检索
query = "可靠的检索"

try:
    retrieved_docs = retriever_with_retry.invoke(query)
    for doc in retrieved_docs:
        print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
except ValueError as e:
    print(f"重试后检索失败：{e}")
```

### **使用重试处理特定异常**

此示例展示如何配置 `with_retry` 方法以处理特定异常类型。检索器仅在遇到指定异常时重试，允许更精细的错误管理。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 使用唯一集合名称初始化向量存储和文档存储
vectorstore = Chroma(embedding_function=embed, collection_name="error_handling_specific_retries")
store = InMemoryStore()

# 初始化 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加示例文档（使用 Document 对象）
documents = [
    Document(page_content="选择性错误处理允许精确控制。", metadata={"doc_id": "doc1"}),
]
retriever.add_documents(documents)

# 应用选择性重试逻辑
retriever_with_retry = retriever.with_retry(
    stop_after_attempt=2,
    retry_if_exception_type=(ValueError,),  # 仅在 ValueError 时重试
    wait_exponential_jitter=False
)

# 使用选择性重试逻辑调用检索
query = "选择性错误处理"

try:
    retrieved_docs = retriever_with_retry.invoke(query)
    for doc in retrieved_docs:
        print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
except Exception as e:
    print(f"检索失败：{e}")
```

### **结合重试与回退**

虽然 `ParentDocumentRetriever` 将回退单独分类，但将重试逻辑与回退结合可以增强健壮性。此示例展示如何设置重试机制和回退检索器，以确保即使面对多次失败也能成功检索。

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema import Document

# 初始化文本分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

# 为主要和回退使用唯一集合名称初始化向量存储和文档存储
primary_vectorstore = Chroma(embedding_function=embed, collection_name="error_handling_retry_with_fallbacks_primary")
fallback_vectorstore = Chroma(embedding_function=embed, collection_name="error_handling_retry_with_fallbacks_fallback")
primary_store = InMemoryStore()
fallback_store = InMemoryStore()

# 初始化主要检索器
primary_retriever = ParentDocumentRetriever(
    vectorstore=primary_vectorstore,
    docstore=primary_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 为主检索器添加示例文档
primary_retriever.add_documents([
    Document(page_content="主检索器文档内容。", metadata={"doc_id": "primary_doc"}),
])

# 为主检索器应用重试逻辑
primary_with_retry = primary_retriever.with_retry(
    stop_after_attempt=2,
    retry_if_exception_type=(ValueError,),
    wait_exponential_jitter=False
)

# 初始化回退检索器
fallback_retriever = ParentDocumentRetriever(
    vectorstore=fallback_vectorstore,
    docstore=fallback_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 为回退检索器添加示例文档
fallback_retriever.add_documents([
    Document(page_content="回退检索器文档内容。", metadata={"doc_id": "fallback_doc"}),
])

# 将主检索器与回退检索器结合
combined_retriever = primary_with_retry.with_fallbacks(
    fallbacks=[fallback_retriever],
    exceptions_to_handle=(ValueError,)
)

# 调用组合检索器
query = "健壮的检索"

retrieved_docs = combined_retriever.invoke(query)

for doc in retrieved_docs:
    print(f"检索到的文档：{doc.metadata['doc_id']}，内容：{doc.page_content}")
```

---

## 6. 最佳实践

### **使用 ParentDocumentRetriever 进行完整和较大块检索**

#### **加载和准备文档**
本节展示如何从文本文件中加载文档并使用 `TextLoader` 准备它们以供检索。

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
loaders = [
    TextLoader("/kaggle/input/paul-graham-essay/paul_graham_essay.txt"),
    TextLoader("/kaggle/input/paul-graham-essay/state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
```

#### **使用小块检索完整文档**
在此模式下，文档被分割成小块以进行索引和检索。`ParentDocumentRetriever` 配置为仅使用子分割器。

```python
# 此文本分割器用于创建子文档
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 用于索引子块的向量存储
vectorstore = Chroma(collection_name="full_documents", embedding_function=embed)

# 父文档的存储层
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# 将文档添加到检索器
retriever.add_documents(docs, ids=None)

# 列出文档存储中的键
print("键的数量：", list(store.yield_keys()))

# 在向量存储中执行相似性搜索
sub_docs = vectorstore.similarity_search("布雷耶法官")
print("内容长度：", len(sub_docs[0].page_content))
print(sub_docs[0].page_content)

# 使用检索器检索文档
retrieved_docs = retriever.invoke("布雷耶法官")
print("内容长度：", len(retrieved_docs[0].page_content))
print(len(retrieved_docs[0].page_content))
```

#### **使用父分割检索较大块**
在此模式下，文档首先被分割成较大块（父文档），然后进一步分割成较小块（子文档）。这在粒度和上下文之间提供了平衡。

```python
# 此文本分割器用于创建父文档
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# 此文本分割器用于创建子文档
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 用于索引子块的向量存储
vectorstore = Chroma(collection_name="split_parents", embedding_function=embed)

# 父文档的存储层
store = InMemoryStore()

# 使用父分割器和子分割器初始化检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 将文档添加到检索器
retriever.add_documents(docs)

# 检查文档存储中的键数量
print("键的数量：", len(list(store.yield_keys())))
```

```python
# 在向量存储中执行相似性搜索
sub_docs = vectorstore.similarity_search("布雷耶法官")
print("内容长度：", len(sub_docs[0].page_content))
print(sub_docs[0].page_content)
```

```python
# 使用检索器检索文档
retrieved_docs = retriever.invoke("布雷耶法官")
print("内容长度：", len(retrieved_docs[0].page_content))
print(retrieved_docs[0].page_content)
```

## 结论

`ParentDocumentRetriever` 是一个多功能的工具，结合了细粒度相似性搜索的优势以及在更广泛上下文级别检索文档的能力。通过允许用户将文档分割成多个层级，它提供了一个可定制且高效的检索过程。无论是检索小片段进行精确搜索，还是检索较大块进行上下文分析，`ParentDocumentRetriever` 都提供了一个直观且可扩展的解决方案来应对文档检索挑战。它与文本分割工具、`Chroma` 等向量存储以及基于元数据的存储的无缝集成，确保其能够适应广泛的使用场景，在检索任务中提供准确性和上下文。