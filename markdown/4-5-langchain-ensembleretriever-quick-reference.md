# LangChain EnsembleRetriever 快速参考

## 简介

**EnsembleRetriever** 是 LangChain 框架内一个强大的检索机制，旨在通过组合多个检索器的结果来增强信息检索能力。通过利用不同检索算法的优势，EnsembleRetriever 相比单一检索器能够实现更优越的性能。这种方法通常被称为“混合搜索”，它集成了稀疏检索器（如 BM25）和密集检索器（如 Chroma 或 FAISS），提供了一个更全面、更准确的检索系统。

### 主要特点
- **多个检索器的组合**：EnsembleRetriever 整合了多种检索方法，例如稀疏和密集检索器，以处理不同类型的查询和文档结构。
- **重新排序机制**：它使用 **倒数排名融合 (RRF)** 算法对来自各个检索器的结果进行重新排序，确保最相关的文档优先。
- **性能提升**：通过结合基于关键词和基于语义的检索，EnsembleRetriever 在复杂搜索场景中提供了更高的准确性和相关性。
- **可定制权重**：用户可以为单个检索器分配权重，以根据其优势优先选择特定的检索方法。
- **运行时配置**：检索器支持动态配置，允许用户在运行时调整参数（如要检索的文档数量）。

### 使用场景
1. **混合搜索**：结合稀疏和密集检索器，有效处理基于关键词和基于语义的查询。
2. **特定领域检索**：为专业领域（如法律、医疗或科学文档）集成定制检索器。
3. **元数据过滤**：通过基于元数据（如来源、日期或类别）过滤文档来优化搜索结果。
4. **动态查询处理**：在运行时调整检索策略，以适应不同的查询类型或用户偏好。
5. **提升搜索相关性**：在聊天机器人、推荐系统和知识库等应用中提升搜索相关性。

### 比较表：稀疏 vs. 密集 vs. 混合检索

| 特性                   | 稀疏检索（如 BM25）         | 密集检索（如 Chroma）       | 混合检索（EnsembleRetriever） |
|------------------------|----------------------------|----------------------------|------------------------------|
| **优势**              | 基于关键词匹配             | 语义相似性匹配             | 结合关键词和语义两种优势     |
| **劣势**              | 处理语义查询较弱           | 处理精确关键词匹配较弱     | 需要更多计算资源             |
| **使用场景**          | 关键词密集型查询           | 语义密集型查询             | 需要兼顾两者的复杂查询       |
| **性能**              | 精确关键词搜索速度快       | 语义搜索较慢但更准确       | 在混合任务中性能平衡         |
| **定制化**            | 限于关键词调优             | 限于嵌入调优               | 可通过权重高度定制化         |

---

## 准备工作

### 安装所需库
本节将安装使用 LangChain、OpenAI 嵌入和 Chroma 向量存储所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。
- `langchain-chroma`：实现与 Chroma 向量数据库的集成。
- `chromadb`：Chroma 向量数据库的核心库。

```python
!pip install -qU langchain-openai
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langchain-chroma>=0.1.2
!pip install -qU chromadb
!pip install -qU rank_bm25
```

### 初始化 OpenAI 嵌入
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI API 密钥并初始化 OpenAI 嵌入模型。`OpenAIEmbeddings` 类用于创建嵌入模型实例，将文本转换为数字嵌入。

主要步骤：
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI API 密钥。
2. **初始化嵌入**：使用 `text-embedding-3-small` 模型和获取的 API 密钥初始化 `OpenAIEmbeddings` 类。

此设置确保嵌入模型可用于后续任务，例如缓存嵌入或创建向量存储。

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
model = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=my_api_key)
```

---

## 1. 文档检索与管理

### 示例 1：基本 Ensemble 检索
本示例展示如何使用 BM25 检索器和 Chroma 向量存储检索器初始化 `EnsembleRetriever`，并为查询检索文档。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 向量存储检索器
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 检索文档
docs = ensemble_retriever.invoke("苹果")
print(docs)
```

### 示例 2：自定义检索参数
本示例展示如何自定义检索参数，例如要检索的文档数量 (`k`) 和为每个检索器分配的权重。

```python
# 更新 BM25 检索器以返回前 3 个文档
bm25_retriever.k = 3

# 更新 Chroma 检索器以返回前 3 个文档
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 3})

# 使用更新后的权重重新初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.7, 0.3]
)

# 检索文档
docs = ensemble_retriever.invoke("橙子")
print(docs)
```

---

## 2. 批量处理

### 示例 1：批量检索
本示例展示如何使用 `batch` 方法并行检索多个查询的文档。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 向量存储检索器
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 定义多个查询
queries = ["苹果", "橙子", "水果"]

# 批量检索所有查询的文档
batch_docs = ensemble_retriever.batch(queries)
for query, docs in zip(queries, batch_docs):
    print(f"查询: {query}")
    print(docs)
```

### 示例 2：自定义配置的批量检索
本示例展示如何在批量检索期间应用自定义配置（例如调整 Chroma 检索器的 `k` 值）。

```python
# 为 Chroma 检索器定义自定义配置
config = {"configurable": {"search_kwargs": {"k": 1}}}

# 使用自定义配置批量检索所有查询的文档
batch_docs = ensemble_retriever.batch(queries, config=config)
for query, docs in zip(queries, batch_docs):
    print(f"查询: {query}")
    print(docs)
```

---

## 3. 流式处理

### 示例 1：流式检索结果
本示例展示如何为查询流式传输检索结果。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 向量存储检索器
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 为查询流式传输文档
for doc in ensemble_retriever.stream("苹果"):
    print(doc)
```

### 示例 2：自定义流式行为
本示例展示如何通过调整流式传输的文档数量来自定义流式行为。

```python
# 更新 Chroma 检索器以在流式传输期间仅返回 1 个文档
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 1})

# 重新初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 为查询流式传输文档
for doc in ensemble_retriever.stream("橙子"):
    print(doc)
```

---

## 4. 错误处理与重试

### 示例 1：添加备用检索器
本示例展示如何添加备用检索器以处理失败情况。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 向量存储检索器
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 定义备用检索器
fallback_retriever = RunnableLambda(lambda x: [{"page_content": "备用文档", "metadata": {"source": "fallback"}}])

# 将备用检索器添加到 EnsembleRetriever
ensemble_retriever_with_fallback = ensemble_retriever.with_fallbacks([fallback_retriever])

# 检索文档（如果主检索器失败，将使用备用检索器）
docs = ensemble_retriever_with_fallback.invoke("未知查询")
print(docs)
```

### 示例 2：在失败时重试
本示例展示如何配置检索器在特定异常时进行重试。

```python
# 为 EnsembleRetriever 配置重试
ensemble_retriever_with_retry = ensemble_retriever.with_retry(
    retry_if_exception_type=(ValueError,), stop_after_attempt=3
)

# 检索文档（失败时将尝试重试）
docs = ensemble_retriever_with_retry.invoke("苹果")
print(docs)
```

---

## 5. 生命周期监听器

### 示例 1：添加生命周期监听器
本示例展示如何为检索器添加生命周期监听器。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 向量存储检索器
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 定义生命周期监听器
def on_start(run_obj):
    print(f"检索开始，输入为: {run_obj.inputs}")

def on_end(run_obj):
    print(f"检索结束，输出为: {run_obj.outputs}")

# 将生命周期监听器添加到 EnsembleRetriever
ensemble_retriever_with_listeners = ensemble_retriever.with_listeners(
    on_start=on_start, on_end=on_end
)

# 检索文档（将触发监听器）
docs = ensemble_retriever_with_listeners.invoke("橙子")
print(docs)
```

### 示例 2：自定义监听器行为
本示例展示如何通过向运行中添加元数据来自定义监听器行为。

```python
def on_start_with_metadata(run_obj):
    print(f"检索开始，输入为: {run_obj.inputs}，元数据为: {run_obj.metadata}")

# 添加带元数据的生命周期监听器
ensemble_retriever_with_listeners = ensemble_retriever.with_listeners(
    on_start=on_start_with_metadata, on_end=on_end
)

# 检索文档（监听器将触发并带有元数据）
docs = ensemble_retriever_with_listeners.invoke("苹果", config={"metadata": {"user": "test"}})
print(docs)
```

---

## 6. 配置与绑定

### 示例 1：在运行时绑定配置
本示例展示如何在运行时绑定配置（例如调整 Chroma 检索器的 `k` 值）。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.runnables import ConfigurableField

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 向量存储检索器
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(
    search_kwargs={"k": 2}
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_chroma",
        name="搜索参数",
        description="要使用的搜索参数",
    )
)

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 在运行时绑定配置
config = {"configurable": {"search_kwargs_chroma": {"k": 1}}}
docs = ensemble_retriever.invoke("橙子", config=config)
print(docs)
```

### 示例 2：为检索器绑定参数
本示例展示如何为检索器绑定额外的参数。

```python
# 为检索器绑定额外参数
ensemble_retriever_with_args = ensemble_retriever.bind(k=1)

# 使用绑定的参数检索文档
docs = ensemble_retriever_with_args.invoke("苹果")
print(docs)
```

---

## 最佳实践

**最佳实践** 示例展示了如何在 LangChain 中有效使用 `EnsembleRetriever`，通过结合多种检索技术来改进文档检索。这些示例突出了关键功能和配置，例如：

1. **混合搜索**：结合稀疏（基于关键词）和密集（基于语义）检索器，利用其互补优势。
2. **运行时配置**：动态调整检索器参数（例如要检索的文档数量）。
3. **自定义权重**：分配自定义权重以优先考虑 Ensemble 中的特定检索器。
4. **自定义检索器**：将自定义检索逻辑集成到 Ensemble 中，适用于特定领域用例。
5. **元数据过滤**：通过基于元数据的过滤优化搜索结果。

每个示例都基于一个通用设置（例如 BM25 和 Chroma 检索器），以确保一致性并减少冗余，便于在 Kaggle 笔记本或类似环境中理解和实现。这些示例旨在帮助用户在现实场景中理解和应用高级检索技术。

### 示例 1：结合稀疏和密集检索器进行混合搜索
本示例展示如何使用 `EnsembleRetriever` 结合稀疏检索器（`BM25Retriever`）和密集检索器（`Chroma Retriever`）。这种混合方法利用了基于关键词和语义搜索的优势。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 示例文档
doc_list_1 = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子是水果",
]

doc_list_2 = [
    "你喜欢苹果",
    "你喜欢橙子",
]

# 初始化 BM25 检索器（稀疏）
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

# 初始化 Chroma 检索器（密集）
chroma_vectorstore = Chroma.from_texts(
    doc_list_2, embed, metadatas=[{"source": 2}] * len(doc_list_2)
)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 检索文档
docs = ensemble_retriever.invoke("苹果")
print(docs)
```

### 示例 2：在运行时配置检索器
本示例展示如何使用 `ConfigurableField` 在运行时配置各个检索器的参数（例如调整要检索的文档数量）。

```python
from langchain_core.runnables import ConfigurableField

# 使 Chroma 检索器可配置
chroma_retriever = chroma_vectorstore.as_retriever(
    search_kwargs={"k": 2}
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_chroma",
        name="搜索参数",
        description="要使用的搜索参数",
    )
)

# 使用可配置的 Chroma 检索器重新初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 在运行时配置 Chroma 检索器
config = {"configurable": {"search_kwargs_chroma": {"k": 1}}}
docs = ensemble_retriever.invoke("苹果", config=config)
print(docs)
```

### 示例 3：调整检索器权重以实现自定义优先级
本示例展示如何调整分配给 Ensemble 中每个检索器的权重，以优先考虑某个检索器。

```python
# 使用自定义权重重新初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.7, 0.3]
)

# 检索文档
docs = ensemble_retriever.invoke("橙子")
print(docs)
```

### 示例 4：结合多个检索器以增强结果
本示例展示如何结合两个以上检索器（例如 BM25、Chroma 和自定义检索器）以进一步提升检索性能。

```python
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# 定义返回 Document 对象的自定义检索器
def custom_retriever(query: str) -> list:
    return [
        Document(
            page_content="自定义文档",
            metadata={"source": "custom"}
        )
    ]

# 将自定义检索器包装在 RunnableLambda 中
custom_retriever_runnable = RunnableLambda(custom_retriever)

# 使用多个检索器初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever, custom_retriever_runnable],
    weights=[0.4, 0.4, 0.2],
)

# 检索文档
docs = ensemble_retriever.invoke("苹果")
print(docs)
```

### 示例 5：使用 EnsembleRetriever 进行元数据过滤
本示例展示如何使用 `EnsembleRetriever` 进行元数据过滤以优化搜索结果。

```python
# 使用元数据过滤重新初始化 Chroma 检索器
chroma_retriever = chroma_vectorstore.as_retriever(
    search_kwargs={"k": 2, "filter": {"source": 2}}
)

# 重新初始化 EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 检索文档
docs = ensemble_retriever.invoke("苹果")
print(docs)
```

## 结论

**EnsembleRetriever** 是一个多功能且强大的工具，可增强信息检索系统。通过结合稀疏和密集检索器的优势，它弥补了单一检索方法的局限性，为复杂搜索任务提供了更稳健的解决方案。其使用倒数排名融合算法重新排序结果的能力确保了最相关文档的优先级，而对运行时配置和自定义权重的支持使其高度适应各种使用场景。

无论您是在构建聊天机器人、推荐系统还是知识库，EnsembleRetriever 都提供了一种灵活且有效的方法来提升搜索相关性和准确性。其混合方法在需要基于关键词和语义检索的场景中尤为有价值。