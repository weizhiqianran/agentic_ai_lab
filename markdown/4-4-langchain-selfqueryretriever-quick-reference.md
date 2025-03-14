# LangChain `SelfQueryRetriever` 快速参考

## **简介**

**SelfQueryRetriever** 是 LangChain 生态系统中一个强大的工具，旨在通过结合**语义搜索**和**结构化过滤**来增强文档检索功能。与仅依赖语义相似性的传统检索方法不同，SelfQueryRetriever 利用大型语言模型（LLM）生成结构化查询，可以根据元数据字段（如体裁、年份、评分或任何其他自定义属性）对文档进行过滤。这种混合方法使用户能够执行更精确且具有上下文意识的搜索，使其成为电影推荐、产品搜索或任何元数据至关重要的领域的宝贵工具。

LangChain 中的 **SelfQueryRetriever** 是一种专门设计的检索器，旨在通过将语义相似性搜索与基于元数据的过滤相结合，增强检索增强生成（RAG）系统。它使用查询构建 LLM 链将自然语言查询转换为可在 Milvus、Pinecone 或 Chroma 等向量数据库上执行的结构化查询。

### 主要特点

1. **自然语言到结构化查询的转换**：
   - 检索器解读用户查询并生成结构化查询。
   - 这些查询包括语义搜索标准和元数据过滤器，从而实现精确的文档检索。
2. **元数据过滤**：
   - 用户可以在查询中指定条件（例如“查找 2023 年的文档”）。
   - 检索器根据元数据字段（如日期、来源或标签）应用这些条件来过滤结果。
3. **与向量数据库的集成**：
   - 支持 Milvus、Pinecone 和 Chroma 等向量存储。
   - 检索器利用数据库的相似性搜索和过滤能力。
4. **可定制的查询翻译器**：
   - `structured_query_translator` 参数允许通过将内部查询格式翻译成特定于数据库的搜索参数，使检索器适应不同的向量存储。

---

## 准备工作

### 安装所需库
本节安装与 LangChain、OpenAI 嵌入和 Chroma 向量存储一起使用所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用程序。
- `langchain-chroma`：实现与 Chroma 向量数据库的集成。
- `chromadb`：Chroma 向量数据库的核心库。

```python
!pip install -qU lark
!pip install -qU langchain-openai
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langchain-chroma>=0.1.2
!pip install -qU chromadb
```

### 初始化 OpenAI 嵌入
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI API 密钥并初始化 OpenAI 嵌入模型。`OpenAIEmbeddings` 类用于创建嵌入模型实例，该实例将用于将文本转换为数值嵌入。

主要步骤：
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全检索 OpenAI API 密钥。
2. **初始化嵌入**：使用 `text-embedding-3-small` 模型和获取的 API 密钥初始化 `OpenAIEmbeddings` 类。

此设置确保嵌入模型可用于下游任务，例如缓存嵌入或创建向量存储。

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入和 LLM
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
model = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=my_api_key)
```

---

## **1. 检索功能**

### **带结构化过滤的基本检索**
此示例展示如何使用 `SelfQueryRetriever` 根据带有结构化过滤的查询（例如按体裁和评分等元数据过滤）检索文档。

```python
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有元数据的文档
documents = [
    Document(
        page_content="与外星人的太空冒险",
        metadata={"genre": "science fiction", "rating": 8.5},
    ),
    Document(
        page_content="关于小镇生活的喜剧",
        metadata={"genre": "comedy", "rating": 7.0},
    ),
]
vectorstore.add_documents(documents)

# 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 使用结构化过滤检索文档
result = retriever.invoke("评分大于 8 的科幻电影")
print(result)
```

### **带自定义元数据的检索**
此示例通过向文档添加自定义元数据并基于其进行过滤，扩展了基本检索。

```python
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
    AttributeInfo(
        name="year",
        description="电影发行的年份",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="电影导演的姓名",
        type="string",
    ),
    AttributeInfo(
        name="language",
        description="电影的语言",
        type="string",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有自定义元数据的文档
documents = [
    Document(
        page_content="与外星人的太空冒险",
        metadata={
            "genre": "science fiction",
            "rating": 8.5,
            "year": 2015,
            "director": "詹姆斯·卡梅隆",
            "language": "英语",
        },
    ),
    Document(
        page_content="关于小镇生活的喜剧",
        metadata={
            "genre": "comedy",
            "rating": 7.0,
            "year": 2010,
            "director": "韦斯·安德森",
            "language": "英语",
        },
    ),
    Document(
        page_content="关于黑客揭露阴谋的惊悚片",
        metadata={
            "genre": "thriller",
            "rating": 9.0,
            "year": 2020,
            "director": "大卫·芬奇",
            "language": "英语",
        },
    ),
    Document(
        page_content="巴黎背景的浪漫剧情片",
        metadata={
            "genre": "romance",
            "rating": 8.0,
            "year": 2018,
            "director": "索菲娅·科波拉",
            "language": "法语",
        },
    ),
    Document(
        page_content="关于机器人的动画电影",
        metadata={
            "genre": "animated",
            "rating": 9.5,
            "year": 2008,
            "director": "安德鲁·斯坦顿",
            "language": "英语",
        },
    ),
]

# 将文档添加到向量存储
vectorstore.add_documents(documents)

# 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)
```

```python
# 使用自定义元数据过滤检索文档
result = retriever.invoke("评分小于 8 的喜剧电影")
print("评分小于 8 的喜剧电影:", result)
```

```python
# 使用额外的元数据过滤器检索文档
result = retriever.invoke("詹姆斯·卡梅隆导演的科幻电影")
print("詹姆斯·卡梅隆导演的科幻电影:", result)
```

```python
result = retriever.invoke("2015 年后发行的电影")
print("2015 年后发行的电影:", result)
```

```python
result = retriever.invoke("评分大于 9 的动画电影")
print("评分大于 9 的动画电影:", result)
```

```python
result = retriever.invoke("法语电影")
print("法语电影:", result)
```

---

## **2. 批量处理**

### **批量检索多个查询**
此示例展示如何使用 `batch` 方法一次性处理多个查询。

```python
# 导入所需库
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有元数据的文档
documents = [
    Document(
        page_content="与外星人的太空冒险",
        metadata={"genre": "science fiction", "rating": 8.5},
    ),
    Document(
        page_content="关于小镇生活的喜剧",
        metadata={"genre": "comedy", "rating": 7.0},
    ),
]
vectorstore.add_documents(documents)

# 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 定义多个查询
queries = [
    "评分大于 8 的科幻电影",
    "评分小于 7.5 的喜剧电影",
]

# 执行批量检索
results = retriever.batch(queries)
for result in results:
    print(result)
```

### **带自定义配置的批量检索**
此示例通过向 `batch` 方法添加自定义配置（例如标签和元数据）扩展了前一示例。

```python
# 定义自定义配置
config = {"tags": ["batch_retrieval"], "metadata": {"user_id": 123}}

# 使用自定义配置执行批量检索
results = retriever.batch(queries, config=config)
for result in results:
    print(result)
```

---

## **3. 流式处理和事件处理**

### **流式检索结果**
此示例展示如何使用 `stream` 方法以流式方式检索文档。它包括所有必要的导入并定义了 `retriever` 对象。

```python
# 导入所需库
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有元数据的文档
documents = [
    Document(
        page_content="与外星人的太空冒险",
        metadata={"genre": "science fiction", "rating": 8.5},
    ),
    Document(
        page_content="关于小镇生活的喜剧",
        metadata={"genre": "comedy", "rating": 7.0},
    ),
]
vectorstore.add_documents(documents)

# 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 流式检索结果
for document in retriever.stream("评分大于 8 的科幻电影"):
    print(document)
```

### **处理检索事件**
此示例通过使用 `astream_events` 在检索期间处理实时事件扩展了前一示例。

```python
# 流式传输检索期间的所有事件
async for event in retriever.astream_events("评分小于 7.5 的喜剧电影", version="v2"):
    print(event)
```

```python
# 仅流式传输与检索器相关的事件
async for event in retriever.astream_events(
    "评分小于 7.5 的喜剧电影",
    version="v2",
    include_types=["retriever"],
):
    print(event)
```

```python
# 流式传输带有特定标签的事件
async for event in retriever.astream_events(
    "评分小于 7.5 的喜剧电影",
    version="v2",
    include_tags=["my_retriever"],
):
    print(event)
```

```python
# 流式传输排除特定类型的事件
async for event in retriever.astream_events(
    "评分小于 7.5 的喜剧电影",
    version="v2",
    exclude_types=["on_retriever_start"],
):
    print(event)
```

```python
# 流式传输带有组合过滤器的事件
async for event in retriever.astream_events(
    "评分小于 7.5 的喜剧电影",
    version="v2",
    include_types=["retriever"],
    include_tags=["my_retriever"],
):
    print(event)
```

```python
# 定义自定义事件模式
async def custom_event_schema(query: str):
    async for event in retriever.astream_events(query, version="v2"):
        custom_event = {
            "event_name": event["event"],
            "run_id": event["run_id"],
            "data": event["data"],
        }
        print(custom_event)

# 运行自定义事件模式
await custom_event_schema("评分小于 7.5 的喜剧电影")
```

---

## **4. 实用功能和生命周期监听器**

### **添加生命周期监听器**
此示例展示如何为检索器添加生命周期监听器以跟踪其执行情况。它包括所有必要的导入并定义了 `retriever` 对象。

```python
# 导入所需库
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有元数据的文档
documents = [
    Document(
        page_content="与外星人的太空冒险",
        metadata={"genre": "science fiction", "rating": 8.5},
    ),
    Document(
        page_content="关于小镇生活的喜剧",
        metadata={"genre": "comedy", "rating": 7.0},
    ),
]
vectorstore.add_documents(documents)

# 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 定义生命周期监听器
def on_start(run):
    print(f"检索开始: {run}")

def on_end(run):
    print(f"检索结束: {run}")

# 将监听器绑定到检索器
listener_retriever = retriever.with_listeners(on_start=on_start, on_end=on_end)

# 使用监听器调用检索器
listener_retriever.invoke("评分大于 8 的科幻电影")
```

### **使用回退检索器**
此示例通过添加回退检索器以优雅地处理失败情况扩展了前一示例。

```python
from langchain_core.runnables import RunnableLambda

# 定义回退检索器
fallback_retriever = RunnableLambda(lambda x: [{"page_content": "回退文档"}])

# 为检索器添加回退
fallback_enabled_retriever = listener_retriever.with_fallbacks([fallback_retriever])

# 使用回退调用检索器
result = fallback_enabled_retriever.invoke("无效查询")
print(result)
```

---

## **5. 配置和定制**

### **自定义搜索参数**
此示例展示如何自定义搜索参数（例如搜索类型和关键字参数）。它包括所有必要的导入并定义了 `retriever` 对象。

```python
# 导入所需库
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有元数据的文档
documents = [
    Document(
        page_content="与外星人的太空冒险",
        metadata={"genre": "science fiction", "rating": 8.5},
    ),
    Document(
        page_content="关于小镇生活的喜剧",
        metadata={"genre": "comedy", "rating": 7.0},
    ),
]
vectorstore.add_documents(documents)

# 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 自定义搜索参数
custom_retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
    search_type="mmr",  # 最大边际相关性
    search_kwargs={"k": 5},  # 检索前 5 个文档
)

# 使用自定义搜索参数检索文档
result = custom_retriever.invoke("评分大于 8 的科幻电影")
print(result)
```

### **检索的可配置替代方案**
此示例通过使用 `configurable_alternatives` 在运行时切换不同的检索器扩展了前一示例。

```python
from langchain_core.runnables import ConfigurableField

# 定义替代检索器
alternative_retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
    search_type="similarity",  # 替代搜索类型
)

# 配置替代方案
configurable_retriever = custom_retriever.configurable_alternatives(
    ConfigurableField(id="retriever"),
    default_key="default",
    alternative=alternative_retriever,
)

# 使用默认检索器
result = configurable_retriever.invoke("评分大于 8 的科幻电影")
print(result)

# 切换到替代检索器
result = configurable_retriever.with_config(configurable={"retriever": "alternative"}).invoke("评分小于 7.5 的喜剧电影")
print(result)
```

---

## **最佳实践**

### **自定义 SelfQueryRetriever 以包含相似性分数**
此示例展示如何通过子类化 `SelfQueryRetriever` 在文档元数据中包含相似性分数。这对于了解每个文档与查询的匹配程度很有用。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from typing import Any, Dict, List

# 定义用于过滤的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影的体裁。可选值包括 ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="电影发行的年份",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="电影导演的姓名",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影的 1-10 分评分",
        type="float",
    ),
]
document_content_description = "电影的简短概要"

# 初始化向量存储和嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
model = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=my_api_key)
vectorstore = Chroma(embedding_function=embed)

# 向向量存储添加带有元数据的文档
documents = [
    Document(
        page_content="一群科学家复活恐龙，混乱随之而来",
        metadata={"genre": "science fiction", "rating": 7.7, "year": 1993, "director": "史蒂文·斯皮尔伯格"},
    ),
    Document(
        page_content="关于一个男孩和他的狗的温馨故事",
        metadata={"genre": "drama", "rating": 8.5, "year": 2009, "director": "拉斯·霍尔斯道姆"},
    ),
]
vectorstore.add_documents(documents)

# 子类化 SelfQueryRetriever 以包含相似性分数
class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        """获取文档并添加分数信息。"""
        try:
            # 执行带分数的相似性搜索
            results = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
            if not results:
                print("没有文档匹配查询。")
                return []
            
            docs, scores = zip(*results)
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score
            return list(docs)
        except Exception as e:
            print(f"检索期间发生错误: {e}")
            return []

# 创建自定义检索器
retriever = CustomSelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 检索带相似性分数的文档
result = retriever.invoke("评分小于 8 的恐龙电影")
print(result)
```

### **根据多个元数据字段过滤文档**
此示例展示如何使用 `SelfQueryRetriever` 根据多个元数据字段（例如体裁、年份和评分）过滤文档。

```python
# 使用多个元数据过滤器检索文档
result = retriever.invoke("1990 年后发行且评分大于 7 的科幻电影")
print(result)
```

### **使用自定义搜索参数的 SelfQueryRetriever**
此示例展示如何为 `SelfQueryRetriever` 自定义搜索参数（例如搜索类型和关键字参数）。

```python
# 自定义搜索参数
custom_retriever = SelfQueryRetriever.from_llm(
    model,
    vectorstore,
    document_content_description,
    metadata_field_info,
    search_type="mmr",       # 最大边际相关性
    search_kwargs={"k": 5},  # 检索前 5 个文档
)

# 使用自定义搜索参数检索文档
result = custom_retriever.invoke("评分大于 8 的剧情电影")
print(result)
```

### **使用回退检索器处理边缘情况**
此示例展示如何添加回退检索器以处理查询无匹配文档的情况。

```python
from langchain_core.runnables import RunnableLambda

# 定义回退检索器
fallback_retriever = RunnableLambda(lambda x: [{"page_content": "未找到匹配的文档。"}])

# 为检索器添加回退
fallback_enabled_retriever = retriever.with_fallbacks([fallback_retriever])

# 使用回退调用检索器
result = fallback_enabled_retriever.invoke("评分大于 9 的恐怖电影")
print(result)
```

---

## **结论**

**SelfQueryRetriever** 对于需要语义理解和结构化过滤的应用来说是一个革命性的工具。通过结合大型语言模型和元数据感知检索的优势，它为广泛的用例提供了灵活且强大的解决方案。无论您是构建电影推荐系统、电子商务搜索引擎，还是依赖元数据的任何应用，SelfQueryRetriever 都能通过提供更准确和相关的结果显著提升用户体验。

其从自然语言输入**自动生成结构化查询**的能力使其对非技术专长的用户也易于使用，而对**自定义元数据字段**的支持确保了它能适应多样化的应用需求。此外，**相似性分数传播**和**回退机制**等功能进一步增强了其健壮性和可用性。

总之，**SelfQueryRetriever** 不仅仅是一个检索文档的工具——它是一个用于构建智能、元数据感知搜索系统的全面解决方案。通过利用其功能，开发者可以创建既用户友好又高效的应用，确保用户能够快速、准确地找到所需信息。