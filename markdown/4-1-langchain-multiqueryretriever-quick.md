# **LangChain `MultiQueryRetriever` 快速参考**

## **简介**

`MultiQueryRetriever` 是 LangChain 框架中的一个强大工具，旨在通过从单一输入查询生成多个查询来增强文档检索功能。它利用语言模型（LLM）创建原始查询的替代版本，为每个版本检索文档，并返回所有检索文档的唯一并集。这种方法有助于克服传统检索方法的局限性，例如仅依赖基于距离的相似性搜索，从而提供更全面的结果集。

本文通过实用示例探索 `MultiQueryRetriever` 的功能，涵盖初始化、查询生成、文档检索、流式处理以及重试机制和生命周期监听器等高级功能。无论您是构建问答系统、知识库还是搜索引擎，`MultiQueryRetriever` 都能显著提高搜索结果的相关性和多样性。

---

## **准备工作**

### **安装所需的库**
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

### **初始化 OpenAI 嵌入**
本节展示了如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI API 密钥并初始化 OpenAI 嵌入模型。`OpenAIEmbeddings` 类用于创建嵌入模型实例，将文本转换为数值嵌入。

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

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 初始化 LLM
model = ChatOpenAI(model="gpt-4o-mini", api_key=my_api_key)
```

---

## **1. 初始化和配置**

### **示例 1：基本初始化**
本示例展示了如何使用向量存储（`Chroma`）和嵌入模型（`OpenAIEmbeddings`）初始化 `MultiQueryRetriever`。它还向向量存储添加示例文档，并为查询检索相关文档。

```python
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

# 初始化向量存储和嵌入
vectorstore = Chroma(embedding_function=embed)
retriever = vectorstore.as_retriever()

# 初始化 MultiQueryRetriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model,
    include_original=True  # 包括原始查询
)

# 向向量存储添加文档（用于演示）
documents = [
    "锻炼改善心血管健康。",
    "健康饮食降低慢性疾病风险。",
    "冥想有助于减少压力和焦虑。"
]
vectorstore.add_texts(documents)

# 使用检索器通过 `invoke` 获取文档
query = "锻炼有哪些好处？"
relevant_docs = multi_query_retriever.invoke(query)

print("检索到的文档：")
for doc in relevant_docs:
    print(doc.page_content)
```

### **示例 2：自定义提示模板**
本示例展示了如何使用自定义提示模板与 `MultiQueryRetriever`。自定义提示生成输入查询的替代版本，检索器根据这些查询获取文档。

```python
from langchain_core.prompts import PromptTemplate

# 定义自定义提示模板
custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="生成此问题的 3 个不同版本：{question}"
)

# 使用自定义提示初始化 MultiQueryRetriever
multi_query_retriever_custom = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model,
    prompt=custom_prompt,
    include_original=False  # 不包括原始查询
)

# 使用自定义检索器获取文档
query = "冥想如何改善心理健康？"
relevant_docs = multi_query_retriever_custom.get_relevant_documents(query)

print("检索到的文档（自定义提示）：")
for doc in relevant_docs:
    print(doc.page_content)
```

---

## **2. 文档检索**

### **示例 1：为单一查询检索文档**
本示例使用 `get_relevant_documents` 方法检索与单一查询相关的文档。

```python
query = "锻炼有哪些好处？"
relevant_docs = multi_query_retriever.get_relevant_documents(query)

print("检索到的文档：")
for doc in relevant_docs:
    print(doc.page_content)
```

### **示例 2：为多个查询检索文档**
本示例展示了如何在循环中为多个查询检索文档。它逐个处理每个查询并打印检索到的文档。

```python
queries = [
    "锻炼有哪些好处？",
    "冥想如何改善心理健康？"
]

for query in queries:
    relevant_docs = multi_query_retriever.get_relevant_documents(query)
    print(f"检索到的文档：{query}")
    for doc in relevant_docs:
        print(doc.page_content)
```

### **示例 3：检索唯一文档**
本示例检索查询的文档，并使用 `unique_union` 方法确保结果唯一。

```python
query = "锻炼有哪些好处？"
relevant_docs = multi_query_retriever.get_relevant_documents(query)
unique_docs = multi_query_retriever.unique_union(relevant_docs)

print("唯一检索到的文档：")
for doc in unique_docs:
    print(doc.page_content)
```

---

## **3. 调用方法**

### **示例 1：使用 `invoke` 处理单一查询**
本示例展示了如何使用 `invoke` 方法为单一查询检索文档。这是同步检索文档的推荐方式。

```python
query = "锻炼有哪些好处？"
relevant_docs = multi_query_retriever.invoke(query)

print("检索到的文档（通过 invoke）：")
for doc in relevant_docs:
    print(doc.page_content)
```

### **示例 2：使用 `batch` 处理多个查询**
本示例展示了如何使用 `batch` 方法并行处理多个查询。它检索每个查询的文档并打印结果。

```python
queries = [
    "锻炼有哪些好处？",
    "冥想如何改善心理健康？"
]
batch_results = multi_query_retriever.batch(queries)

print("批量结果：")
for i, result in enumerate(batch_results):
    print(f"查询 {i + 1} 的结果：")
    for doc in result:
        print(doc.page_content)
```

### **示例 3：使用 `batch_as_completed` 进行并行处理**
本示例展示了如何使用 `batch_as_completed` 方法并行处理多个查询，并在完成后逐个生成结果。

```python
queries = [
    "锻炼有哪些好处？",
    "冥想如何改善心理健康？"
]
for idx, result in multi_query_retriever.batch_as_completed(queries):
    print(f"查询 {idx + 1} 的结果：")
    for doc in result:
        print(doc.page_content)
```

---

## **4. 查询生成**

### **示例 1：为单一问题生成查询**
本示例展示了如何使用 `generate_queries` 方法从单一输入问题生成多个查询。它使用回调管理器进行日志记录。

```python
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.callbacks.base import BaseCallbackHandler
import uuid

# 定义一个问题
question = "健康饮食有哪些好处？"

# 创建基本的回调处理器（可选）
class SimpleCallbackHandler(BaseCallbackHandler):
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"检索器开始处理查询：{query}")

# 初始化 CallbackManagerForRetrieverRun
run_id = str(uuid.uuid4())  # 生成唯一的运行 ID
handlers = [SimpleCallbackHandler()]  # 添加回调处理器
inheritable_handlers = []  # 可继承的处理器（可选）

run_manager = CallbackManagerForRetrieverRun(
    run_id=run_id,
    handlers=handlers,
    inheritable_handlers=inheritable_handlers
)

# 使用 MultiQueryRetriever 生成查询
generated_queries = multi_query_retriever.generate_queries(
    question=question,
    run_manager=run_manager  # 提供回调管理器
)

print("生成的查询：")
for query in generated_queries:
    print(query)
```

### **示例 2：为多个问题生成查询**
本示例展示了如何在循环中为多个问题生成查询。它为每个问题初始化一个新的回调管理器。

```python
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.callbacks.base import BaseCallbackHandler
import uuid

# 定义问题列表
questions = [
    "锻炼有哪些好处？",
    "冥想如何改善心理健康？"
]

# 创建基本的回调处理器（可选）
class SimpleCallbackHandler(BaseCallbackHandler):
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"检索器开始处理查询：{query}")

# 为每个问题生成查询
for question in questions:
    # 为每个问题初始化 CallbackManagerForRetrieverRun
    run_id = str(uuid.uuid4())  # 生成唯一的运行 ID
    handlers = [SimpleCallbackHandler()]  # 添加回调处理器
    inheritable_handlers = []  # 可继承的处理器（可选）

    run_manager = CallbackManagerForRetrieverRun(
        run_id=run_id,
        handlers=handlers,
        inheritable_handlers=inheritable_handlers
    )

    # 使用 MultiQueryRetriever 生成查询
    generated_queries = multi_query_retriever.generate_queries(
        question=question,
        run_manager=run_manager  # 提供回调管理器
    )
    print(f"生成的查询：{question}")
    for query in generated_queries:
        print(query)
```

---

## **5. 重试机制和生命周期监听器**

### **示例 1：添加重试机制**
本示例展示了如何为 `MultiQueryRetriever` 添加重试机制。如果发生异常，重试机制最多重试 3 次。

```python
retriever_with_retry = multi_query_retriever.with_retry(
    retry_if_exception_type=(Exception,),  # 在任何异常时重试
    stop_after_attempt=3  # 最大重试次数
)

query = "锻炼有哪些好处？"
relevant_docs = retriever_with_retry.invoke(query)

print("检索到的文档（带重试）：")
for doc in relevant_docs:
    print(doc.page_content)
```

### **示例 2：添加生命周期监听器**
本示例展示了如何为 `MultiQueryRetriever` 添加生命周期监听器。`on_start` 和 `on_end` 监听器分别在检索器开始和完成处理查询时触发。

```python
def on_start(run_obj):
    print(f"检索器开始，输入为：{run_obj.input}")

def on_end(run_obj):
    print(f"检索器完成，输出为：{run_obj.output}")

retriever_with_listeners = multi_query_retriever.with_listeners(
    on_start=on_start,
    on_end=on_end
)

query = "锻炼有哪些好处？"
relevant_docs = retriever_with_listeners.invoke(query)

print("检索到的文档（带监听器）：")
for doc in relevant_docs:
    print(doc.page_content)
```

### **示例 3：结合重试和监听器**
本示例将重试机制和生命周期监听器结合到一个检索器中。检索器将在发生异常时重试，并在执行期间触发 `on_start` 和 `on_end` 监听器。

```python
retriever_with_retry_and_listeners = multi_query_retriever.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=3
).with_listeners(
    on_start=on_start,
    on_end=on_end
)

query = "锻炼有哪些好处？"
relevant_docs = retriever_with_retry_and_listeners.invoke(query)

print("检索到的文档（带重试和监听器）：")
for doc in relevant_docs:
    print(doc.page_content)
```

---

## **6. 最佳实践**

### **关键要点**

- **构建向量数据库**：加载、分割和嵌入文档以创建可搜索的向量数据库。
- **简单使用 MultiQueryRetriever**：使用语言模型生成多个查询并检索文档。
- **自定义提示和输出解析器**：定义自定义提示和解析器，以针对特定用例定制查询生成过程。

### **构建示例向量数据库的代码**
此代码展示了如何使用博客文章作为数据源构建向量数据库。它加载博客文章，将其分割成较小的块，并使用 `Chroma` 和 `OpenAIEmbeddings` 创建向量数据库。

```python
# 构建示例向量数据库
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载博客文章
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# 分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# 向量数据库
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
```

### **简单使用 MultiQueryRetriever 的代码**
本示例展示了如何使用预构建的向量数据库与 `MultiQueryRetriever`。它使用语言模型（`ChatOpenAI`）初始化检索器，并为特定问题检索文档。启用了日志记录以显示生成的查询。

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# 初始化 LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)

question = "任务分解的方法有哪些？"
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=model)

# 设置查询日志
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.invoke(question)
len(unique_docs)
```

### **自定义提示和输出解析器的代码**
本示例展示了如何为 `MultiQueryRetriever` 自定义提示和输出解析器。它定义了一个自定义提示模板和输出解析器，以从单一输入问题生成多个查询。然后使用检索器根据生成的查询获取文档。

```python
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 输出解析器将 LLM 结果分割成查询列表
class LineListOutputParser(BaseOutputParser[List[str]]):
    """用于行列表的输出解析器。"""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 移除空行

output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个 AI 语言模型助手。你的任务是从给定的用户问题生成五个不同的版本，以从向量数据库中检索相关文档。通过从多个角度生成用户问题的变体，你的目标是帮助用户克服基于距离的相似性搜索的一些限制。请提供这些替代问题，每行一个。
    原始问题：{question}""",
)

# 初始化 LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=my_api_key)

# 链
llm_chain = QUERY_PROMPT | model | output_parser

# 其他输入
question = "任务分解的方法有哪些？"

# 运行
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)  # "lines" 是解析输出的键（属性名称）

# 结果
unique_docs = retriever.invoke("课程中关于回归的内容是什么？")
len(unique_docs)
```

## **结论**

`MultiQueryRetriever` 是 LangChain 框架中一个多功能且强大的组件，通过查询生成和检索优化提供高级文档检索功能。通过为查询生成多个版本并为每个版本检索文档，它确保了更全面和多样化的结果集，使其非常适合需要高质量搜索功能的应用。

通过本文提供的示例，我们展示了如何初始化检索器、生成查询、检索文档以及利用流式处理、重试机制和生命周期监听器等高级功能。这些工具使开发者能够构建更健壮和高效的检索系统，能够处理复杂查询并提供准确的结果。

无论您是在处理小型项目还是大规模应用，`MultiQueryRetriever` 都提供了增强文档检索工作流所需的灵活性和能力。通过将这些技术集成到您的项目中，您可以为提升搜索准确性、用户体验和系统可靠性开启新的可能性。