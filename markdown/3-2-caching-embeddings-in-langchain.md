# 在 LangChain 中缓存嵌入：提升 NLP 应用的性能

## 引言

嵌入（Embeddings）是现代自然语言处理（NLP）系统的核心组成部分，通过将文本数据转换为数字表示，使机器能够理解和处理文本。然而，为大型数据集或频繁重复的文本计算嵌入可能会消耗大量计算资源和时间。为了应对这一挑战，**缓存嵌入**成为一种优化性能和资源利用率的有效技术。通过将预计算的嵌入存储在键值存储中，系统可以避免重复计算，从而显著加快操作速度。

LangChain 中的 `CacheBackedEmbeddings` 类提供了一种无缝的方式来实现嵌入缓存。它作为一个嵌入模型的包装器，允许将嵌入缓存到多种存储后端，例如内存存储或基于磁盘的存储。这种方法不仅提高了效率，还增强了可扩展性，非常适合需要重复计算嵌入的应用场景，例如向量存储创建、语义搜索和检索增强生成（RAG）系统。

在本指南中，我们将探讨如何使用 `CacheBackedEmbeddings` 来缓存嵌入，展示其与 Chroma 等向量存储的集成，并通过实际示例突出其优势。无论您是处理大型数据集还是构建实时 NLP 应用，缓存嵌入都能帮助您实现更快、更高效的工作流程。

---

## 准备工作

### 安装必要的库
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
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全地获取 OpenAI API 密钥并初始化 OpenAI 嵌入模型。`OpenAIEmbeddings` 类用于创建嵌入模型实例，将文本转换为数字嵌入。

主要步骤：
1. **获取 API 密钥**：通过 Kaggle 的 `UserSecretsClient` 安全地检索 OpenAI API 密钥。
2. **初始化嵌入**：使用 `text-embedding-3-small` 模型和获取的 API 密钥初始化 `OpenAIEmbeddings` 类。

此设置确保嵌入模型已准备好用于下游任务，例如缓存嵌入或创建向量存储。

```python
from langchain_openai import OpenAIEmbeddings
from kaggle_secrets import UserSecretsClient

# 获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 示例替代配置
embed = OpenAIEmbeddings(model="text-embedding-3-large", base_url="http://20.243.34.136:2999/v1",
                        api_key="sk-RapHwqOGWbKT68V1531b7011388549F3Bb4316EcF8Ac28De")
```

---

## CacheBackedEmbeddings

### 示例 1：使用 `embed_documents()`
此示例展示如何使用 `embed_documents()` 函数为一组文本生成嵌入。如果嵌入尚未缓存，则会将其缓存。

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryStore

# 初始化文档嵌入存储（例如 InMemoryStore）
document_embedding_store = InMemoryStore()

# 创建 CacheBackedEmbeddings 实例
cached_embedder = CacheBackedEmbeddings(underlying_embeddings=embed, document_embedding_store=document_embedding_store)

# 要嵌入的文本列表
texts = ["你好，世界！", "这是一个测试。", "缓存嵌入很有用。"]

# 嵌入文档
embeddings = cached_embedder.embed_documents(texts)

# 打印嵌入
for text, embedding in zip(texts, embeddings):
    print(f"文本: {text}\n嵌入长度: {len(embedding)}\n")
```

### 示例 2：使用 `embed_query()`
此示例展示如何使用 `embed_query()` 函数为单个查询文本生成嵌入。如果提供了 `query_embedding_store`，则会缓存查询嵌入。

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryStore

# 初始化文档和查询嵌入存储（例如 InMemoryStore）
document_embedding_store = InMemoryStore()
query_embedding_store = InMemoryStore()

# 创建支持查询缓存的 CacheBackedEmbeddings 实例
cached_embedder = CacheBackedEmbeddings(
    underlying_embeddings=embed,
    document_embedding_store=document_embedding_store,
    query_embedding_store=query_embedding_store
)

# 要嵌入的查询文本
query_text = "生命的意义是什么？"

# 嵌入查询
query_embedding = cached_embedder.embed_query(query_text)

# 打印查询嵌入
print(f"查询: {query_text}\n嵌入长度: {len(query_embedding)}\n")
```

### 示例 3：使用 `from_bytes_store()`
此示例展示如何使用 `from_bytes_store()` 函数创建一个基于字节存储的 `CacheBackedEmbeddings` 实例。

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

# 初始化基于字节的文档嵌入存储（例如 InMemoryByteStore）
document_embedding_cache = InMemoryByteStore()

# 使用 from_bytes_store 创建 CacheBackedEmbeddings 实例
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embed,
    document_embedding_cache=document_embedding_cache,
    namespace="openai_embeddings"  # 可选命名空间以避免缓存冲突
)

# 要嵌入的文本列表
texts = ["使用字节存储缓存嵌入。", "这是另一个示例。"]

# 嵌入文档
embeddings = cached_embedder.embed_documents(texts)

# 打印嵌入
for text, embedding in zip(texts, embeddings):
    print(f"文本: {text}\n嵌入长度: {len(embedding)}\n")
```

---

## InMemoryByteStore

### 示例 1：使用 `mset()` 存储键值对
此示例展示如何使用 `mset()` 函数在 `InMemoryByteStore` 中存储多个键值对。

```python
from langchain.storage import InMemoryByteStore

# 初始化空存储
store = InMemoryByteStore()

# 设置多个键值对
store.mset([('key1', b'value1'), ('key2', b'value2'), ('key3', b'value3')])

# 验证键和值
print("使用 mset() 后的存储:", store.store)
```

### 示例 2：使用 `mget()` 检索值
此示例展示如何使用 `mget()` 函数检索与多个键关联的值。

```python
from langchain.storage import InMemoryByteStore

# 初始化带有数据的存储
store = InMemoryByteStore()
store.mset([('key1', b'value1'), ('key2', b'value2'), ('key3', b'value3')])

# 检索多个键的值
values = store.mget(['key1', 'key2', 'key4'])

# 打印检索到的值
print("检索到的值:", values)
```

### 示例 3：使用 `mdelete()` 删除键
此示例展示如何使用 `mdelete()` 函数删除特定键及其关联值。

```python
from langchain.storage import InMemoryByteStore

# 初始化带有数据的存储
store = InMemoryByteStore()
store.mset([('key1', b'value1'), ('key2', b'value2'), ('key3', b'value3')])

# 删除特定键
store.mdelete(['key1', 'key3'])

# 验证删除后的存储
print("使用 mdelete() 后的存储:", store.store)
```

### 示例 4：结合使用 `mset()`、`mget()` 和 `mdelete()`
此示例展示如何一起使用 `mset()`、`mget()` 和 `mdelete()`。

```python
from langchain.storage import InMemoryByteStore

# 初始化空存储
store = InMemoryByteStore()

# 设置多个键值对
store.mset([('key1', b'value1'), ('key2', b'value2'), ('key3', b'value3')])

# 检索多个键的值
values_before_deletion = store.mget(['key1', 'key2', 'key3'])
print("删除前的值:", values_before_deletion)

# 删除特定键
store.mdelete(['key1', 'key3'])

# 删除后检索值
values_after_deletion = store.mget(['key1', 'key2', 'key3'])
print("删除后的值:", values_after_deletion)
```

### 示例 5：使用 `yield_keys()` 迭代键
此示例展示如何使用 `yield_keys()` 函数迭代存储中的键，可选择按前缀过滤。

```python
from langchain.storage import InMemoryByteStore

# 初始化带有数据的存储
store = InMemoryByteStore()
store.mset([('key1', b'value1'), ('key2', b'value2'), ('key3', b'value3'), ('other_key', b'other_value')])

# 迭代所有键
print("所有键:")
for key in store.yield_keys():
    print(key)

# 迭代带有特定前缀的键
print("带有 'key' 前缀的键:")
for key in store.yield_keys(prefix='key'):
    print(key)
```

---

## 缓存嵌入

### 第一部分：使用 `LocalFileStore` 缓存嵌入

#### 目的：
本节展示如何使用 `LocalFileStore` 在磁盘上缓存嵌入，并使用 **Chroma** 创建向量存储。缓存机制避免了对相同文本重复计算嵌入，显著加快了重复操作的速度。

#### 步骤：
1. **初始化缓存**：
   - 创建一个 `LocalFileStore`，将嵌入存储在 `./cache/` 目录中。
   - 使用 `from_bytes_store()` 方法创建 `CacheBackedEmbeddings` 实例，包装嵌入模型并将嵌入缓存到指定存储中。

2. **加载和分割文档**：
   - 使用 `TextLoader` 加载文档（`state_of_the_union.txt`）。
   - 使用 `CharacterTextSplitter` 将文档分割成较小的块。

3. **创建 Chroma 向量存储**：
   - 使用 `Chroma.from_documents()` 方法从文档块创建向量存储。
   - 使用 Python 的 `time` 模块测量创建向量存储所需的时间。

4. **重用缓存嵌入**：
   - 使用相同的文档和缓存嵌入再次创建向量存储。
   - 测量第二次创建的时间，以展示缓存带来的性能提升。

5. **检查缓存嵌入**：
   - 打印缓存嵌入的键，验证嵌入是否已存储。

```python
import os
import time
import shutil
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# 定义缓存目录
cache_dir = "./cache/"

# 检查缓存文件夹是否存在，如果存在则删除
if os.path.exists(cache_dir):
    print(f"删除现有缓存文件夹: {cache_dir}")
    shutil.rmtree(cache_dir)

# 创建用于缓存嵌入的 LocalFileStore
store = LocalFileStore(cache_dir)

# 创建 CacheBackedEmbeddings 实例
cached_embedder = CacheBackedEmbeddings.from_bytes_store(embed, store, namespace=embed.model)

# 检查缓存（初始应为空）
print("初始缓存键:", list(store.yield_keys()))

# 加载文档并将其分割成块
raw_documents = TextLoader("/kaggle/input/state-of-the-union-txt/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# 使用缓存嵌入器创建 Chroma 向量存储
start_time = time.time()  # 开始计时
db = Chroma.from_documents(documents, cached_embedder, persist_directory="./chroma_db")
end_time = time.time()    # 结束计时
print(f"\n创建向量存储（初始）所需时间: {end_time - start_time:.2f} 秒")

# 再次创建向量存储（由于缓存，应更快）
start_time = time.time()  # 开始计时
db2 = Chroma.from_documents(documents, cached_embedder, persist_directory="./chroma_db")
end_time = time.time()    # 结束计时
print(f"创建向量存储（缓存）所需时间: {end_time - start_time:.2f} 秒")

# 检查部分缓存嵌入
print("\n缓存嵌入键:", list(store.yield_keys())[:5])
```

### 第二部分：使用 `InMemoryByteStore` 缓存嵌入

#### 目的：
本节展示如何使用内存存储（`InMemoryByteStore`）而不是基于磁盘的存储来缓存嵌入，并展示如何检索、检查和删除缓存嵌入。

#### 步骤：
1. **初始化内存缓存**：
   - 创建一个 `InMemoryByteStore`，将嵌入存储在内存中。
   - 使用内存存储创建 `CacheBackedEmbeddings` 实例。

2. **嵌入文档**：
   - 使用 `embed_documents()` 方法嵌入一组文本。
   - 将嵌入缓存到内存存储中。

3. **打印嵌入**：
   - 打印文本的嵌入（为简洁起见，仅显示前 10 个值）。

4. **检查缓存键**：
   - 打印缓存嵌入的键，验证嵌入已存储。

5. **检索缓存嵌入**：
   - 使用 `mget()` 方法检索缓存嵌入并打印（为简洁起见，仅显示前 10 个值）。

6. **从缓存中删除键**：
   - 使用 `mdelete()` 方法从缓存中删除一些键。
   - 打印剩余的缓存键以验证删除。

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

# 创建内存存储
store = InMemoryByteStore()

# 使用内存存储创建 CacheBackedEmbeddings 实例
cached_embedder = CacheBackedEmbeddings.from_bytes_store(embed, store, namespace=embed.model)

# 要嵌入的文本列表
texts = ["使用内存存储缓存嵌入。", "这是另一个示例。"]

# 嵌入文档
embeddings_list = cached_embedder.embed_documents(texts)

# 打印嵌入
print("文本的嵌入:")
for text, embedding in zip(texts, embeddings_list):
    print(f"文本: {text}\n嵌入（前 10 个值）: {embedding[:10]}\n")

# 检查嵌入后的缓存键
print("嵌入后的缓存键:")
cache_keys = list(store.yield_keys())
print(cache_keys)

# 使用 mget 检索缓存嵌入
print("\n检索缓存嵌入:")
cached_embeddings = store.mget(cache_keys)
for key, cached_embedding in zip(cache_keys, cached_embeddings):
    print(f"键: {key}\n缓存嵌入（前 10 个值）: {cached_embedding[:10]}\n")

# 从缓存中删除一些键
keys_to_delete = cache_keys[:1]  # 删除第一个键
store.mdelete(keys_to_delete)

# 检查删除后的缓存键
print("删除后的缓存键:")
print(list(store.yield_keys()))
```

## 结论

缓存嵌入是优化 NLP 工作流的一项重大变革，尤其是在处理大型数据集或重复计算时。通过利用 `CacheBackedEmbeddings`，开发者可以显著减少计算嵌入所需的时间和资源，从而实现更快、更具扩展性的应用。无论是构建向量存储、执行语义搜索还是实现检索增强生成，缓存嵌入都能确保系统保持高效和响应迅速。

支持不同存储后端（例如内存存储或基于磁盘的存储）的灵活性使 `CacheBackedEmbeddings` 成为适用于广泛用例的多功能工具。此外，命名空间支持和查询嵌入缓存等功能进一步增强了其实用性，使开发者能够根据具体需求定制缓存机制。

随着 NLP 系统复杂性和规模的不断增长，像缓存嵌入这样的技术将在确保最佳性能方面发挥越来越重要的作用。通过采用这些策略，您可以构建更快、更高效、更可靠的 NLP 应用，为用户和利益相关者创造价值。