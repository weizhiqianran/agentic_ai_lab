# Chroma 向量存储快速参考 (LangChain)

## 引言

Chroma 向量存储 API 是一个强大的工具，用于管理和查询向量化的数据，能够无缝集成机器学习模型和自然语言处理任务。通过利用 Chroma，开发者可以高效地存储、检索和操作高维嵌入，使其成为构建智能应用的重要组成部分。本指南详细介绍了 Chroma 的核心功能，包括数据库持久化、文档操作、搜索功能以及实用工具函数。无论您处理的是文本、图像还是其他数据类型，Chroma 都提供了健壮且可扩展的向量存储和检索解决方案。

```python
!pip install -qU langchain-openai
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langchain-chroma>=0.1.2
!pip install -qU chromadb
```

---

## **1. 数据库持久化和加载**

### **1.1 保存 Chroma 数据库**
使用 `persist_directory` 参数将 Chroma 数据库保存到磁盘。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 使用 persist_directory 初始化 Chroma
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embed,
    persist_directory="./chroma_db"  # 数据将保存在此目录
)

# 创建文档
documents = [
    Document(page_content="敏捷的棕狐跳过懒狗。", metadata={"source": "寓言"}),
    Document(page_content="人工智能正在改变世界。", metadata={"source": "科技"}),
]

# 将文档添加到向量存储
vector_store.add_documents(documents=documents, ids=["doc1", "doc2"])
```

### **1.2 加载 Chroma 数据库**
从磁盘加载之前保存的 Chroma 数据库。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("api-key-openai")

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 从 persist_directory 加载 Chroma 数据库
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embed,
    persist_directory="./chroma_db"  # 与保存数据库时使用的相同目录
)

# 执行相似性搜索以验证加载
results = vector_store.similarity_search(query="AI", k=2)
for doc in results:
    print(f"加载的文档: {doc.page_content}")
```

### **1.3 检查集合是否存在**
在加载集合之前检查其是否存在。

```python
# 访问 langchain_chroma 使用的内部 Chroma 客户端
chroma_client = vector_store._client

# 检查集合是否存在
try:
    collection = chroma_client.get_collection(name="my_collection")
    print("集合存在且已加载。")
except Exception as e:
    print("集合不存在。")
```

### **1.4 删除持久化集合**
通过删除其目录来删除持久化集合。

```python
import shutil

# 删除持久化集合目录
shutil.rmtree("./chroma_db")
print("持久化集合目录已删除。")
```

---

## **2. 文档操作**

### **2.1 使用 `add_documents()` 添加文档**
将文档添加到 Chroma 向量存储。

```python
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 使用 /kaggle/working/ 作为 persist_directory
persist_directory = "/kaggle/working/chroma_db"

# 确保 persist_directory 存在
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 使用 persist_directory 初始化 Chroma
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embed,
    persist_directory=persist_directory
)

# 创建文档
documents = [
    Document(page_content="敏捷的棕狐跳过懒狗。", metadata={"source": "寓言"}),
    Document(page_content="人工智能正在改变世界。", metadata={"source": "科技"}),
]

# 将文档添加到向量存储
try:
    ids = ["doc1", "doc2"]
    added_ids = vector_store.add_documents(documents=documents, ids=ids)
    if added_ids == ids:
        print("文档添加成功。")
    else:
        print("添加文档失败，返回的 ID 不匹配。")
except Exception as e:
    print(f"添加文档时出错: {e}")
```

### **2.2 使用 `add_texts()` 添加文本**
本示例展示如何使用 `add_texts` 方法将文本数据添加到 Chroma 向量存储。提供的文本将被嵌入并存储，可选择附带元数据和 ID。

```python
# 使用 add_texts 的示例
texts = [
    "敏捷的棕狐跳过懒狗。",
    "人工智能正在改变世界。"
]

metadatas = [
    {"source": "寓言"},
    {"source": "科技"}
]

ids = ["text1", "text2"]

try:
    added_ids = vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    if added_ids == ids:
        print("文本添加成功。")
    else:
        print("添加文本失败，返回的 ID 不匹配。")
except Exception as e:
    print(f"添加文本时出错: {e}")
```

### **2.3 使用 `get` 和 `get_by_ids` 检索文档**
本示例展示如何使用 `get` 和 `get_by_ids` 函数从 Chroma 向量存储中检索文档。`get` 函数支持按元数据过滤、限制结果和分页，而 `get_by_ids` 通过特定 ID 检索文档。

```python
# 使用 `get` 函数
print("\n使用 `get` 函数:")
results = vector_store.get(
    ids=["doc1", "text1"],      # 按 ID 检索特定文档
    where={"source": "寓言"},  # 按元数据过滤
    limit=5,                    # 限制结果数量
    offset=0                    # 跳过前 N 个结果
)

# 打印结果
print("检索到的文档:")
for doc_id, document in zip(results["ids"], results["documents"]):
    print(f"ID: {doc_id}, 内容: {document}")
```

```python
# 使用 `get_by_ids` 函数
print("\n使用 `get_by_ids` 函数:")
document_ids = ["doc2", "text2"]
results = vector_store.get_by_ids(document_ids)

# 打印结果
print("按 ID 检索到的文档:")
for doc_id, document in zip(document_ids, results):
    print(f"ID: {doc_id}, 内容: {document}")
```

### **2.4 更新文档**
更新向量存储中的现有文档。

```python
# 更新文档
updated_document = Document(
    page_content="AI 正在革命化各行业。",
    metadata={"source": "科技"}
)

try:
    vector_store.update_documents(ids=["doc2"], documents=[updated_document])
    updated_doc = vector_store.get(ids=["doc2"])["documents"][0]
    if updated_doc == updated_document.page_content:
        print("文档更新成功。")
    else:
        print("更新文档失败，内容不匹配。")
except Exception as e:
    print(f"更新文档时出错: {e}")
```

### **2.5 删除文档**
按 ID 删除文档。

```python
# 删除文档
try:
    vector_store.delete(ids=["doc1"])
    deleted_doc = vector_store.get(ids=["doc1"])
    if not deleted_doc["documents"]:
        print("文档删除成功。")
    else:
        print("删除文档失败，文档仍然存在。")
except Exception as e:
    print(f"删除文档时出错: {e}")
```

---

## **3. 搜索操作**

1. **相似性搜索**：
   - 适用于根据语义相似性检索最相关的文档。
   - 当您需要简单的前 k 个结果时使用此方法。

2. **带分数的相似性搜索**：
   - 提供每个文档与查询匹配程度的额外洞察。
   - 可用于基于相似性阈值对结果进行排名或过滤。

3. **最大边际相关性 (MMR)**：
   - 在搜索结果中平衡相关性和多样性。
   - 当您想避免冗余或过于相似的文档时使用此方法。

### **3.1 相似性搜索**
搜索与查询相似的文档。

`similarity_search` 方法从向量存储中检索与给定查询最相似的文档。这对于基于语义相似性查找相关信息非常有用。

#### **参数**：
- `query` (str): 要搜索的输入查询。
- `k` (int): 要返回的文档数量，默认为 4。

```python
# 执行相似性搜索
query = "什么是 AI？"
results = vector_store.similarity_search(query, k=2)

# 打印结果
for doc in results:
    print(f"内容: {doc.page_content}, 元数据: {doc.metadata}")
```

### **3.2 带分数的相似性搜索**
搜索文档并检索相似性分数。

`similarity_search_with_score` 方法返回文档及其相似性分数。分数指示每个文档与查询的匹配程度，分数越低表示相似性越高。

#### **参数**：
- `query` (str): 要搜索的输入查询。
- `k` (int): 要返回的文档数量，默认为 4。

```python
# 执行带分数的相似性搜索
results = vector_store.similarity_search_with_score(query="AI", k=2)

for doc, score in results:
    print(f"分数: {score}, 内容: {doc.page_content}")
```

### **3.3 最大边际相关性 (MMR) 搜索**
使用 MMR 在搜索结果中平衡相似性和多样性。

`max_marginal_relevance_search` 方法优化了查询的相似性和所选文档的多样性。当您想避免冗余结果并确保获得多样化的相关文档时，这非常有用。

#### **参数**：
- `query` (str): 要搜索的输入查询。
- `k` (int): 要返回的文档数量，默认为 4。
- `fetch_k` (int): 在应用 MMR 之前获取的文档数量，默认为 20。
- `lambda_mult` (float): 介于 0 和 1 之间的值，决定相似性与多样性之间的权衡。值越高越倾向于相似性，值越低越倾向于多样性，默认为 0.5。

```python
# 执行 MMR 搜索
results = vector_store.max_marginal_relevance_search(
    query="AI",
    k=3,
    fetch_k=10,
    lambda_mult=0.5  # 值越高越倾向于相似性，越低越倾向于多样性
)

for doc in results:
    print(f"MMR 结果: {doc.page_content}")
```

---

## **4. 存储检索器**

### **4.1 将 Chroma 用作检索器**
将向量存储转换为检索器，以在 LangChain 管道中使用。

```python
# 创建检索器
retriever = vector_store.as_retriever()

# 使用检索器
query = "什么是 AI？"
docs = retriever.invoke(query)
for doc in docs:
    print(f"检索到的文档: {doc.page_content}")
```

### **4.2 使用更高的多样性 (MMR) 检索更多文档**
使用最大边际相关性 (MMR) 算法检索兼顾相关性和多样性的文档。

```python
# 创建带 MMR 的检索器
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# 使用检索器
query = "什么是 AI？"
docs = retriever.invoke(query)
for doc in docs:
    print(f"检索到的文档: {doc.page_content}")
```

### **4.3 为 MMR 获取更多文档但仅返回前 5 个**
为 MMR 获取较大的文档池，但仅返回前 5 个最相关且多样化的文档。

```python
# 创建带有 MMR 和更大获取池的检索器
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 50}
)

# 使用检索器
query = "什么是 AI？"
docs = retriever.invoke(query)
for doc in docs:
    print(f"检索到的文档: {doc.page_content}")
```

### **4.4 根据相关性分数阈值检索文档**
仅检索相似性分数高于指定阈值的文档。

```python
from sklearn.preprocessing import MinMaxScaler

# 将分数标准化到 [0, 1]
def normalize_scores(docs_with_scores):
    scores = [score for _, score in docs_with_scores]
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_scores = scaler.fit_transform([[score] for score in scores]).flatten()
    return [(doc, score) for (doc, _), score in zip(docs_with_scores, normalized_scores)]

# 获取带有相关性分数的文档
query = "天空是什么颜色？"
docs_with_scores = vector_store.similarity_search_with_relevance_scores(query)

# 标准化分数
normalized_docs_with_scores = normalize_scores(docs_with_scores)

# 根据标准化分数阈值过滤文档
score_threshold = 0.8
filtered_docs = [doc for doc, score in normalized_docs_with_scores if score >= score_threshold]

# 打印过滤后的文档
for doc in filtered_docs:
    print(f"检索到的文档: {doc.page_content}")
```

### **4.5 仅检索单个最相似的文档**
仅检索与查询最相关的单个文档。

```python
# 创建仅获取顶部文档的检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# 使用检索器
query = "什么是 AI？"
docs = retriever.invoke(query)
for doc in docs:
    print(f"检索到的文档: {doc.page_content}")
```

### **4.6 按元数据过滤文档**
检索匹配特定元数据过滤条件的文档，例如论文标题或出版年份。

```python
# 创建带有元数据过滤器的检索器
retriever = vector_store.as_retriever(
    search_kwargs={"filter": {"paper_title": "GPT-4 技术报告"}}
)

# 使用检索器
query = "什么是 AI？"
docs = retriever.invoke(query)
for doc in docs:
    print(f"检索到的文档: {doc.page_content}")
```

---

## **5. 类方法**

### **5.1 从文档创建向量存储**
直接从文档列表创建 Chroma 向量存储。

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 创建文档
documents = [
    Document(page_content="太阳从东方升起。", metadata={"source": "科学"}),
    Document(page_content="月亮围绕地球运行。", metadata={"source": "科学"}),
]

# 从文档创建 Chroma 向量存储
try:
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embed,
        collection_name="science_collection",
        persist_directory="./chroma_db_science"
    )
    
    # 通过检查集合是否存在来验证成功
    if vector_store._collection:
        print("向量存储创建成功。")
    else:
        print("创建向量存储失败，集合为空。")
except Exception as e:
    print(f"创建向量存储时出错: {e}")
```

### **5.2 从文本创建向量存储**
直接从原始文本创建 Chroma 向量存储。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 初始化 OpenAI 嵌入
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_api_key)

# 创建文本和元数据
texts = ["天空是蓝色的。", "草地是绿色的。"]
metadatas = [{"source": "自然"}, {"source": "自然"}]

# 从文本创建 Chroma 向量存储
try:
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embed,
        metadatas=metadatas,
        collection_name="nature_collection",
        persist_directory="./chroma_db_nature"
    )
    
    # 通过检查集合是否存在来验证成功
    if vector_store._collection:
        print("向量存储创建成功。")
    else:
        print("创建向量存储失败，集合为空。")
except Exception as e:
    print(f"创建向量存储时出错: {e}")
```

## 结论

在本指南中，我们通过实用示例探索了 Chroma 向量存储 API 的多功能性，从保存和加载数据库到执行高级搜索操作。通过遵循这些示例，您可以有效管理向量化的数据，将 Chroma 集成到您的工作流程中，并构建利用嵌入功能智能系统。无论您是初学者还是经验丰富的开发者，Chroma 直观的 API 和强大功能使其成为现代 AI 应用中不可或缺的工具。今天开始尝试 Chroma，释放基于向量的数据管理的全部潜力。