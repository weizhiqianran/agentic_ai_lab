# 如何为你的 RAG 设计选择合适的文本分割器

## 引言

在自然语言处理（NLP）和文档处理领域，将文本分割成较小、可管理的块是生成嵌入、语义搜索、摘要等任务的关键步骤。文本分割方法的选择会显著影响下游应用的性能。本文探讨了 `langchain-text-splitters` 库中的三种强大文本分割类：`**RecursiveCharacterTextSplitter**`、`**MarkdownHeaderTextSplitter**` 和 `**SentenceTransformersTokenTextSplitter**`。这些工具各有特定的使用场景，提供了独特的功能和能力。通过实际示例，我们将展示如何有效使用这些类并突出它们的优势。此外，通过比较表格帮助你选择适合自己需求的工具。

### 比较表格

| 功能/类别                        | RecursiveCharacterTextSplitter       | MarkdownHeaderTextSplitter           | SentenceTransformersTokenTextSplitter |
|----------------------------------|--------------------------------------|--------------------------------------|----------------------------------------|
| **主要使用场景**                 | 通用文本分割                         | 按Markdown标题分割                   | 基于token的嵌入分割                   |
| **分割机制**                     | 基于分隔符递归分割                   | 基于标题分割                         | 使用Sentence Transformers分词         |
| **可自定义分隔符**               | 是                                   | 否（标题预定义）                    | 否（使用模型分词器）                  |
| **保留文档结构**                 | 否                                   | 是（标题层次结构）                  | 否                                    |
| **基于Token分割**                | 否                                   | 否                                   | 是                                    |
| **块大小控制**                   | 是（基于字符）                       | 是（基于标题）                      | 是（基于token）                       |
| **块之间重叠**                   | 是                                   | 否                                   | 是                                    |
| **最佳适用场景**                 | 通用文本处理                         | Markdown文档                         | 嵌入生成、NLP任务                     |

这段代码使用 `pip` 安装了两个 Python 库：`langchain_community` 和 `langchain_experimental`。

- **`langchain_community`**：可能包含 LangChain 框架的社区驱动扩展或集成。
- **`langchain_experimental`**：包括 LangChain 的实验性功能或工具，可能处于积极开发或测试阶段。

`-qU` 参数确保安装过程安静（最少输出）并更新到最新版本（如果已安装）。

```python
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```

---

## 1. RecursiveCharacterTextSplitter

`RecursiveCharacterTextSplitter` 根据分隔符列表（例如 `\n\n`、`\n`、` ` 等）递归分割文本。它适用于将大段文本分解成较小的块，同时保留上下文。

### 示例 1：基本用法

本示例展示了 `RecursiveCharacterTextSplitter` 的基本用法。它初始化了一个指定块大小和重叠的分割器，然后将示例文本分割成较小的块，最后将每个块输出到控制台。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 初始化分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 示例文本
text = "这是一个示例文本。它将被分割成较小的块。目标是确保每个块易于管理并保留上下文。"

# 分割文本
chunks = splitter.split_text(text)

# 带索引输出块
for index, chunk in enumerate(chunks):
    print(f"块 {index + 1}: {chunk}")
```

### 示例 2：自定义分隔符

在此示例中，分割器使用自定义分隔符初始化，包括换行符、空格和句号。这允许更精细地控制文本分割，确保在逻辑标点或换行处进行分割。

```python
# 使用自定义分隔符初始化分割器
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", " ", "."],  # 按换行、空格和句号分割
    chunk_size=50,
    chunk_overlap=10
)

# 示例文本
text = "这是一个示例文本。\n它将被分割成较小的块。\n目标是确保每个块易于管理。"

# 分割文本
chunks = splitter.split_text(text)

# 带索引输出块
for index, chunk in enumerate(chunks):
    print(f"块 {index + 1}: {chunk}")
```

### 示例 3：分割文档

本示例展示了如何使用 `RecursiveCharacterTextSplitter` 分割多个文档。它引入了 `Document` 类，初始化分割器，并处理一组示例文档。然后打印每个分割文档的内容。

```python
from langchain_core.documents import Document

# 初始化分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 示例文档
documents = [
    Document(page_content="这是第一个文档。它包含一些文本。"),
    Document(page_content="这是第二个文档。它包含更多文本。")
]

# 分割文档
split_docs = splitter.split_documents(documents)

# 带索引输出分割文档
for index, doc in enumerate(split_docs):
    print(f"文档 {index + 1}: {doc.page_content}")
```

### 示例 4：使用元数据

在此示例中，分割器用于创建包含元数据的文档。它初始化分割器，准备示例文本及其对应的元数据，并生成将每个文本块与其元数据配对的文档。然后打印结果文档，显示内容和元数据。

```python
# 初始化分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 带元数据的示例文本
texts = ["这是一个示例文本。", "它将被分割成较小的块。"]
metadatas = [{"source": "doc1"}, {"source": "doc2"}]

# 创建带元数据的文档
docs = splitter.create_documents(texts, metadatas=metadatas)

# 带索引输出文档
for index, doc in enumerate(docs):
    print(f"文档 {index + 1}:\n  内容: {doc.page_content}\n  元数据: {doc.metadata}")
```

---

## 2. MarkdownHeaderTextSplitter

`MarkdownHeaderTextSplitter` 根据指定的标题分割 Markdown 文档，保留文档的层次结构。

### 示例 1：基本用法

本示例展示了 `MarkdownHeaderTextSplitter` 的基本用法。它定义了要分割的标题层次结构，初始化带有这些标题的分割器，并处理示例 Markdown 文本。然后打印对应于每个标题级别的块。

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 定义要分割的标题
headers_to_split_on = [
    ("#", "标题 1"),
    ("##", "标题 2"),
    ("###", "标题 3"),
]

# 初始化分割器
splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# 示例 Markdown 文本
markdown_text = """
# 一级标题
这是标题 1 下的文本。

## 二级标题
这是标题 2 下的文本。

### 三级标题
这是标题 3 下的文本。
"""

# 分割文本
chunks = splitter.split_text(markdown_text)

# 带索引输出块
for index, chunk in enumerate(chunks):
    print(f"块 {index + 1}: {chunk}")
```

### 示例 2：保留内容中的标题

在此示例中，分割器配置为在每个块的内容中保留标题。通过设置 `strip_headers=False`，原始标题保留在结果文本块中。

```python
# 初始化分割器并保留内容中的标题
splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)

# 分割文本
chunks = splitter.split_text(markdown_text)

# 输出块
for chunk in chunks:
    print(f"{chunk.page_content}\n")
```

### 示例 3：将每一行作为单独文档返回

本示例展示了如何配置分割器将 Markdown 文本的每一行视为单独文档。通过设置 `return_each_line=True`，分割器单独处理并返回每一行。

```python
# 初始化分割器将每一行作为单独文档返回
splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line=True)

# 分割文本
chunks = splitter.split_text(markdown_text)

# 带索引输出块
for index, chunk in enumerate(chunks):
    print(f"块 {index + 1}: {chunk}")
```

### 示例 4：与 RecursiveCharacterTextSplitter 结合

此示例展示了如何将 `MarkdownHeaderTextSplitter` 与 `RecursiveCharacterTextSplitter` 结合使用以实现更精细的分割。首先按标题分割 Markdown 文本，然后将每个块进一步分割成基于字符的较小块。最后打印最终分割块。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 初始化 Markdown 分割器
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# 分割 Markdown 文本
md_chunks = markdown_splitter.split_text(markdown_text)

# 初始化字符分割器
char_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

# 进一步分割 Markdown 块
final_chunks = char_splitter.split_documents(md_chunks)

# 带索引输出块
for index, chunk in enumerate(final_chunks):
    print(f"块 {index + 1}: {chunk.page_content}")
```

---

## 3. SentenceTransformersTokenTextSplitter

`SentenceTransformersTokenTextSplitter` 使用 Sentence Transformers 模型的分词器将文本分割成 token，确保与模型的 token 边界对齐。

### 示例 1：基本用法

本示例展示了 `SentenceTransformersTokenTextSplitter` 的基本用法。它初始化带有指定 token 参数的分割器，根据分词分割示例文本，并打印每个结果块。

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# 初始化分割器
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=20, tokens_per_chunk=100)

# 较长的示例文本
text = """
自然语言处理（NLP）是语言学、计算机科学和人工智能的一个子领域，关注计算机与人类语言之间的交互。
它研究如何编程计算机来处理和分析大量的自然语言数据。
目标是使计算机能够理解、解释和生成人类语言，以一种有意义且有用的方式。
NLP 技术被广泛应用于机器翻译、情感分析、语音识别和文本摘要等应用中。
NLP 的一个关键挑战是处理人类语言的模糊性和复杂性。
例如，同一个词在不同上下文中可能有多种含义，句子也可以有多种结构方式。
为了应对这些挑战，NLP 研究人员使用了多种技术，包括统计模型、机器学习算法和深度学习架构。
近年来，深度学习的进步显著提升了 NLP 任务的表现，例如语言建模、文本生成和问答。
这些进步得益于大规模预训练语言模型的发展，例如 BERT、GPT 和 T5，这些模型在海量文本数据上训练，并可针对特定任务进行微调。
"""

# 分割文本
chunks = splitter.split_text(text)

# 带索引输出块
for index, chunk in enumerate(chunks):
    print(f"块 {index + 1}:\n{chunk}\n")
```

### 示例 2：统计 Token 数量

在此示例中，分割器用于统计给定文本中的 token 数量。这有助于了解文本如何被分词并确保其符合模型约束。

```python
# 统计文本中的 token 数量
token_count = splitter.count_tokens(text=text)
print(f"Token 数量: {token_count}")
```

### 示例 3：分割文档

本示例展示了如何使用 `SentenceTransformersTokenTextSplitter` 分割多个文档。它处理一组示例文档，根据分词分割每个文档，并打印每个分割文档的内容。

```python
from langchain_core.documents import Document

# 示例文档
documents = [
    Document(page_content="这是第一个文档。"),
    Document(page_content="这是第二个文档。")
]

# 分割文档
split_docs = splitter.split_documents(documents)

# 输出分割文档
for doc in split_docs:
    print(doc.page_content)
```

### 示例 4：使用自定义模型

在此处，分割器使用自定义 Sentence Transformers 模型初始化。这允许根据所选模型的分词器进行分词，提供基于模型 token 边界的更准确分割。

```python
# 使用自定义 Sentence Transformers 模型初始化分割器
splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    tokens_per_chunk=50,
    chunk_overlap=10
)

# 分割文本
chunks = splitter.split_text(text)

# 带索引输出块
for index, chunk in enumerate(chunks):
    print(f"块 {index + 1}:\n{chunk}\n")
```

---

## 结论

文本分割是许多 NLP 和文档处理工作流程中的基础步骤。选择合适的工具取决于你的数据性质和任务的具体要求。

- **`RecursiveCharacterTextSplitter`** 是通用文本分割的多功能选择，提供定义分隔符和块大小的灵活性。
- **`MarkdownHeaderTextSplitter`** 擅长保留 Markdown 文档的层次结构，非常适合处理结构化内容。
- **`SentenceTransformersTokenTextSplitter`** 专为基于 token 的分割设计，确保与 Sentence Transformers 模型兼容，适用于嵌入生成和语义搜索等任务。

通过了解每个类的优势和使用场景，你可以做出明智的决策并优化你的文本处理流程。无论你处理的是纯文本、Markdown 还是分词数据，这些工具都能为有效分割文本提供强大的解决方案。