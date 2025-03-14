# LangChain 内置工具：DuckDuckGo（网络搜索）

## 介绍

`DuckDuckGoSearchRun` 是 LangChain 框架中的一款强大工具，它将 DuckDuckGo 的搜索功能集成到您的应用程序中。该工具专为开发者和 AI 从业者设计，使您能够以编程方式执行网络搜索，检索实时信息，并将搜索结果集成到您的工作流程中。无论是构建聊天机器人、研究助手还是数据管道，`DuckDuckGoSearchRun` 都能提供一种无缝的方式，让您访问网络上的最新信息，同时不会牺牲用户隐私，因为 DuckDuckGo 以其隐私优先的方法而闻名。

该工具高度可定制，支持重试、回退和可配置的替代方案等功能，使其在各种用例中都表现得既强大又灵活。凭借其简单的 API 和与 LangChain 生态系统的集成，`DuckDuckGoSearchRun` 是需要动态、实时数据检索的应用程序中不可或缺的组成部分。

---

## 准备工作

### 安装必需的库
本节将安装使用 LangChain、OpenAI 嵌入模型、Anthropic 模型、DuckDuckGo 搜索及其他实用工具所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型和 API 的集成。
- `langchain-anthropic`：支持与 Anthropic 模型和 API 的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。
- `langgraph`：用于在 LangChain 中构建和可视化基于图的工作流的库。
- `duckduckgo-search`：支持以编程方式访问 DuckDuckGo 的搜索功能。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU duckduckgo-search
```

### 初始化 OpenAI 和 Anthropic 聊天模型
本节展示了如何使用 Kaggle 的 `UserSecretsClient` 安全地获取 OpenAI 和 Anthropic 的 API 密钥，并初始化它们的聊天模型。`ChatOpenAI` 和 `ChatAnthropic` 类用于创建这些模型的实例，可用于自然语言处理任务，如文本生成和对话 AI。

**关键步骤：**
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全地检索 OpenAI 和 Anthropic 的 API 密钥。
2. **初始化聊天模型**：
   - 使用 `gpt-4o-mini` 模型和获取的 OpenAI API 密钥初始化 `ChatOpenAI` 类。
   - 使用 `claude-3-5-latest` 模型和获取的 Anthropic API 密钥初始化 `ChatAnthropic` 类。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化语言模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
# model = ChatAnthropic(model="claude-3-5-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

---

## 1. 初始化和设置

### 示例 1：基本初始化
此示例展示了如何使用默认设置初始化 `DuckDuckGoSearchRun` 工具。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化 DuckDuckGoSearchRun 工具
tool = DuckDuckGoSearchRun()

# 示例用法
result = tool.invoke("Python 编程")

word_count = len(result.split())
print("词数：", word_count)
print(result)
```

### 示例 2：自定义配置初始化
此示例展示了如何使用自定义配置初始化工具，例如设置描述并启用详细日志记录。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 使用自定义设置初始化工具
tool = DuckDuckGoSearchRun(
    description="用于查找编程资源的自定义 DuckDuckGo 搜索工具。",
    verbose=True
)

# 示例用法
result = tool.invoke("LangChain 框架")

word_count = len(result.split())
print("\n词数：", word_count)
print(result)
```

---

## 2. 搜索执行

### 示例 1：简单搜索执行
此示例展示了如何使用 `invoke` 方法执行简单搜索。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化工具
tool = DuckDuckGoSearchRun()

# 执行搜索
result = tool.invoke("最新 AI 新闻")

word_count = len(result.split())
print("\n词数：", word_count)
print(result)
```

### 示例 2：使用 ToolCall 输入进行搜索
此示例展示了如何使用 `ToolCall` 输入与工具配合使用，这对于结构化输入非常有用。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化工具
tool = DuckDuckGoSearchRun()

# 使用 ToolCall 输入执行搜索
tool_input = {
    "args": {"query": "2023 年机器学习趋势"},
    "id": "1",
    "name": tool.name,
    "type": "tool_call"
}
result = tool.invoke(tool_input)

# 访问特定属性（如果适用）
if hasattr(result, 'tool_call_id'):
    print("工具调用 ID：", result.tool_call_id)
if hasattr(result, 'name'):
    print("名称：", result.name)
if hasattr(result, 'content'):
    print("内容：", result.content)
```

---

## 3. 流式处理和批量处理

### 示例 1：批量处理
此示例展示了如何以批量模式处理多个搜索查询。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化工具
tool = DuckDuckGoSearchRun()

# 要处理的查询列表
queries = ["Python 编程", "机器学习", "数据科学"]

# 执行批量处理
results = tool.batch(queries)
for result in results:
    print(result)
    print("-"*80)
```

### 示例 2：流式搜索结果
此示例展示了如何为单个查询流式传输搜索结果。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化工具
tool = DuckDuckGoSearchRun()

# 流式传输搜索结果
query = "最新 AI 新闻"
for result in tool.stream(query):
    print(result)
```

---

## 4. 错误处理和重试

### 重试配置参数

1. **`retry_if_exception_type`**：
   - 此参数指定应触发重试的异常类型。
   - 在示例中，`retry_if_exception_type=(Exception,)` 表示**任何异常**（因为 `Exception` 是 Python 中所有异常的基类）都会触发重试。

2. **`stop_after_attempt`**：
   - 此参数指定在放弃之前的最多重试次数。
   - 在示例中，`stop_after_attempt=3` 表示工具在失败后最多重试 **3 次**（包括初次尝试）。

3. **`wait_exponential_jitter`**（示例中未使用，但值得一提）：
   - 如果启用，此参数会在重试之间的等待时间中添加随机“抖动”（短暂延迟），以避免分布式系统中的“雷鸣群效应”。
   - 重试之间的等待时间呈指数增长（例如 1 秒、2 秒、4 秒等），并通过抖动避免同步重试。

### 示例 1：失败后重试
此示例展示了如何配置工具以在特定异常时进行重试。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 使用重试配置初始化工具
tool = DuckDuckGoSearchRun().with_retry(
    retry_if_exception_type=(Exception,),  # 在任何异常时重试
    stop_after_attempt=3                   # 最多重试 3 次
)

# 使用重试逻辑执行搜索
result = tool.invoke("Python 编程")
print(result)
```

```python
# 带有抖动的指数退避
tool = DuckDuckGoSearchRun().with_retry(
    retry_if_exception_type=(Exception,),
    wait_exponential_jitter=True,  # 启用带抖动的指数退避
    stop_after_attempt=3
)

# 使用重试逻辑执行搜索
result = tool.invoke("Python 编程")
print(result)
```

```python
# 特定异常类型：
# 如果只想在特定异常（如 ConnectionError 或 TimeoutError）时重试，可以指定它们。
tool = DuckDuckGoSearchRun().with_retry(
    retry_if_exception_type=(ConnectionError, TimeoutError),
    stop_after_attempt=3
)

# 使用重试逻辑执行搜索
result = tool.invoke("Python 编程")
print(result)
```

### 5. 回退机制如何工作

1. **主要工具**：
   - 主要工具是序列中的第一个工具。在示例中，这是 `DuckDuckGoSearchRun` 实例（`primary_tool`）。
   - 当调用工具（例如 `tool.invoke("Python 编程")`）时，首先执行主要工具。

2. **回退工具**：
   - 如果主要工具失败（例如引发异常），回退机制就会启动。
   - 回退工具按照在 `with_fallbacks` 方法中指定的顺序依次尝试。在示例中，有一个回退工具（`fallback_tool`），它是另一个 `DuckDuckGoSearchRun` 实例。

3. **执行流程**：
   - 首先执行主要工具。
   - 如果主要工具成功，则返回其结果，不使用回退工具。
   - 如果主要工具失败，则执行第一个回退工具。
   - 如果第一个回退工具失败，则执行下一个回退工具，依此类推，直到：
     - 某个工具成功并返回结果。
     - 所有工具都失败，抛出异常。

4. **异常处理**：
   - 默认情况下，如果所有工具都失败，则抛出最后一个工具的异常。
   - 您可以使用 `with_fallbacks` 方法中的 `exceptions_to_handle` 和 `exception_key` 参数（示例中未展示）来自定义异常处理方式。

### 示例 2：失败时回退
此示例展示了如何配置回退行为，以便在主要搜索失败时使用备用工具。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化主要工具
primary_tool = DuckDuckGoSearchRun()

# 初始化回退工具（例如另一个搜索工具或模拟响应）
fallback_tool = DuckDuckGoSearchRun(description="回退搜索工具")

# 配置带有回退的工具
tool = primary_tool.with_fallbacks([fallback_tool])

# 使用回退逻辑执行搜索
result = tool.invoke("Python 编程")
print(result)
```

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化主要工具
primary_tool = DuckDuckGoSearchRun()

# 初始化回退工具
fallback_tool_1 = DuckDuckGoSearchRun(description="回退工具 1")
fallback_tool_2 = DuckDuckGoSearchRun(description="回退工具 2")

# 配置多个回退的工具
tool = primary_tool.with_fallbacks([fallback_tool_1, fallback_tool_2])

# 使用回退逻辑执行搜索
result = tool.invoke("Python 编程")
print(result)
```

---

## 6. 配置和绑定

### 示例 1：绑定附加参数

#### 描述
**绑定**允许您将附加参数或配置附加到工具上，从而创建一个预设这些参数的新工具实例。当您希望在使用工具时重复使用特定设置而无需每次都传递这些设置时，这非常有用。

在此示例中：
- 初始化 `DuckDuckGoSearchRun` 工具。
- 使用 `bind` 方法将自定义参数（`query_filter="site:github.com"`）附加到工具。
- 然后使用绑定的工具（`custom_tool`）调用搜索查询，自定义参数会自动应用。

#### 工作原理
1. **初始化**：
   - 使用默认设置创建 `DuckDuckGoSearchRun` 工具。

2. **绑定**：
   - 在工具上调用 `bind` 方法，传递参数 `query_filter="site:github.com"`。
   - 这会创建一个新的工具实例（`custom_tool`），其中预设了 `query_filter` 参数。

3. **调用**：
   - 当调用 `custom_tool.invoke("Python 编程")` 时，搜索查询 `"Python 编程"` 与绑定的参数 `query_filter="site:github.com"` 结合。
   - 工具执行搜索，仅返回来自 `github.com` 的结果。

4. **输出**：
   - 返回并打印搜索结果。

```python
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化工具
tool = DuckDuckGoSearchRun()

# 绑定附加参数（例如自定义搜索过滤器）
custom_tool = tool.bind(query_filter="site:github.com")

# 使用绑定的参数执行搜索
result = custom_tool.invoke("Python 编程")
print(result)
```

### 示例 2：可配置的替代方案

#### 描述
**可配置的替代方案**允许您定义工具的多个版本，并在运行时在它们之间切换。当您希望根据上下文或配置为同一工具提供不同的实现或行为时，这非常有用。

在此示例中：
- 初始化主要工具（`primary_tool`）和替代工具（`alternative_tool`）。
- 使用 `configurable_alternatives` 方法配置带有这些替代方案的工具。
- 根据运行时配置，工具可以调用主要实现或替代实现。

#### 工作原理
1. **初始化**：
   - 创建主要工具（`primary_tool`）和替代工具（`alternative_tool`）。

2. **配置**：
   - 在主要工具上调用 `configurable_alternatives` 方法，传递：
     - 一个 `ConfigurableField` 实例，ID 为 `"search_tool"`。
     - 默认键 `"primary"`，指定默认使用的工具。
     - 替代工具（`alternative_tool`），可在运行时选择。

3. **默认调用**：
   - 当调用 `tool.invoke("Python 编程")` 时，默认使用主要工具。

4. **替代调用**：
   - 当调用 `tool.with_config(configurable={"search_tool": "alternative"}).invoke("Python 编程")` 时，使用替代工具。

5. **输出**：
   - 返回并打印所选工具的搜索结果。

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables.utils import ConfigurableField

# 初始化主要工具
primary_tool = DuckDuckGoSearchRun()

# 初始化替代工具（例如模拟搜索工具）
alternative_tool = DuckDuckGoSearchRun(description="替代搜索工具")

# 配置带有替代方案的工具
tool = primary_tool.configurable_alternatives(
    ConfigurableField(id="search_tool"),
    default_key="primary",
    alternative=alternative_tool
)

# 使用默认工具执行搜索
result = tool.invoke("Python 编程")
print(result)

# 使用替代工具执行搜索
result = tool.with_config(configurable={"search_tool": "alternative"}).invoke("Python 编程")
print(result)
```

---

## 最佳实践

### 示例 1：通过实时网络搜索增强聊天机器人
此示例展示了如何将 `DuckDuckGoSearchRun` 集成到由大型语言模型（LLM）驱动的聊天机器人中，以提供实时、最新信息。聊天机器人可以回答有关当前事件、最新新闻或任何需要网络实时数据的话题。

#### 说明
1. **搜索工具集成**：
   - 使用 `DuckDuckGoSearchRun` 工具执行实时网络搜索。
   - 将搜索结果作为输入的一部分传递给 LLM。

2. **LLM 提示**：
   - 指示 LLM 使用搜索结果提供准确且最新的答案。

3. **输出**：
   - 聊天机器人结合 LLM 的推理能力和实时搜索结果，有效回答用户查询。

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化 DuckDuckGoSearchRun 工具
search_tool = DuckDuckGoSearchRun()

# 为聊天机器人定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。如有需要，使用搜索工具查找最新信息。"),
    ("user", "{input}")
])

# 创建一个集成搜索工具和 LLM 的链
chain = (
    {"input": lambda x: x["input"], "search_results": lambda x: search_tool.invoke(x["input"])}
    | prompt
    | model
    | StrOutputParser()
)

# 示例用法
response = chain.invoke({"input": "2023 年 AI 的最新趋势是什么？"})
print(response)
```

### 示例 2：构建研究助手
此示例展示了如何创建一个研究助手，使用 `DuckDuckGoSearchRun` 收集特定主题的信息，并使用 LLM 进行总结。这对于市场研究、学术研究或竞争分析等任务尤其有用。

#### 说明
1. **搜索工具集成**：
   - 使用 `DuckDuckGoSearchRun` 工具收集特定主题（例如 2023 年可再生能源趋势）的信息。

2. **LLM 总结**：
   - 将搜索结果传递给 LLM，生成简洁且易读的总结。

3. **输出**：
   - 研究助手提供主题最新信息的结构化总结，帮助用户轻松理解关键见解。

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化 DuckDuckGoSearchRun 工具
search_tool = DuckDuckGoSearchRun()

# 为总结定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个研究助手。请总结以下信息："),
    ("user", "{search_results}")
])

# 创建一个集成搜索工具和 LLM 的链
chain = (
    {"search_results": lambda x: search_tool.invoke(x["topic"])}
    | prompt
    | model
    | StrOutputParser()
)

# 示例用法
topic = "2023 年可再生能源趋势"
summary = chain.invoke({"topic": topic})
print(f"关于 '{topic}' 的总结：\n{summary}")
```

---

## 结论

`DuckDuckGoSearchRun` 是一个多功能且可靠的工具，可将网络搜索功能集成到您的应用程序中。其注重隐私的设计，结合 LangChain 的强大功能（如重试、回退和可配置替代方案），使其成为开发需要实时信息的智能系统的开发者的绝佳选择。无论是创建聊天机器人、自动化研究任务还是增强数据管道，`DuckDuckGoSearchRun` 提供了处理动态搜索需求的灵活性和鲁棒性。

通过利用此工具，您可以确保应用程序在保持用户隐私承诺的同时，始终掌握最新数据。随着对实时、准确信息的需求不断增长，`DuckDuckGoSearchRun` 作为创新且注重隐私的解决方案的关键推动者而脱颖而出。