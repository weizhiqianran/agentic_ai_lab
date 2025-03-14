# LangChain 中的消息设计：实用示例和用例

## 介绍

消息是 LangChain 中通信的支柱，能够与聊天模型进行无缝交互。无论您是处理基于文本的输入还是多模态输入，LangChain 统一的消息格式都确保了不同聊天模型提供商之间的兼容性。本指南探讨了在 LangChain 中使用消息的实用示例，从基本消息创建到高级功能，如修剪、过滤和合并消息。最后，您将对如何利用消息构建健壮的对话式 AI 应用程序有扎实的理解。

### 消息的属性

消息是聊天模型中通信的基本单元。它们表示聊天模型的输入和输出，以及与对话相关的任何其他上下文或元数据。LangChain 中的每条消息都具有：

1. **角色**: 指定消息的发送者（例如，“用户”、“助手”、“系统”）。
2. **内容**: 消息的实际内容，可以是文本或多模态数据（例如，图像、音频）。
3. **其他元数据**: 可选信息，如消息 ID、名称、令牌使用情况或特定于模型的元数据。

LangChain 的消息格式旨在在不同的聊天模型中保持一致，从而更容易在提供商之间切换或将多个模型集成到单个应用程序中。

### 消息的关键组件

#### 1. **角色**
消息的角色定义了谁在发送它，并帮助聊天模型理解如何响应。主要角色是：

- **系统**: 向模型提供指令或上下文（例如，“你是一个乐于助人的助手。”）。
- **用户**: 表示与模型交互的用户的输入。
- **助手**: 表示模型的响应，可以包括文本或调用工具的请求。
- **工具**: 用于将工具调用的结果传递回模型。

#### 2. **内容**
消息的内容可以是：
- **文本**: 最常见的内容类型。
- **多模态数据**: 表示图像、音频或视频的字典列表（某些模型支持）。

#### 3. **其他元数据**
消息可以包含可选元数据，例如：
- **ID**: 消息的唯一标识符。
- **名称**: 用于区分具有相同角色的实体的名称属性。
- **令牌使用情况**: 关于消息的令牌计数的信息。
- **工具调用**: 模型发出的调用外部工具的请求。


```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```

1. **检索 API 密钥**: 
   - `UserSecretsClient` 用于安全地检索存储在 Kaggle 环境中的 API 密钥。这确保了敏感凭据不会在代码中暴露。
   - `get_secret` 方法使用密钥名称（例如，`"my-anthropic-api-key"` 或 `"my-openai-api-key"`）获取 API 密钥。

2. **初始化 LLM**:
   - **Anthropic**: `ChatAnthropic` 类用于初始化 Claude 模型（例如，`claude-3-5-sonnet-latest`）。 `temperature` 参数控制模型响应的随机性。
   - **OpenAI**: `ChatOpenAI` 类用于初始化 GPT 模型（例如，`gpt-4o-mini`）。 `temperature` 参数设置为 `0` 以获得确定性响应。


```python
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from kaggle_secrets import UserSecretsClient

# Retrieve LLM API Key
# user_secrets = UserSecretsClient()

# Initialize LLM (Anthropic, OpenAI)
#model = ChatAnthropic(temperature=0, model="claude-3-5-sonnet-latest", api_key=user_secrets.get_secret("my-anthropic-api-key"))
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
```


```python

```

---

## **1. 基本消息创建和使用**

### 1.1: 创建和发送消息
此示例演示如何在 LangChain 中创建和发送消息。 消息使用特定角色（`system`、`human`、`ai`）创建，并组合成对话。 然后打印对话以显示每条消息的角色和内容。


```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Create a system message to set the context
system_message = SystemMessage(content="You are a helpful assistant.")

# Create a user message
user_message = HumanMessage(content="What is the capital of France?")

# Create an assistant message (model's response)
assistant_message = AIMessage(content="The capital of France is Paris.")

# Combine messages into a conversation
conversation = [system_message, user_message, assistant_message]

# Print the conversation
for msg in conversation:
    print(f"{msg.type}: {msg.content}")
```

### 1.2: 自动将字符串转换为 HumanMessage
调用聊天模型时，LangChain 会自动将字符串转换为 `HumanMessage`。 这简化了向模型发送用户输入的过程。


```python
from langchain_openai import ChatOpenAI

# Initialize the chat model
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# Invoke the model with a string (automatically converted to HumanMessage)
response = model.invoke("Tell me a joke about programming.")
print(response.content)
```

---

## **2. 使用多模态消息**

### 2.1: 发送多模态输入（文本 + 图像）
某些模型支持多模态输入，例如文本和图像。 此示例显示如何创建多模态消息并将其发送到模型。


```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Initialize the chat model
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# Create a multimodal message with text and an image URL
multimodal_message = HumanMessage(
    content=[
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://www.telegraph.co.uk/content/dam/news/2016/03/31/starling_trans_NvBQzQNjv4BqpJliwavx4coWFCaEkEsb3kvxIt-lGGWCWqwLa_RXJU8.jpg?imwidth=960"}},
    ]
)

# Send the message to a multimodal model
response = model.invoke([multimodal_message])
print(response.content)
```

### 2.2: 处理多模态输出
如果模型返回多模态内容（例如，文本 + 音频），您可以按如下方式处理它：


```python
from langchain_core.messages import AIMessage

# Simulate a multimodal response from the model
multimodal_response = AIMessage(
    content=[
        {"type": "text", "text": "Here is your answer."},
        {"type": "audio_url", "audio_url": "https://example.com/audio.mp3"},
    ]
)

# Extract and process the content
for content_block in multimodal_response.content:
    if content_block["type"] == "text":
        print("Text:", content_block["text"])
    elif content_block["type"] == "audio_url":
        print("Audio URL:", content_block["audio_url"])
```

---

## **3. 管理聊天记录**

### 3.1: 根据令牌计数修剪聊天记录
当聊天记录过长时，您可以修剪它以适应模型的令牌限制。 此示例显示如何在保留对话结构的同时修剪聊天记录。


```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages

# Example chat history
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="What is the population of France?"),
    AIMessage(content="The population of France is approximately 67 million."),
]

# Trim the chat history to the last 50 tokens
trimmed_messages = trim_messages(messages, token_counter=model, max_tokens=50, strategy="last", include_system=True)

# Print the trimmed messages
for msg in trimmed_messages:
    print(f"{msg.type}: {msg.content}")
```

### 3.2: 按类型过滤消息
您可以过滤消息以仅包括特定类型（例如，仅用户消息）。 此示例演示如何按类型过滤消息。


```python
from langchain_core.messages import filter_messages

# Filter to include only HumanMessage (user messages)
filtered_messages = filter_messages(messages, include_types="human")

# Print the filtered messages
for msg in filtered_messages:
    print(f"{msg.type}: {msg.content}")
```

---

## **4. 高级功能**

### 4.1: 合并连续消息
某些模型不支持相同类型的连续消息。 使用 `merge_message_runs` 将它们合并为一条消息。


```python
from langchain_core.messages import merge_message_runs

# Example messages with consecutive HumanMessages
messages = [
    HumanMessage(content="What is the capital of France?"),
    HumanMessage(content="And what is the population?"),
]

# Merge consecutive messages
merged_messages = merge_message_runs(messages)

# Print the merged messages
for msg in merged_messages:
    print(f"{msg.type}: {msg.content}")
```

### 4.2: 流式响应
使用 `AIMessageChunk` 实时流式传输来自模型的响应。 此示例显示如何流式传输模型的响应。


```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Initialize the chat model
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# Stream the model's response
for chunk in model.stream([HumanMessage(content="Tell me a joke about cats that has at least 5 sentences.")]):
    print(chunk.content, end="", flush=True)
```

---

## **5. 工具调用和函数消息**

### 5.1: 使用 ToolMessage 进行工具调用
当模型请求调用工具时，使用 `ToolMessage` 将结果传递回去。 此示例演示如何处理工具调用和响应。


```python
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Initialize the chat model
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# Simulate a tool call request from the model
tool_call_message = AIMessage(
    content="",
    tool_calls=[
        {
            "id": "call_123",  # Unique ID for the tool call
            "name": "get_weather",  # Name of the tool
            "args": {"location": "Paris"},  # Arguments for the tool
        }
    ],
)

# Simulate the tool's response
tool_response = ToolMessage(
    content="The weather in Paris is sunny.",
    tool_call_id=tool_call_message.tool_calls[0]["id"],  # Use the same ID as the tool call
)

# Pass the tool response back to the model
response = model.invoke([tool_call_message, tool_response])
print(response.content)
```

### 5.2: 传统 FunctionMessage (OpenAI)
对于 OpenAI 的传统函数调用 API，请使用 `FunctionMessage`。 此示例显示如何处理传统函数调用。


```python
from langchain_core.messages import FunctionMessage
from langchain_openai import ChatOpenAI

# Initialize the chat model
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# Simulate a function response
function_response = FunctionMessage(
    name="get_weather",
    content='{"temperature": 22, "condition": "sunny"}',
)

# Pass the function response back to the model
response = model.invoke([function_response])
print(response.content)
```

---

## **6. 与聊天模型集成**

### 6.1: 将消息与 ChatOpenAI 结合使用
以下是如何将消息与 OpenAI 的聊天模型结合使用。 此示例演示如何创建对话并获取模型的响应。


```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Initialize the chat model
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# Create a conversation
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Germany?"),
]

# Get the model's response
response = model.invoke(messages)
print(response.content)
```

### 6.2: 将消息与 Anthropic 的 Claude 结合使用
以下是如何将消息与 Anthropic 的 Claude 模型结合使用。 此示例演示如何发送用户消息并获取模型的响应。


```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Initialize the chat model
model = ChatAnthropic(temperature=0, model="claude-3-5-sonnet-latest", api_key=user_secrets.get_secret("my-anthropic-api-key"))

# Create a user message
user_message = HumanMessage(content="Tell me a fun fact about space.")

# Get the model's response
response = model.invoke([user_message])
print(response.content)
```

---

## **7. 将消息与其他组件链接**

### 7.1: 将消息与提示链接
在链中将消息与提示结合起来。 此示例显示如何使用提示模板和聊天模型创建链。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the chat model
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{query}"),
])

# Create a chain
chain = prompt | model

# Invoke the chain
response = chain.invoke({"query": "What is the capital of Italy?"})
print(response.content)
```

### 7.2: 与消息修剪链接
将消息修剪与聊天模型结合起来。 此示例演示如何在将消息传递给模型之前对其进行修剪。


```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

# Initialize the chat model
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))

# Create a trimmer
trimmer = trim_messages(
    token_counter=model,
    max_tokens=50,
    strategy="last",
    include_system=True,
)

# Create a chain
chain = trimmer | model

# Invoke the chain
response = chain.invoke(messages)
print(response.content)
```

## 结论

LangChain 的消息设计为使用聊天模型提供了一个强大而灵活的框架。 从创建简单的对话到处理复杂的多模态输入和管理聊天记录，本指南中的示例演示了 LangChain 消息系统的多功能性。 通过掌握这些技术，您可以构建更高效、更有效的对话式 AI 应用程序，从而确保跨不同模型和用例的兼容性。
