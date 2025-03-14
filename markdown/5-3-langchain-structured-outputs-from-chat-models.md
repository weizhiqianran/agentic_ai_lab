# **LangChain 从聊天模型中获取结构化输出**

## 引言

随着像 ChatGPT 和 Claude 这样的大型语言模型（LLMs）越来越多地融入各种应用，对精确且可靠的数据交换需求也在增长。结构化输出通过使聊天模型以特定、可预测的格式生成响应，提供了一种解决方案。本文探讨了结构化输出的概念，深入研究了 LangChain 的 `.with_structured_output()` 方法，并提供了实用示例来说明其实现方式和优势。

### 什么是结构化输出？

结构化输出是指聊天模型生成的响应，遵循预定义的数据格式，例如 JSON、Pydantic 模型或 TypedDicts。与非结构化文本不同，结构化输出确保了一致性，便于与其他系统、API 或需要特定数据格式的工作流程无缝集成。

### 为什么结构化输出重要？

- **可预测性**：下游系统可以可靠地解析和使用数据，避免歧义。
- **效率**：自动化数据格式化和验证，减少手动干预的需求。
- **健壮性**：通过验证机制强制数据完整性，最大限度减少错误。

---

## `.with_structured_output()` 方法

LangChain 是一个用于构建 LLM 应用程序的强大框架，其 `.with_structured_output()` 方法通过启用聊天模型的结构化响应增强了交互性。这一方法对于希望将 LLM 集成到需要特定数据格式的系统中的开发者至关重要。

### 主要特性

1. **声明式数据模型**：
   - 允许使用 Pydantic 模型或 JSON 模式定义所需的输出结构。
   - **示例**：
     ```python
     from pydantic import BaseModel, Field, EmailStr
     
     class Person(BaseModel):
         name: str = Field(description="人的全名")
         age: int = Field(description="人的年龄（以年为单位）")
     ```

2. **自定义输出解析**：
   - 将原始 LLM 输出转换为匹配定义的模式。
   - 能够处理复杂的数据转换，例如将逗号分隔的文本转换为列表。

3. **错误处理和验证**：
   - 自动根据指定模式验证输出。
   - 实现错误纠正循环，通过备用机制修复无效输出。

4. **与 OpenAI 函数调用集成**：
   - 利用 OpenAI 的函数调用 API，确保输出采用 JSON 等结构化格式。
   - **示例**：
     ```python
     from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
     chain = (
         prompt
         | model.bind(function_call={"name": "joke"}, functions=functions)
         | JsonOutputFunctionsParser()
     )
     ```

5. **并行和模块化处理**：
   - 与其他链或组件无缝集成，用于多步骤处理。
   - **示例**：
     ```python
     retrieval_chain = (
         {"context": retriever, "question": RunnablePassthrough()}
         | prompt
         | model
         | StrOutputParser()
     )
     ```

---

## 方法比较

| **特性**                        | **Pydantic**                                                  | **TypedDict**                                                | **JSON Schema**                                              |
|---------------------------------|---------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| **类型强制**                    | 强类型强制与验证                                              | 有限的类型强制                                              | 基于模式的验证                                              |
| **易用性**                      | 需要定义类；更冗长                                            | 对于类似字典的结构更简单                                    | 需要理解 JSON Schema 语法                                   |
| **灵活性**                      | 对于复杂数据模型高度灵活                                      | 适用于较简单的数据结构                                      | 对于跨系统互操作性极佳                                      |
| **集成性**                      | 与 Python 应用程序和框架无缝集成                              | 易于与 Python 代码库集成                                    | 语言无关，适合 API 集成                                     |
| **验证能力**                    | 广泛的验证选项                                                | 基本的类型注解                                              | 全面的验证规则                                              |
| **用例**                        | 企业应用、数据管道、API                                        | 轻量级应用、快速原型                                        | API 规范、跨语言数据交换                                    |
| **示例工具/框架**               | FastAPI、Django、LangChain                                    | LangChain                                                  | OpenAPI、LangChain                                          |

### 选择正确的方法

- **使用 Pydantic**：当需要在 Python 中心生态系统中进行健壮的数据验证时。
- **使用 TypedDict**：在简单场景中，轻量级的类型注解就足够时。
- **使用 JSON Schema**：当互操作性和跨语言支持至关重要时。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient
from langchain_core.output_parsers import StrOutputParser

# 获取 LLM API 密钥
user_secrets = UserSecretsClient()

# 初始化语言模型
#model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
#model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

---

## 1. Pydantic

### 1.1 使用 Pydantic 的标准字段
Pydantic 是一个使用 Python 类型注解进行数据验证和设置管理的库，广泛用于定义具有类型强制的模型。

```python
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 定义笑话模式的 Pydantic 类
class Joke(BaseModel):
    """要讲给用户的笑话。"""
    setup: str = Field(description="笑话的铺垫")
    punchline: str = Field(description="笑话的妙语")
    rating: Optional[int] = Field(default=None, description="笑话的搞笑程度，从 1 到 10")

# 初始化 ChatOpenAI 模型
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 Pydantic 类返回结构化输出
structured_llm = model.with_structured_output(Joke)

# 生成一个关于猫的笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")

# 使用点号访问字段
print("铺垫:", result.setup)
print("妙语:", result.punchline)
print("评分:", result.rating)
```

### 1.2 使用 Pydantic 模型的 Literal 字段
Pydantic 是一个利用 Python 类型注解进行数据验证和设置管理的库，提供了一种定义具有类型检查的数据结构的方式。

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

# 定义结构化输出的 Pydantic 模型
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    unit: Literal["Celsius", "Fahrenheit"]

# 初始化 ChatOpenAI 模型
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 Pydantic 模型输出结构化数据
structured_model = model.with_structured_output(WeatherResponse)

# 通过调用 RunnableSequence 生成结构化响应
response = structured_model.invoke("巴黎的天气如何？")
print(f"地点     : {response.location}")
print(f"温度     : {response.temperature}")
print(f"单位     : {response.unit}")
```

### 1.3 使用 Pydantic 和 Union
可以通过在模式中使用 Union 类型配置模型，以处理多种类型的结构化输出。这允许模型根据输入或上下文在不同的输出格式之间进行选择。

```python
from typing import Union, Optional
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

# 为不同响应类型定义 Pydantic 类
class Joke(BaseModel):
    """要讲给用户的笑话。"""
    setup: str = Field(description="笑话的铺垫")
    punchline: str = Field(description="笑话的妙语")
    rating: Optional[int] = Field(default=None, description="笑话的搞笑程度，从 1 到 10")

class Fact(BaseModel):
    """要讲给用户的事实。"""
    topic: str = Field(description="事实的主题")
    fact: str = Field(description="事实本身")
    source: Optional[str] = Field(default=None, description="事实的来源")

class FinalResponse(BaseModel):
    """最终响应，可以是笑话或事实。"""
    response: Union[Joke, Fact]

# 初始化 ChatAnthropic 模型
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 Union 模式返回结构化输出
structured_llm = model.with_structured_output(FinalResponse)

# 生成一个笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")
print(result)

# 访问嵌套响应的字段
if isinstance(result.response, Joke):
    print("铺垫:", result.response.setup)
    print("妙语:", result.response.punchline)
    print("评分:", result.response.rating)

# 生成一个事实
result = structured_llm.invoke("给我讲一个关于月球的事实")
print(result)

if isinstance(result.response, Fact):
    print("主题:", result.response.topic)
    print("事实:", result.response.fact)
    print("来源:", result.response.source)
```

---

## 2. TypedDict

### 2.1 使用 TypedDict 的标准字段
TypedDict 允许对字典进行类型注解，使其适合在无需完整模型开销的情况下定义结构化输出。

```python
from typing import Optional
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI

# 定义笑话模式的 TypedDict
class Joke(TypedDict):
    """要讲给用户的笑话。"""
    setup: Annotated[str, ..., "笑话的铺垫"]
    punchline: Annotated[str, ..., "笑话的妙语"]
    rating: Annotated[Optional[int], None, "笑话的搞笑程度，从 1 到 10"]

# 初始化 ChatOpenAI 模型
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 TypedDict 返回结构化输出
structured_llm = model.with_structured_output(Joke)

# 生成一个关于猫的笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")
print(result)

# 使用键访问字段
print("铺垫:", result["setup"])
print("妙语:", result["punchline"])
print("评分:", result["rating"])
```

### 2.2 使用 TypedDict 的 Literal 字段
TypedDict 允许定义类似字典的数据结构并带有类型注解，增强了类型安全性和清晰度。

```python
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Literal

# 定义结构化输出的 TypedDict
class ActionResponse(TypedDict):
    action: Literal["create", "update", "delete"]
    target: str
    details: str

# 初始化 ChatAnthropic 模型
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 TypedDict 输出结构化数据
structured_model = model.with_structured_output(ActionResponse)

# 通过调用 RunnableSequence 生成结构化响应
response = structured_model.invoke("创建一个名为 John Doe 的新用户。")
print(f"动作 : {response['action']}")
print(f"目标 : {response['target']}")
print(f"细节 : {response['details']}")
```

### 2.3 使用 TypedDict 和 Union
类似地，可以使用 TypedDict 和 Union 来处理多种响应类型，而无需使用完整的模型。

```python
from typing import Union, Optional
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI

# 为不同响应类型定义 TypedDict 类
class Joke(TypedDict):
    """要讲给用户的笑话。"""
    setup: Annotated[str, ..., "笑话的铺垫"]
    punchline: Annotated[str, ..., "笑话的妙语"]
    rating: Annotated[Optional[int], None, "笑话的搞笑程度，从 1 到 10"]

class Fact(TypedDict):
    """要讲给用户的事实。"""
    topic: Annotated[str, ..., "事实的主题"]
    fact: Annotated[str, ..., "事实本身"]
    source: Annotated[Optional[str], None, "事实的来源"]

class FinalResponse(TypedDict):
    """最终响应，可以是笑话或事实。"""
    response: Union[Joke, Fact]

# 初始化 ChatOpenAI 模型
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 Union 模式返回结构化输出
structured_llm = model.with_structured_output(FinalResponse)

# 生成一个笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")
print(result)

# 访问嵌套响应的字段
if isinstance(result["response"], dict):  # 检查响应是否为字典（TypedDict）
    if "setup" in result["response"]:  # 检查是否为笑话
        print("铺垫:", result["response"]["setup"])
        print("妙语:", result["response"]["punchline"])
        print("评分:", result["response"]["rating"])

# 生成一个事实
result = structured_llm.invoke("给我讲一个关于月球的事实")
print(result)

# 访问嵌套响应的字段
if isinstance(result["response"], dict):  # 检查响应是否为字典（TypedDict）
    if "topic" in result["response"]:  # 检查是否为事实
        print("主题:", result["response"]["topic"])
        print("事实:", result["response"]["fact"])
        print("来源:", result["response"]["source"])
```

---

## 3. JSON Schema

### 3.1 使用 JSON Schema 的标准字段
JSON Schema 提供了一种描述 JSON 数据结构和验证约束的方法，特别适用于跨系统互操作性。

```python
from langchain_anthropic import ChatAnthropic

# 定义笑话的 JSON Schema
json_schema = {
    "title": "joke",
    "description": "要讲给用户的笑话。",
    "type": "object",
    "properties": {
        "setup": {"type": "string", "description": "笑话的铺垫"},
        "punchline": {"type": "string", "description": "笑话的妙语"},
        "rating": {"type": "integer", "description": "笑话的搞笑程度，从 1 到 10", "default": None},
    },
    "required": ["setup", "punchline"],
}

# 初始化 ChatAnthropic 模型
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 JSON Schema 返回结构化输出
structured_llm = model.with_structured_output(json_schema)

# 生成一个关于猫的笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")
print(result)

# 使用键访问字段
print("铺垫:", result["setup"])
print("妙语:", result["punchline"])
print("评分:", result["rating"])
```

### 3.2 使用 JSON Schema 和 Union
JSON Schema 还通过 `oneOf` 关键字支持 Union，使模型能够返回多个预定义模式之一。

```python
from langchain_anthropic import ChatAnthropic

# 为不同响应类型定义 JSON Schema
joke_schema = {
    "title": "joke",
    "description": "要讲给用户的笑话。",
    "type": "object",
    "properties": {
        "setup": {"type": "string", "description": "笑话的铺垫"},
        "punchline": {"type": "string", "description": "笑话的妙语"},
        "rating": {"type": "integer", "description": "笑话的搞笑程度，从 1 到 10", "default": None},
    },
    "required": ["setup", "punchline"],
}

fact_schema = {
    "title": "fact",
    "description": "要讲给用户的事实。",
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "事实的主题"},
        "fact": {"type": "string", "description": "事实本身"},
        "source": {"type": "string", "description": "事实的来源", "default": None},
    },
    "required": ["topic", "fact"],
}

final_schema = {
    "title": "final_response",
    "description": "最终响应，可以是笑话或事实。",
    "type": "object",
    "properties": {
        "response": {
            "oneOf": [joke_schema, fact_schema],
        },
    },
    "required": ["response"],
}

# 初始化 ChatAnthropic 模型
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 JSON Schema 返回结构化输出
structured_llm = model.with_structured_output(final_schema)

# 生成一个笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")
print(result)

# 生成一个事实
result = structured_llm.invoke("给我讲一个关于月球的事实")
print(result)
```

---

## 4. 高级技术

### 4.1 在多个模式之间选择
在预期不同类型响应的场景中，定义多个模式并允许模型在其中选择，可以确保灵活性和适应性。

```python
from typing import Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 为不同响应类型定义 Pydantic 类
class Joke(BaseModel):
    """要讲给用户的笑话。"""
    setup: str = Field(description="笑话的铺垫")
    punchline: str = Field(description="笑话的妙语")
    rating: Optional[int] = Field(default=None, description="笑话的搞笑程度，从 1 到 10")

class ConversationalResponse(BaseModel):
    """以对话方式响应。友善且乐于助人。"""
    response: str = Field(description="对用户查询的对话式响应")

class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]

# 初始化 ChatOpenAI 模型
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 Union 模式返回结构化输出
structured_llm = model.with_structured_output(FinalResponse)

# 生成一个关于猫的笑话
result = structured_llm.invoke("给我讲一个关于猫的笑话")
print(result)

# 生成一个对话式响应
result = structured_llm.invoke("你今天过得怎么样？")
print(result)
```

### 4.2 流式传输结构化输出
流式传输允许在生成过程中逐步传递结构化数据，这对于大型响应或实时应用非常有益。

```python
from typing_extensions import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic

# 定义笑话模式的 TypedDict
class Joke(TypedDict):
    """要讲给用户的笑话。"""
    setup: Annotated[str, ..., "笑话的铺垫"]
    punchline: Annotated[str, ..., "笑话的妙语"]
    rating: Annotated[Optional[int], None, "笑话的搞笑程度，从 1 到 10"]

# 初始化 ChatAnthropic 模型
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以使用 TypedDict 返回结构化输出
structured_llm = model.with_structured_output(Joke)

# 流式输出一个关于猫的笑话
for chunk in structured_llm.stream("给我讲一个关于猫的笑话"):
    print(chunk)
```

### 4.3 使用结构化输出的少样本提示
少样本提示涉及为模型提供示例以指导其响应。当与结构化输出结合使用时，可以增强模型生成一致且准确数据结构的能力。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 定义带有少样本示例的系统消息
system = """你是一个幽默的喜剧演员。你的专长是敲门笑话。\
返回一个笑话，包括铺垫（对“谁在那儿？”的回答）和最终妙语（对“<铺垫>是谁？”的回答）。

以下是一些笑话示例：

example_user: 给我讲一个关于飞机的笑话
example_assistant: {{"setup": "为什么飞机从不累？", "punchline": "因为它们有休息翼！", "rating": 2}}

example_user: 再给我讲一个关于飞机的笑话
example_assistant: {{"setup": "货运", "punchline": "货运‘嗡嗡’，但飞机‘嗖嗖’！", "rating": 10}}

example_user: 现在讲一个关于毛毛虫的
example_assistant: {{"setup": "毛毛虫", "punchline": "毛毛虫真慢，但看我变成蝴蝶抢风头！", "rating": 5}}"""

# 创建 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

# 初始化 ChatOpenAI 模型
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="请输入地址",
                        openai_api_key="请输入密钥")

# 配置模型以返回结构化输出
structured_llm = model.with_structured_output(Joke)

# 组合提示和结构化 LLM
few_shot_structured_llm = prompt | structured_llm

# 生成一个关于啄木鸟的笑话
result = few_shot_structured_llm.invoke("啄木鸟有什么好笑的？")
print(result)
```

## 结论

结构化输出显著增强了与大型语言模型交互的可靠性和效率。通过强制执行预定义的数据格式，开发者可以确保与各种系统的无缝集成，减少歧义，并自动化数据验证过程。LangChain 的 `.with_structured_output()` 方法提供了一个多功能且强大的工具集，支持使用 Pydantic、TypedDict 或 JSON Schema 实现结构化输出，满足广泛的应用和用例。

无论您是构建企业级应用程序、API 还是交互式系统，利用结构化输出都可以带来更可预测和可维护的解决方案。本文展示的实用示例证明了将结构化输出集成到您的工作流程中的简便性和灵活性，为更健壮和可扩展的 AI 驱动应用程序铺平了道路。