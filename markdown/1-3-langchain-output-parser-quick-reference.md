# **LangChain 输出解析器快速参考：从字符串到结构化数据**

## **简介**

在快速发展的自然语言处理（NLP）领域，使用语言模型（LLM）通常涉及处理复杂且非结构化的输出。为了简化这个过程，LangChain 提供了一套强大的输出解析器，旨在从 LLM 响应中提取、结构化和验证数据。本文探讨了 LangChain 框架中的三个关键解析器：`StrOutputParser`、`PydanticOutputParser` 和 `StructuredOutputParser`。这些工具各有独特的用途，从将 LLM 输出简化为纯字符串，到强制执行结构化数据模式以及处理流式传输或部分结果。无论您是构建聊天机器人、数据提取管道还是实时应用程序，这些解析器都提供了将 LLM 有效集成到您的工作流程中所需的灵活性和稳健性。

### **对比表**

| 特性                     | StrOutputParser          | PydanticOutputParser      | StructuredOutputParser    |
|-----------------------------|--------------------------|---------------------------|---------------------------|
| **主要用例**        | 提取纯字符串   | 解析为 Pydantic 模型 | 解析为结构化格式（例如，字典） |
| **模式强制执行**      | 否                       | 是                       | 是                       |
| **流式传输支持**       | 是                      | 是                       | 是                       |
| **错误处理**          | 基础                    | 强大（带验证）  | 强大（带回退）   |
| **与提示的集成**| 有限                  | 是                       | 是                       |
| **复杂度**              | 低                      | 中等                    | 高                      |


```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```

---

## **1. StrOutputParser**

### **1.1. 使用语言模型的基本用法**
此示例演示如何使用 `StrOutputParser` 将 `ChatOpenAI` 模型的输出解析为字符串。


```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient
from langchain_core.output_parsers import StrOutputParser

# 检索 LLM API 密钥
user_secrets = UserSecretsClient()

# 初始化语言模型
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
#llm = = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://20.243.34.136:2999/v1",
                        openai_api_key="sk-j8r3Pxztstd3wBjF8fEe44E63f69486bAdC2C4562bD1E1F3")

# 创建一个使用 LLM 然后解析输出的链
chain = llm | StrOutputParser()

# 使用提示调用链
response = chain.invoke("给我讲个笑话。")
print(response)
```

### **1.2. 将 `StrOutputParser` 与提示模板一起使用**
此示例展示了如何将 `StrOutputParser` 与 `ChatPromptTemplate` 一起使用来格式化输入并将输出解析为字符串。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("user", "{input}")
])

# 初始化语言模型
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://20.243.34.136:2999/v1",
                        openai_api_key="sk-j8r3Pxztstd3wBjF8fEe44E63f69486bAdC2C4562bD1E1F3")

# 创建一个组合提示模板、LLM 和输出解析器的链
chain = prompt_template | llm | StrOutputParser()

# 使用输入调用链
response = chain.invoke({"input": "法国的首都是什么？"})
print(response)
```

### **1.3. 使用 `StrOutputParser` 进行流式输出**
此示例演示如何使用 `StrOutputParser` 以块的形式解析流式 LLM 的输出。


```python
from langchain_core.output_parsers import StrOutputParser

# 初始化语言模型
# llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=user_secrets.get_secret("my-openai-api-key"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://20.243.34.136:2999/v1",
                        openai_api_key="sk-j8r3Pxztstd3wBjF8fEe44E63f69486bAdC2C4562bD1E1F3")


# 创建一个使用 LLM 然后解析输出的链
chain = llm | StrOutputParser()

# 流式传输输出
for chunk in chain.stream("给我讲一个关于龙的故事。"):
    print(chunk, end="", flush=True)
```

### **1.4. 将 `StrOutputParser` 与重试机制一起使用**
此示例演示如何在包含重试机制的链中使用 `StrOutputParser`。


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 初始化语言模型
# llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=user_secrets.get_secret("my-openai-api-key"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://20.243.34.136:2999/v1",
                        openai_api_key="sk-j8r3Pxztstd3wBjF8fEe44E63f69486bAdC2C4562bD1E1F3")


# 定义一个可能失败的函数
def unreliable_function(input: str) -> str:
    print(f"尝试处理输入：{input}")
    if "fail" in input:
        raise ValueError("输入包含 'fail'。处理已中止。")
    return input.upper()

# 创建一个将重试机制应用于整个链的链
chain = (
    RunnableLambda(unreliable_function)
    | llm
    | StrOutputParser()
).with_retry(stop_after_attempt=3)  # 最多重试整个链 3 次

# 使用将失败的输入调用链
try:
    response = chain.invoke("这将失败。")
    print("响应：", response)
except Exception as e:
    print(f"重试后出错：{e}")
```

### **1.5. 将 `StrOutputParser` 与回退一起使用**
此示例展示了如何在包含回退机制的链中使用 `StrOutputParser`。


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 初始化语言模型
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=user_secrets.get_secret("my-openai-api-key"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://20.243.34.136:2999/v1",
                        openai_api_key="sk-j8r3Pxztstd3wBjF8fEe44E63f69486bAdC2C4562bD1E1F3")


# 定义一个可能失败的函数
def unreliable_function(input: str) -> str:
    if "fail" in input:
        raise ValueError("处理输入失败。")
    return input.upper()

# 定义一个回退函数
def fallback_function(input: str) -> str:
    return "回退响应。"

# 创建一个带有回退的链
chain = (
    RunnableLambda(unreliable_function)
    .with_fallbacks([RunnableLambda(fallback_function)])
    | llm
    | StrOutputParser()
)

# 调用链
response = chain.invoke("这将失败。")
print(response)
```

---

## **2. PydanticOutputParser**

### **2.1. 使用 Pydantic 模型的基本用法**
此示例演示如何使用 `PydanticOutputParser` 将语言模型的输出解析为 Pydantic 模型。


```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

# 定义一个 Pydantic 模型
class Person(BaseModel):
    name: str
    age: int
    email: str

# 使用 Person 模型创建 PydanticOutputParser 的实例
parser = PydanticOutputParser(pydantic_object=Person)

# 来自语言模型的示例输出
model_output = '{"name": "John Doe", "age": 30, "email": "john.doe@example.com"}'

# 将输出解析为 Person 模型
parsed_output = parser.parse(model_output)

print(parsed_output)
```

### **2.2. 处理部分输出**
此示例展示了如何使用 `PydanticOutputParser` 处理部分 JSON 输出。


```python
from langchain_core.outputs import Generation

# 来自语言模型的示例部分输出
partial_output = '{"name": "John Doe", "age": 30}'

# 将输出包装在 Generation 对象中
generation = Generation(text=partial_output)

# 使用 `partial=True` 使用 `parse_result` 解析部分输出
parsed_output = parser.parse_result([generation], partial=True)

print(parsed_output)
# 输出：None（因为 `partial=True` 允许缺少字段）
```

### **2.3. 与 LangChain Runnable 一起使用**
此示例展示了如何将 `PydanticOutputParser` 与 LangChain 的 `Runnable` 接口集成。


```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义一个简单的提示模板
prompt = PromptTemplate.from_template("告诉我关于 {name} 的信息")

# 定义一个简单的链
chain = (
    prompt 
    | RunnableLambda(lambda x: f'{{"name": "{x.text}", "age": 30, "email": "john.doe@example.com"}}')
    | parser
)

# 调用链
result = chain.invoke({"name": "John Doe"})
print(result)
```

### **2.4. 处理错误**
此示例演示如何在使用 `PydanticOutputParser` 解析无效输出时处理错误。


```python
# 示例无效输出
invalid_output = '{"name": "John Doe", "age": "thirty", "email": "john.doe@example.com"}'

try:
    parsed_output = parser.parse(invalid_output)
except Exception as e:
    print(f"错误：{e}")
```

### **2.5. 与 LangChain 的 `Runnable` 接口一起使用**
此示例展示了如何将 `PydanticOutputParser` 与 LangChain 的 `Runnable` 接口一起使用来创建更复杂的链。


```python
from langchain_core.runnables import RunnableLambda

# 定义一个返回 JSON 字符串的简单函数
def generate_json(input: str) -> str:
    return f'{{"name": "{input}", "age": 30, "email": "john.doe@example.com"}}'

# 创建一个使用该函数和解析器的链
chain = (
    RunnableLambda(generate_json)
    | parser
)

# 调用链
result = chain.invoke("John Doe")
print(result)
```

### **2.6. 解析多个输出**
此示例演示如何使用 `PydanticOutputParser` 并行解析多个输出。


```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

# 定义一个 Pydantic 模型
class Person(BaseModel):
    name: str
    age: int
    email: str

# 使用 Person 模型创建 PydanticOutputParser 的实例
parser = PydanticOutputParser(pydantic_object=Person)

# 示例多个输出
outputs = [
    '{"name": "John Doe", "age": 30, "email": "john.doe@example.com"}',
    '{"name": "Jane Doe", "age": 25, "email": "jane.doe@example.com"}'
]

# 使用同步 `batch` 方法并行解析输出
parsed_outputs = parser.batch(outputs)
print(parsed_outputs)
```

### **2.7. 与 LangChain 的 `Runnable` 和 `PromptTemplate` 一起使用**
此示例展示了如何将 `PydanticOutputParser` 与 LangChain 的 `PromptTemplate` 结合使用来创建更复杂的链。


```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 定义一个提示模板
prompt_template = PromptTemplate.from_template("为名为 {name} 的人生成一个 JSON 对象")

# 创建一个使用提示模板、语言模型和解析器的链
chain = (
    prompt_template
    | RunnableLambda(lambda x: f'{{"name": "{x.text}", "age": 30, "email": "john.doe@example.com"}}')
    | parser
)

# 调用链
result = chain.invoke({"name": "John Doe"})
print(result)
```

---

## **3. StructuredOutputParser**

### **3.1. 基本用法**
此示例演示如何使用 `StructuredOutputParser` 将语言模型的输出解析为结构化格式。


```python
import warnings
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

# 抑制特定的 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine 'Runnable.abatch' was never awaited")

# 定义输出的模式
response_schemas = [
    ResponseSchema(name="name", description="人的姓名"),
    ResponseSchema(name="age", description="人的年龄"),
    ResponseSchema(name="favorite_color", description="人最喜欢的颜色"),
]

# 创建解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 来自语言模型的示例输出
model_output = """
```json
{
    "name": "John Doe",
    "age": 30,
    "favorite_color": "blue"
}
```
"""

# 解析输出
parsed_output = parser.parse(model_output)
print(parsed_output)
```

### **3.2. 使用 `get_format_instructions`**
此示例展示了如何使用 `get_format_instructions` 来指导语言模型以特定格式生成输出。


```python
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

# 定义输出的模式
response_schemas = [
    ResponseSchema(name="name", description="人的姓名"),
    ResponseSchema(name="age", description="人的年龄"),
    ResponseSchema(name="favorite_color", description="人最喜欢的颜色"),
]

# 创建解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式说明
format_instructions = parser.get_format_instructions()
print(format_instructions)
```

### **3.3. 解析部分结果**
此示例演示如何使用 `StructuredOutputParser` 解析部分结果。


```python
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain.schema import Generation

# 定义输出的模式
response_schemas = [
    ResponseSchema(name="name", description="人的姓名"),
    ResponseSchema(name="age", description="人的年龄", optional=True),  # 标记为可选
    ResponseSchema(name="favorite_color", description="人最喜欢的颜色", optional=True),  # 标记为可选
]

# 创建解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 来自语言模型的示例输出（部分结果）
model_output = [
    Generation(text='```json\n{"name": "John Doe", "age": 30}\n```'),                  # 缺少 favorite_color
    Generation(text='```json\n{"name": "Jane Doe", "favorite_color": "green"}\n```'),  # 缺少 age
]

# 解析结果
try:
    parsed_output = parser.parse_result(model_output)
    print("解析的输出：", parsed_output)
except Exception as e:
    print("解析输出时出错：", e)
```

### **3.4. 使用 `parse_with_prompt`**
此示例展示了如何在考虑输入提示的同时使用 `parse_with_prompt` 解析输出。


```python
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# 定义输出的模式
response_schemas = [
    ResponseSchema(name="name", description="人的姓名"),
    ResponseSchema(name="age", description="人的年龄"),
    ResponseSchema(name="favorite_color", description="人最喜欢的颜色"),
]

# 创建解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 来自语言模型的示例输出
model_output = """
```json
{
    "name": "John Doe",
    "age": 30,
    "favorite_color": "blue"
}
```
"""

# 创建一个提示模板
prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("告诉我关于一个人的信息。")
])

# 使用模板生成提示值
prompt_value = prompt_template.format_prompt()

# 使用提示解析输出
parsed_output = parser.parse_with_prompt(model_output, prompt_value)
print(parsed_output)
```

### **3.5. 流式输出**
此示例演示如何使用 `StructuredOutputParser` 解析来自语言模型的流式输出。


```python
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

# 定义输出的模式
response_schemas = [
    ResponseSchema(name="name", description="人的姓名"),
    ResponseSchema(name="age", description="人的年龄"),
    ResponseSchema(name="favorite_color", description="人最喜欢的颜色"),
]

# 创建解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 来自语言模型的示例输出
model_output = '```json\n{"name": "John Doe", "age": 30, "favorite_color": "blue"}\n```'

# 使用 stream 方法解析输出
for chunk in parser.stream(model_output):
    print(chunk)
```

### **3.6. 使用 `with_fallbacks` 处理错误**
此示例展示了如何使用 `with_fallbacks` 来指定当主解析器失败时的回退解析器。


```python
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

# 定义输出的模式
response_schemas = [
    ResponseSchema(name="name", description="人的姓名"),
    ResponseSchema(name="age", description="人的年龄"),
    ResponseSchema(name="favorite_color", description="人最喜欢的颜色"),
]

# 创建主解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 定义一个自定义回退解析器
def custom_fallback_parser(text: str) -> dict:
    """处理非结构化文本的回退解析器。"""
    print("custom_fallback_parser() 被调用")
    return {"name": "未知", "age": 0, "favorite_color": "未知"}

# 将自定义回退解析器包装在 RunnableLambda 中
fallback_runnable = RunnableLambda(custom_fallback_parser)

# 将解析器与回退组合
parser_with_fallbacks = parser.with_fallbacks([fallback_runnable])

# 来自语言模型的示例输出（无效格式）
model_output = "这不是有效的 JSON 输出。"

# 使用回退解析输出
try:
    parsed_output = parser_with_fallbacks.parse(model_output)
    print("解析的输出：", parsed_output)
except OutputParserException as e:
    print("解析输出时出错：", e)
    # 提供一个回退值
    parsed_output = custom_fallback_parser(model_output)
    print("回退输出：", parsed_output)
```

## **结论**

LangChain 框架的输出解析器——`StrOutputParser`、`PydanticOutputParser` 和 `StructuredOutputParser`——为开发者提供了多功能的工具来处理语言模型的各种输出。 `StrOutputParser` 擅长提取简单的字符串输出，使其成为直接文本提取任务的理想选择。 `PydanticOutputParser` 为 LLM 输出带来了结构和验证，确保数据符合预定义的模式。 同时，`StructuredOutputParser` 提供了一种灵活的、基于模式的方法，用于将复杂输出解析为字典或其他结构化格式。 总之，这些解析器使 LLM 能够无缝集成到应用程序中，从实时流式传输到强大的错误处理。 通过利用这些工具，开发者可以构建更可靠、高效和可维护的 NLP 系统。
