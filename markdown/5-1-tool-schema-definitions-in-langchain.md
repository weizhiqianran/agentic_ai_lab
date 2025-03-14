# LangChain 中的工具 Schema 定义：`Pydantic` 与 `TypedDict`

## 引言

在人工智能和机器学习迅速发展的背景下，创建健壮且可靠的工具至关重要。LangChain 是一个用于构建语言模型应用程序的多功能框架，它为开发者提供了定义结构化工具 Schema 的能力，以确保输入经过验证并符合特定格式。定义这些 Schema 的两种主要方法是使用 **Pydantic 类** 和 **TypedDict**。这两种方法都提供了强制输入验证的机制，但在语法、灵活性和使用场景上有所不同。

本文将深入探讨这两种方法，并通过详细示例说明它们在 LangChain 中的实现方式。阅读完本指南后，您将清楚了解何时使用 Pydantic 类、何时使用 TypedDict，从而在 LangChain 应用程序中构建更可靠且易维护的工具。

### 对比：`Pydantic 类` vs. `TypedDict`

| 特性                      | **Pydantic 类**                                                                 | **TypedDict**                                                     |
|---------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------|
| **验证**                  | 开箱即用提供强大的数据验证和错误处理。                                          | 仅限于类型注解，无内置验证功能。                                  |
| **默认值**                | 支持默认值和复杂的字段配置。                                                    | 本身不支持默认值，需手动处理。                                    |
| **类型强制**              | 严格执行类型，类型不匹配时会抛出错误。                                          | 使用类型提示，但在运行时不强制执行。                              |
| **复杂数据结构**          | 轻松处理嵌套和复杂数据结构。                                                    | 可定义嵌套结构，但缺乏高级验证。                                  |
| **性能**                  | 由于验证过程，存在轻微性能开销。                                                | 依赖类型提示，轻量且开销极小。                                    |
| **易用性**                | 更冗长，但提供全面的数据管理功能。                                              | 更简单直观，适合定义基本 Schema。                                 |
| **扩展性**                | 支持自定义验证器和方法，扩展性强。                                              | 扩展性有限，主要用于静态类型定义。                                |
| **工具集成**              | 与支持 Pydantic 模型验证的工具无缝集成。                                        | 适用于仅需简单类型定义的场景。                                    |

---

## 准备工作

### 安装所需库
本节将安装使用 LangChain、OpenAI 嵌入、Anthropic 模型及其他实用工具所需的 Python 库，包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型和 API 的集成。
- `langchain-anthropic`：实现与 Anthropic 模型和 API 的集成。
- `langchain_community`：包含 LangChain 社区贡献的模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU pydantic[email]
```

### 初始化 OpenAI 和 Anthropic 聊天模型
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI 和 Anthropic 的 API 密钥，并初始化它们的聊天模型。使用 `ChatOpenAI` 和 `ChatAnthropic` 类创建这些模型的实例，可用于文本生成和对话 AI 等自然语言处理任务。

**关键步骤：**
1. **获取 API 密钥**：通过 Kaggle 的 `UserSecretsClient` 安全检索 OpenAI 和 Anthropic 的 API 密钥。
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
#model = ChatAnthropic(model="claude-3-5-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

```python
import json

def pretty_print(aimessage):
    """
    将 AIMessage 对象转换为 JSON 并以缩进格式美化打印。
    
    参数：
        aimessage: 要美化打印的 AIMessage 对象。
    """
    # 将 AIMessage 对象转换为字典
    aimessage_dict = {
        "content": aimessage.content,
        "additional_kwargs": aimessage.additional_kwargs,
        "response_metadata": aimessage.response_metadata,
        "id": aimessage.id,
        "tool_calls": aimessage.tool_calls,
        "usage_metadata": aimessage.usage_metadata,
    }
    
    # 将字典转换为带缩进的 JSON 格式字符串
    pretty_json = json.dumps(aimessage_dict, indent=4, ensure_ascii=False)
    
    # 打印美化后的 JSON
    print(pretty_json)
```

---

## 使用 Pydantic 类定义工具 Schema

Pydantic 是一个强大的数据验证和设置管理库，利用 Python 类型注解。在 LangChain 中使用 Pydantic 类定义工具 Schema，可以为工具创建结构化输入，确保输入经过验证并符合特定格式。以下是一些使用 Pydantic 类定义工具 Schema 的示例：

### 示例 1：简单搜索工具
本示例展示如何使用 Pydantic 类为搜索工具定义输入 Schema。该工具接受搜索查询和结果数量限制。Pydantic 的验证功能确保输入类型和格式正确，然后再执行搜索函数。

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 使用 Pydantic 定义输入 Schema
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")
    limit: int = Field(description="返回的最大结果数", default=10)

# 定义将包装为工具的函数
def search(query: str, limit: int) -> str:
    """根据查询和限制搜索信息。"""
    return f"正在搜索：{query}，限制：{limit}"

# 创建 StructuredTool
search_tool = StructuredTool.from_function(
    func=search,
    name="search",
    description="根据查询和限制搜索信息。",
    args_schema=SearchInput
)

# 不使用 LLM 调用工具
search_result = search_tool.invoke({"query": "LangChain", "limit": 5})
print(search_result)  # 输出：正在搜索：LangChain，限制：5
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([search_tool])
ai_msg = model_with_tools.invoke("搜索 LangChain，结果限制为 5。")

pretty_print(ai_msg)
```

### 示例 2：计算器工具
本示例中，使用 Pydantic 定义计算器工具的输入 Schema。该工具接受两个数值和一个操作（加、减、乘、除）。Pydantic 确保输入是有效数字，且操作是指定字面值之一，从而防止无效操作。

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal

# 使用 Pydantic 定义输入 Schema
class CalculatorInput(BaseModel):
    num1: float = Field(description="第一个数字")
    num2: float = Field(description="第二个数字")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(description="要执行的操作")

# 定义将包装为工具的函数
def calculate(num1: float, num2: float, operation: str) -> float:
    """根据提供的数字和操作执行计算。"""
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    else:
        raise ValueError("无效操作")

# 创建 StructuredTool
calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="执行基本算术运算。",
    args_schema=CalculatorInput
)

# 不使用 LLM 调用工具
calc_result = calculator_tool.invoke({"num1": 10, "num2": 5, "operation": "add"})
print(calc_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([calculator_tool])
ai_msg = model_with_tools.invoke("将 10 和 5 相加。")
pretty_print(ai_msg)
```

### 示例 3：天气预报工具
本示例展示如何使用 Pydantic 构建天气预报工具。该工具需要位置和特定日期以提供天气预报。Pydantic 将位置验证为字符串，日期验证为正确的日期对象，确保预测函数的输入数据准确可靠。

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from datetime import date

# 使用 Pydantic 定义输入 Schema
class WeatherInput(BaseModel):
    location: str = Field(description="获取天气预报的位置")
    forecast_date: date = Field(description="天气预报的日期")

# 定义将包装为工具的函数
def get_weather(location: str, forecast_date: date) -> str:
    """获取特定位置和日期的天气预报。"""
    # 在实际场景中，此函数将调用外部 API
    return f"{location} 在 {forecast_date} 的天气预报：晴天"

# 创建 StructuredTool
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="获取特定位置和日期的天气预报。",
    args_schema=WeatherInput
)

# 不使用 LLM 调用工具
weather_result = weather_tool.invoke({"location": "纽约", "forecast_date": date(2023, 10, 1)})
print(weather_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([weather_tool])
ai_msg = model_with_tools.invoke("获取纽约 2023 年 10 月 1 日的天气预报。")
pretty_print(ai_msg)
```

### 示例 4：发送电子邮件工具
在此示例中，使用 Pydantic 定义发送电子邮件工具，确保提供所有必要字段并格式正确。该工具需要收件人的电子邮件地址、主题和邮件正文。Pydantic 的 `EmailStr` 类型验证收件人邮件地址，提高邮件发送过程的可靠性。

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field, EmailStr

# 使用 Pydantic 定义输入 Schema
class EmailInput(BaseModel):
    recipient: EmailStr = Field(description="收件人的电子邮件地址")
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件正文")

# 定义将包装为工具的函数
def send_email(recipient: str, subject: str, body: str) -> str:
    """向指定收件人发送电子邮件。"""
    # 在实际场景中，此函数将调用邮件发送服务
    return f"邮件已发送至 {recipient}，主题：{subject}"

# 创建 StructuredTool
email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="向指定收件人发送电子邮件。",
    args_schema=EmailInput
)

# 不使用 LLM 调用工具
email_result = email_tool.invoke({"recipient": "user@example.com", "subject": "你好", "body": "这是一封测试邮件。"})
print(email_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([email_tool])
ai_msg = model_with_tools.invoke("向 user@example.com 发送一封主题为‘你好’、正文为‘这是一封测试邮件’的电子邮件。")
pretty_print(ai_msg)
```

### 示例 5：数据库查询工具
本示例展示一个数据库查询工具，使用 Pydantic 定义执行 SQL 查询的 Schema。该工具接受查询字符串和返回行数的限制。Pydantic 确保查询是有效字符串，限制是整数，有助于防止 SQL 注入等潜在问题。

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal

# 使用 Pydantic 定义输入 Schema
class QueryInput(BaseModel):
    query: str = Field(description="要执行的 SQL 查询（不含 LIMIT 或 ORDER BY 子句）")
    limit: int = Field(description="返回的最大行数", default=100)
    order_by: str = Field(description="结果排序的列", default="id")
    order_direction: Literal["ASC", "DESC"] = Field(description="结果排序方向（ASC 或 DESC）", default="ASC")

# 定义将包装为工具的函数
def execute_query(query: str, limit: int, order_by: str, order_direction: str) -> str:
    """执行带有限制和排序子句的 SQL 查询，并返回结果。"""
    # 构造完整的 SQL 查询，包含 LIMIT 和 ORDER BY
    full_query = f"{query} ORDER BY {order_by} {order_direction} LIMIT {limit}"
    
    # 在实际场景中，此函数将连接数据库并执行查询
    return f"正在执行查询：{full_query}"

# 创建 StructuredTool
query_tool = StructuredTool.from_function(
    func=execute_query,
    name="execute_query",
    description="执行带有限制和排序子句的 SQL 查询，并返回结果。",
    args_schema=QueryInput
)

# 不使用 LLM 调用工具
query_result = query_tool.invoke({
    "query": "SELECT * FROM users",
    "limit": 10,
    "order_by": "name",
    "order_direction": "ASC"
})
print(query_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([query_tool])
ai_msg = model_with_tools.invoke("执行查询‘SELECT * FROM users’，限制为 10 行，按名称升序排序。")
pretty_print(ai_msg)
```

---

## 使用 TypedDict 定义工具 Schema

在 LangChain 中使用 `TypedDict` 定义工具 Schema 是另一种为工具创建结构化输入的方法。`TypedDict` 是 Python 的一个特性，允许定义具有特定键值对和类型的字典。以下是一些使用 `TypedDict` 定义工具 Schema 的示例：

### 示例 1：简单搜索工具
本示例展示如何使用 `TypedDict` 为搜索工具定义输入 Schema。该工具接受搜索查询和结果数量限制。`TypedDict` 确保输入字典包含正确键和适当类型，为工具输入提供清晰结构。

```python
from langchain.tools import StructuredTool
from typing import TypedDict
from typing_extensions import Annotated

# 使用 TypedDict 定义输入 Schema 并添加 Annotated
class SearchInput(TypedDict):
    """根据查询和限制搜索信息。"""
    query: Annotated[str, "搜索查询"]
    limit: Annotated[int, "返回的最大结果数"]

# 定义将包装为工具的函数
def search(query: str, limit: int) -> str:
    """
    根据查询和限制搜索信息。
    
    参数：
        query: 搜索查询
        limit: 返回的最大结果数
        
    返回：
        str: 搜索结果消息
    """
    return f"正在搜索：{query}，限制：{limit}"

# 创建 StructuredTool
search_tool = StructuredTool.from_function(
    func=search,
    name="search",
    description="根据查询和限制搜索信息。",
)

# 情况 1：使用 invoke() 和 input 参数
result1 = search_tool.invoke(
    input={"query": "LangChain", "limit": 5}
)
print("情况 1:", result1)

# 情况 2：使用 run() 方法的替代方式
result2 = search_tool.run({"query": "Python", "limit": 3})
print("情况 2:", result2)

# 情况 3：使用 invoke() 将输入作为第一个参数
result3 = search_tool.invoke({"query": "AI", "limit": 10})
print("情况 3:", result3)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([search_tool])
ai_msg = model_with_tools.invoke("搜索 LangChain，结果限制为 5。")
pretty_print(ai_msg)
```

### 示例 2：计算器工具
本示例中，使用 `TypedDict` 定义计算器工具的输入 Schema。该工具需要两个数值和一个操作（加、减、乘、除）。`TypedDict` 强制输入字典包含所有必要键和正确类型，确保计算器可靠运行。

```python
from langchain.tools import StructuredTool
from typing import TypedDict, Literal
from typing_extensions import Annotated

# 使用 TypedDict 定义输入 Schema
class CalculatorInput(TypedDict):
    num1: Annotated[float, "用于计算的第一个数字"]
    num2: Annotated[float, "用于计算的第二个数字"]
    operation: Annotated[Literal["add", "subtract", "multiply", "divide"], "要执行的操作"]

# 定义将包装为工具的函数
def calculate(num1: float, num2: float, operation: str) -> float:
    """根据提供的数字和操作执行计算。"""
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    else:
        raise ValueError("无效操作")

# 创建 StructuredTool
calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="执行基本算术运算。",
)

# 不使用 LLM 调用工具
calc_result = calculator_tool.invoke(input={"num1": 10, "num2": 5, "operation": "add"})
print(calc_result)  # 输出：15.0
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([calculator_tool])
ai_msg = model_with_tools.invoke("将 10 和 5 相加。")
pretty_print(ai_msg)
```

### 示例 3：天气预报工具
本示例展示如何使用 `TypedDict` 为天气预报工具定义输入 Schema。该工具需要位置和特定日期以提供天气预报。`TypedDict` 确保输入字典包含位置和日期，并具有正确的数据类型，有助于准确预测天气。

```python
from datetime import date

# 使用 TypedDict 定义输入 Schema
class WeatherInput(TypedDict):
    location: Annotated[str, "获取天气的位置"]
    forecast_date: Annotated[date, "获取预报的日期"]

# 定义将包装为工具的函数
def get_weather(location: str, forecast_date: date) -> str:
    """获取特定位置和日期的天气预报。"""
    # 在实际场景中，此函数将调用外部 API
    return f"{location} 在 {forecast_date} 的天气预报：晴天"

# 创建 StructuredTool
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="获取特定位置和日期的天气预报。",
)

# 不使用 LLM 调用工具
weather_result = weather_tool.invoke(input={"location": "纽约", "forecast_date": date(2023, 10, 1)})
print(weather_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([weather_tool])
ai_msg = model_with_tools.invoke("获取纽约 2023 年 10 月 1 日的天气预报。")
pretty_print(ai_msg)
```

### 示例 4：发送电子邮件工具
在此示例中，使用 `TypedDict` 定义发送电子邮件工具的输入结构。该工具需要收件人的电子邮件地址、邮件主题和正文内容。`TypedDict` 确保包含所有必要键并正确类型化，提高邮件发送过程的可靠性和准确性。

```python
# 使用 TypedDict 定义输入 Schema
class EmailInput(TypedDict):
    recipient: Annotated[str, "发送到的电子邮件地址"]
    subject: Annotated[str, "邮件主题"]
    body: Annotated[str, "邮件正文内容"]

# 定义将包装为工具的函数
def send_email(recipient: str, subject: str, body: str) -> str:
    """向指定收件人发送电子邮件。"""
    # 在实际场景中，此函数将调用邮件发送服务
    return f"邮件已发送至 {recipient}，主题：{subject}"

# 创建 StructuredTool
email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="向指定收件人发送电子邮件。",
)

# 不使用 LLM 调用工具
email_result = email_tool.invoke(input={"recipient": "user@example.com", "subject": "你好", "body": "这是一封测试邮件。"})
print(email_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([email_tool])
ai_msg = model_with_tools.invoke("向 user@example.com 发送一封主题为‘你好’、正文为‘这是一封测试邮件’的电子邮件。")
pretty_print(ai_msg)
```

### 示例 5：数据库查询工具
本示例展示一个数据库查询工具，使用 `TypedDict` 定义执行 SQL 查询的输入 Schema。该工具接受查询字符串和返回行数的限制。`TypedDict` 确保输入字典包含查询和限制，并具有正确类型，促进安全高效的数据库交互。

```python
class QueryInput(TypedDict):
    query: Annotated[str, "要执行的 SQL 查询"]
    limit: Annotated[int, "最大行数"]
    order_by: Annotated[str, "排序的列"]
    order_direction: Annotated[Literal["ASC", "DESC"], "排序方向"]

# 定义将包装为工具的函数
def execute_query(query: str, limit: int, order_by: str, order_direction: str) -> str:
    """执行带有限制和排序子句的 SQL 查询，并返回结果。"""
    # 构造完整的 SQL 查询，包含 LIMIT 和 ORDER BY
    full_query = f"{query} ORDER BY {order_by} {order_direction} LIMIT {limit}"
    
    # 在实际场景中，此函数将连接数据库并执行查询
    return f"正在执行查询：{full_query}"

# 创建 StructuredTool
query_tool = StructuredTool.from_function(
    func=execute_query,
    name="execute_query",
    description="执行带有限制和排序子句的 SQL 查询，并返回结果。",
)

# 不使用 LLM 调用工具
query_result = query_tool.invoke(input={
    "query": "SELECT * FROM users",
    "limit": 10,
    "order_by": "name",
    "order_direction": "ASC"
})
print(query_result)
```

```python
# 将工具绑定到 LLM 以在链中使用
model_with_tools = model.bind_tools([query_tool])
ai_msg = model_with_tools.invoke("执行查询‘SELECT * FROM users’，限制为 10 行，按名称升序排序。")
pretty_print(ai_msg)
```

## 结论

定义工具 Schema 是使用 LangChain 构建可靠且易维护应用程序的基础步骤。**Pydantic 类** 和 **TypedDict** 都为结构化工具输入提供了独特的优势，满足不同的需求和偏好。

- **Pydantic 类** 适用于需要强大的数据验证、默认值和复杂数据结构的场景。它们能够强制执行严格的类型检查并提供详细的错误消息，使其适合数据完整性至关重要的应用程序。

- **TypedDict** 则提供了一种轻量且直观的方法来定义输入 Schema。它最适合验证需求较少的简单用例，且需尽量减少性能开销。

通过仔细评估工具的需求并理解每种方法的优势，您可以选择最适合的方式来定义工具 Schema。这一决策不仅会增强工具的可靠性，还能简化其在 LangChain 生态系统中更大工作流程和应用程序中的集成。