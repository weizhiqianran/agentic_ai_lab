# 掌握 LangChain 中的 `StructuredTool.from_function()`

## 引言

在人工智能和自然语言处理快速发展的背景下，将自定义功能无缝集成到语言模型中的能力显得尤为重要。**StructuredTool.from_function()** 是 LangChain 框架中的一个强大工具，它使开发者能够将 Python 函数封装成结构化工具，从而轻松地在复杂的 AI 驱动工作流程中进行集成、执行和管理。本文深入探讨了 `StructuredTool.from_function()` 的多功能应用，提供了全面的示例，展示了其初始化、执行、批量处理能力、错误处理机制以及配置选项。无论您是希望通过算术运算、文本操作还是强大的错误管理来增强 AI 模型，本指南都提供了宝贵的见解和实用实现，以提升您的开发工作。

### 比较表

为了更好地理解 `StructuredTool.from_function()` 的多功能性和适用性，以下比较表总结了示例中探讨的关键功能和配置：

| **功能**                        | **描述**                                                                 | **示例**                             |
|------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------|
| **初始化与配置**                 | 使用输入模式、元数据和标签设置工具以进行分类。                                  | 创建加法和乘法工具                 |
| **直接执行**                     | 使用即时输入运行工具并接收直接输出。                                            | 问候和减法工具                     |
| **批量处理**                     | 同时处理多个输入，可选择附加配置。                                              | 平方和立方运算                     |
| **错误处理**                     | 优雅地管理异常并实现重试机制。                                                  | 除法和随机失败工具                 |
| **配置与绑定**                   | 绑定固定参数并添加可配置字段以实现动态操作。                                    | 幂运算和重复工具                   |

此表概括了 `StructuredTool.from_function()` 提供的多样化功能，展示其满足 AI 驱动项目中广泛应用需求的能力。

---

## 准备工作

### 安装所需库
本节安装使用 LangChain、OpenAI 嵌入、Anthropic 模型及其他工具所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型和 API 的集成。
- `langchain-anthropic`：支持与 Anthropic 模型和 API 的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包含 LangChain 的实验性功能和工具。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```

### 初始化 OpenAI 和 Anthropic 聊天模型
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI 和 Anthropic 的 API 密钥，并初始化它们各自的聊天模型。`ChatOpenAI` 和 `ChatAnthropic` 类用于创建这些模型的实例，可用于自然语言处理任务，如文本生成和对话 AI。

**关键步骤：**
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全检索 OpenAI 和 Anthropic 的 API 密钥。
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
    将 AIMessage 对象转换为 JSON 并格式化后美观打印。
    
    参数：
        aimessage：需要美观打印的 AIMessage 对象。
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
    
    # 将字典转换为格式化的 JSON 字符串
    pretty_json = json.dumps(aimessage_dict, indent=4, ensure_ascii=False)
    
    # 打印格式化后的 JSON
    print(pretty_json)
```

---

## **1. 初始化与配置**

### 示例 1：使用 Pydantic 模式创建工具
此示例展示如何使用 `StructuredTool.from_function()` 初始化和配置结构化工具。它展示了使用 Pydantic 的 `BaseModel` 和 `Field` 定义输入模式，实现一个简单的加法函数，使用定义的模式创建工具，执行特定输入的工具，并将工具绑定到语言模型以集成到链中。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 使用 Pydantic 和 Field 定义输入模式
class AddInput(BaseModel):
    a: int = Field(description="要相加的第一个数字。")
    b: int = Field(description="要相加的第二个数字。")

# 定义函数
def add(a: int, b: int) -> int:
    """将两个数字相加。"""
    return a + b

# 创建工具
add_tool = StructuredTool.from_function(
    func=add,
    name="add",
    description="将两个数字相加。",
    args_schema=AddInput,
)

# 使用工具
result = add_tool.run({"a": 3, "b": 5})
print(result)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([add_tool])
ai_msg = model_with_tools.invoke("3 加 5 是多少？")
pretty_print(ai_msg)
```

---

### 示例 2：使用元数据和标签创建工具
此示例中，通过添加元数据和标签来增强工具的分类和可发现性，创建了一个结构化工具。过程包括使用 Pydantic 定义输入模式，实现乘法函数，并利用 `StructuredTool.from_function()` 添加元数据和标签。然后使用特定输入执行工具，并将其绑定到语言模型以在链中使用。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 使用 Field 定义输入模式
class MultiplyInput(BaseModel):
    x: int = Field(description="要相乘的第一个数字。")
    y: int = Field(description="要相乘的第二个数字。")

# 定义函数
def multiply(x: int, y: int) -> int:
    """将两个数字相乘。"""
    return x * y

# 使用元数据和标签创建工具
multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="将两个数字相乘。",
    args_schema=MultiplyInput,
    metadata={"category": "math"},
    tags=["arithmetic", "math"],
)

# 使用工具
result = multiply_tool.run({"x": 4, "y": 6})
print(result)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([multiply_tool])
ai_msg = model_with_tools.invoke("4 乘以 6 是多少？")
pretty_print(ai_msg)
```

---

## **2. 执行与运行**

### 示例 3：使用直接输入运行工具
此示例说明如何通过提供直接输入来执行结构化工具。它涉及使用 Pydantic 定义输入模式，实现问候函数，使用 `StructuredTool.from_function()` 创建工具，使用特定输入数据运行工具，并将工具绑定到语言模型以集成到对话链中。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 使用 Field 定义输入模式
class GreetInput(BaseModel):
    name: str = Field(description="要问候的人的姓名。")

# 定义函数
def greet(name: str) -> str:
    """问候一个人。"""
    return f"你好，{name}！"

# 创建工具
greet_tool = StructuredTool.from_function(
    func=greet,
    name="greet",
    description="按姓名问候一个人。",
    args_schema=GreetInput,
)

# 使用工具
result = greet_tool.run({"name": "Alice"})
print(result)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([greet_tool])
ai_msg = model_with_tools.invoke("问候 Alice。")
pretty_print(ai_msg)
```

### 示例 4：使用直接返回运行工具
此示例展示如何将 `return_direct` 参数设置为 `True` 来执行工具，使工具直接返回结果而无需额外处理。包括定义输入模式，实现减法函数，使用 `return_direct=True` 创建工具，使用特定输入运行工具，并将工具绑定到语言模型以在链中使用。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 使用 Field 定义输入模式
class SubtractInput(BaseModel):
    a: int = Field(description="要减去的数字。")
    b: int = Field(description="减去的数字。")

# 定义函数
def subtract(a: int, b: int) -> int:
    """将两个数字相减。"""
    return a - b

# 使用 return_direct=True 创建工具
subtract_tool = StructuredTool.from_function(
    func=subtract,
    name="subtract",
    description="将两个数字相减。",
    args_schema=SubtractInput,
    return_direct=True,
)

# 使用工具
result = subtract_tool.run({"a": 10, "b": 4})
print(result)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([subtract_tool])
ai_msg = model_with_tools.invoke("10 减 4 是多少？")
pretty_print(ai_msg)
```

---

## **3. 批量处理**

### 示例 5：使用多个输入进行批量处理
此示例展示如何使用结构化工具执行批量处理。涉及定义平方数的输入模式，实现平方函数，创建工具，准备多个输入，执行批量处理，并将工具绑定到语言模型以在单个操作中处理多个请求。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 第 1 步：使用 Pydantic 和 Field 定义输入模式
# 此模式描述工具的预期输入。
class SquareInput(BaseModel):
    number: int = Field(description="要平方的数字。")

# 第 2 步：定义工具将执行的函数
# 此函数计算数字的平方。
def square(number: int) -> int:
    """对数字进行平方。"""
    return number**2

# 第 3 步：使用 StructuredTool.from_function 创建工具
# 将 `square` 函数封装为具有输入验证的可重用工具。
square_tool = StructuredTool.from_function(
    func=square,  # 要封装的函数
    name="square",  # 工具名称
    description="对数字进行平方。",  # 工具描述
    args_schema=SquareInput,  # 输入模式用于验证
)

# 第 4 步：为批量处理准备输入列表
# 每个输入是与工具输入模式匹配的字典。
inputs = [
    {"number": 2},  # 输入 1：2 的平方
    {"number": 3},  # 输入 2：3 的平方
    {"number": 4},  # 输入 3：4 的平方
]

# 第 5 步：执行批量处理
# `batch` 方法并行处理多个输入。
# 返回与每个输入对应的结果列表。
results = square_tool.batch(inputs)

# 第 6 步：打印结果
# 输出是每个输入的平方值列表。
print(results)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([square_tool])
ai_msg = model_with_tools.invoke("对数字 2、3 和 4 进行平方。")
pretty_print(ai_msg)
```

### 示例 6：带配置的批量处理
此示例说明如何使用附加配置参数进行批量处理。定义立方数的输入模式，实现立方函数，创建工具，准备多个输入，并使用限制并行执行最大并发数的配置执行批量处理。然后将工具绑定到语言模型，以高效处理多个请求并控制资源使用。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 第 1 步：使用 Pydantic 和 Field 定义输入模式
# 此模式描述工具的预期输入。
class CubeInput(BaseModel):
    number: int = Field(description="要立方的数字。")

# 第 2 步：定义工具将执行的函数
# 此函数计算数字的立方。
def cube(number: int) -> int:
    """对数字进行立方。"""
    return number**3

# 第 3 步：使用 StructuredTool.from_function 创建工具
# 将 `cube` 函数封装为具有输入验证的可重用工具。
cube_tool = StructuredTool.from_function(
    func=cube,  # 要封装的函数
    name="cube",  # 工具名称
    description="对数字进行立方。",  # 工具描述
    args_schema=CubeInput,  # 输入模式用于验证
)

# 第 4 步：为批量处理准备输入列表
# 每个输入是与工具输入模式匹配的字典。
inputs = [{"number": 2}, {"number": 3}]

# 第 5 步：使用配置执行批量处理
# `batch` 方法并行处理多个输入。
# `config` 参数允许控制批量处理的行为。
# 此处设置 `max_concurrency=2` 将并行执行限制为 2。
results = cube_tool.batch(inputs, config={"max_concurrency": 2})

# 第 6 步：打印结果
# 输出是每个输入对应的结果列表。
print(results)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([cube_tool])
ai_msg = model_with_tools.invoke("对数字 2 和 3 进行立方。")
pretty_print(ai_msg)
```

## **4. 错误处理与重试**

### 示例 7：处理工具错误
此示例关注结构化工具内的错误处理。定义除法操作的输入模式，实现一个在除以零时引发错误的除法函数，并创建启用了错误处理的工具。然后使用无效输入执行工具，以展示错误如何被管理，并将其绑定到语言模型以在链中实现健壮的错误感知交互。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 使用 Field 定义输入模式
class DivideInput(BaseModel):
    a: int = Field(description="被除数。")
    b: int = Field(description="除数（不得为零）。")

# 定义函数
def divide(a: int, b: int) -> float:
    """将两个数字相除。"""
    if b == 0:
        raise ValueError("不能除以零。")
    return a / b

# 使用错误处理创建工具
divide_tool = StructuredTool.from_function(
    func=divide,
    name="divide",
    description="将两个数字相除。",
    args_schema=DivideInput,
    handle_tool_error=True,  # 启用错误处理
)

# 使用工具
try:
    result = divide_tool.run({"a": 10, "b": 0})
    print(result)
except Exception as e:
    print(f"工具错误：{e}")
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([divide_tool])
ai_msg = model_with_tools.invoke("10 除以 0。")
pretty_print(ai_msg)
```

### 示例 8：在异常时重试
此示例展示如何创建封装易发生随机失败的函数的 `StructuredTool`，并添加重试逻辑以优雅地处理此类失败。涵盖以下关键点：

1. **TypedDict 模式**：使用 `TypedDict` 和 `Annotated` 定义输入模式，清楚描述预期参数及其用途。
2. **随机失败函数**：`random_fail` 函数根据指定的失败概率 (`failure_rate`) 模拟随机失败，并在未失败时返回成功消息。
3. **重试机制**：使用 `with_retry` 方法配置工具，在函数因异常失败时最多重试 3 次。
4. **使用示例**：测试两种情况：
   - 高失败率（80%）以展示重试和错误处理。
   - 较低失败率（30%）以展示重试后的成功执行。

```python
from langchain.tools import StructuredTool
from typing import TypedDict
from typing_extensions import Annotated
import random

# 使用 TypedDict 定义输入模式
class RandomFailInput(TypedDict):
    """随机失败函数的输入。"""
    failure_rate: Annotated[float, "失败概率，介于 0 和 1 之间"]
    message: Annotated[str, "成功时返回的消息"]

# 定义可能随机失败的函数
def random_fail(failure_rate: float, message: str) -> str:
    """
    根据给定失败率随机失败的函数。
    
    参数：
        failure_rate：失败概率（介于 0 和 1 之间）
        message：成功时返回的消息
    """
    if random.random() < failure_rate:
        raise ValueError("发生随机失败！")
    return f"成功：{message}"

# 创建带有重试的工具
random_fail_tool = StructuredTool.from_function(
    func=random_fail,
    name="random_fail",
    description="用于演示的随机失败。",
).with_retry(stop_after_attempt=3)  # 最多重试 3 次

# 使用示例
try:
    # 高失败率以展示重试
    result = random_fail_tool.invoke(input={
        "failure_rate": 0.8,  # 80% 失败概率
        "message": "操作完成！"
    })
    print(f"最终结果：{result}")
except Exception as e:
    print(f"所有重试失败。最终错误：{str(e)}")
```

```python
# 使用较低失败率的示例
try:
    # 较低失败率更可能成功
    result = random_fail_tool.invoke(input={
        "failure_rate": 0.3,  # 30% 失败概率
        "message": "这次尝试应该成功！"
    })
    print(f"最终结果：{result}")
except Exception as e:
    print(f"所有重试失败。最终错误：{str(e)}")
```

---

## **5. 配置与绑定**

### 示例 9：绑定参数
此示例说明如何将特定参数绑定到结构化工具，有效地预配置某些参数。涉及定义幂运算的输入模式，实现幂函数，创建工具，将基数参数绑定到固定值，并使用不同指数执行绑定的工具。然后将工具绑定到语言模型，以在链中无缝集成预配置参数。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 第 1 步：使用 Pydantic 和 Field 定义输入模式
# 此模式描述工具的预期输入。
class PowerInput(BaseModel):
    base: int = Field(description="底数。")
    exponent: int = Field(description="指数。")

# 第 2 步：定义工具将执行的函数
# 此函数计算底数的指数幂。
def power(base: int, exponent: int) -> int:
    """将底数提高到指数的幂。"""
    return base**exponent

# 第 3 步：使用 StructuredTool.from_function 创建工具
# 将 `power` 函数封装为具有输入验证的可重用工具。
power_tool = StructuredTool.from_function(
    func=power,  # 要封装的函数
    name="power",  # 工具名称
    description="将底数提高到指数的幂。",  # 工具描述
    args_schema=PowerInput,  # 输入模式用于验证
)

# 第 4 步：将参数绑定到工具
# `bind` 方法允许固定工具的某些参数。
# 此处将 `base=2` 绑定，因此工具仅需要 `exponent` 作为输入。
bound_tool = power_tool.bind(base=2)

# 第 5 步：使用绑定的工具
# 现在工具仅需 `exponent` 输入，因 `base` 已被固定。
# 需要修改输入以包含绑定的 `base` 值。
result = bound_tool.run({"base": 2, "exponent": 3})  # 在输入中包含 `base`
print("第 5 步：", result)  # 输出：8

# 第 6 步：使用不同指数重用绑定的工具
# 展示如何使用不同输入重用同一绑定工具。
result = bound_tool.run({"base": 2, "exponent": 4})  # 在输入中包含 `base`
print("第 6 步：", result)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([power_tool])
ai_msg = model_with_tools.invoke("计算 2 的 3 次方。")
pretty_print(ai_msg)
```

### 示例 10：可配置字段
此示例展示如何使用 `ConfigurableField` 动态调整语言模型的行为，特别是其温度参数，该参数控制输出的随机性和创造性。

1. **可配置字段定义**：
   - 定义一个名为 `model_temperature` 的 `ConfigurableField`，清楚描述温度如何影响模型输出。较低值使输出更确定，较高值使其更有创造性。
2. **ChatOpenAI 模型初始化**：
   - 创建一个默认温度为 `0.7` 的 `ChatOpenAI` 模型。温度字段设置为可配置，允许运行时调整。
3. **使用默认温度 (0.7)**：
   - 模型使用默认温度生成输出，在创造性和确定性之间取得平衡。
4. **重新配置为较低温度 (0.2)**：
   - 将温度设置为较低值，导致输出更确定且随机性减少。
5. **重新配置为较高温度 (1.0)**：
   - 将温度设置为较高值，鼓励更具创造性和多样性的响应。

```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

config = ConfigurableField(
    id="model_temperature",  # 可配置字段的唯一标识符
    name="模型温度",  # 字段的人类可读名称
    description="控制模型输出的随机性。较低值使输出更确定，较高值使其更有创造性。",  # 字段描述
)

# 第 1 步：创建具有可配置温度的 ChatOpenAI 模型
llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0.7, 
                 api_key=user_secrets.get_secret("my-openai-api-key")).configurable_fields(temperature=config)

# 第 2 步：使用默认温度 (0.7) 的模型
print(
    "温度 0.7（默认）：",
    llm.invoke("讲个笑话。").content
)
```

```python
# 第 3 步：将模型重新配置为较低温度 (0.2)
# 较低温度使输出更确定。
print(
    "温度 0.2：",
    llm.with_config(configurable={"model_temperature": 0.2}).invoke("讲个笑话。").content
)
```

```python
# 第 4 步：将模型重新配置为较高温度 (1.0)
# 较高温度使输出更有创造性。
print(
    "温度 1.0：",
    llm.with_config(configurable={"model_temperature": 1.0}).invoke("讲个笑话。").content
)
```

---

## **最佳实践**

### 示例 11：使用 `from_function` 的所有参数
此示例展示如何充分利用 LangChain 的 `StructuredTool` 中 `from_function` 方法的所有参数。示例突出显示了高级自定义选项，例如：

1. **自定义 Pydantic 模式**：定义的模式确保对输入的精确验证。
2. **Google 风格文档字符串解析**：解析函数的文档字符串以推断模式并增强工具文档。
3. **响应格式**：工具输出由消息和元数据组成的元组。
4. **错误处理**：如果文档字符串格式无效则引发错误。
5. **元数据和标签**：添加元数据用于分类，标签帮助描述工具用途。
6. **直接响应返回**：结果直接返回，无需额外处理。

此高级设置适用于需要详细自定义和严格验证的复杂函数。

```python
from pydantic import BaseModel, Field
from typing import Awaitable, Any, Literal
from langchain_core.tools import StructuredTool

# 定义带有 Google 风格文档字符串的函数
def complex_function(a: int, b: str) -> tuple[str, dict]:
    """
    处理输入并返回元组的复杂函数。

    参数：
        a (int)：第一个输入，一个整数。
        b (str)：第二个输入，一个字符串。

    返回：
        tuple[str, dict]：包含消息和元数据字典的元组。
    """
    message = f"已处理：{a} 和 {b}"
    metadata = {"input_a": a, "input_b": b}
    return message, metadata

# 定义用于输入验证的自定义 Pydantic 模式
class ComplexInput(BaseModel):
    a: int = Field(description="第一个输入，一个整数。")
    b: str = Field(description="第二个输入，一个字符串。")

# 使用 from_function 的所有参数创建工具
complex_tool = StructuredTool.from_function(
    func=complex_function,  # 要封装的函数
    coroutine=None,  # 未提供异步版本
    name="complex_tool",  # 自定义名称
    description="处理输入并返回元组的工具。",  # 自定义描述
    return_direct=True,  # 直接返回结果
    args_schema=ComplexInput,  # 使用自定义 Pydantic 模式
    infer_schema=True,  # 从函数签名推断模式
    response_format="content_and_artifact",  # 期望元组作为输出
    parse_docstring=True,  # 解析 Google 风格文档字符串
    error_on_invalid_docstring=True,  # 如果文档字符串无效则引发错误
    metadata={"category": "demo"},  # 附加元数据
    tags=["example", "advanced"],  # 用于分类的标签
)

# 使用工具
result = complex_tool.run({"a": 42, "b": "example"})
print(result)
```

```python
# 将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([complex_tool])
ai_msg = model_with_tools.invoke("将 'hi' 重复 3 次。")
pretty_print(ai_msg)
```

### 示例 12：使用多个工具与 `from_function`
此示例展示如何使用 `ConfigurableField` 动态调整语言模型的行为，特别是其温度参数，该参数控制输出的随机性和创造性。

1. **可配置字段定义**：
   - 定义一个名为 `model_temperature` 的 `ConfigurableField`，清楚描述温度如何影响模型输出。较低值使输出更确定，较高值使其更有创造性。
2. **ChatOpenAI 模型初始化**：
   - 创建一个默认温度为 `0.7` 的 `ChatOpenAI` 模型。温度字段设置为可配置，允许运行时调整。
3. **使用默认温度 (0.7)**：
   - 模型使用默认温度生成输出，在创造性和确定性之间取得平衡。
4. **重新配置为较低温度 (0.2)**：
   - 将温度设置为较低值，导致输出更确定且随机性减少。
5. **重新配置为较高温度 (1.0)**：
   - 将温度设置为较高值，鼓励更具创造性和多样性的响应。

此设置特别适用于需要动态控制模型行为的应用，例如交互式系统或需要不同创造性水平的场景。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 第 1 步：使用 Pydantic 和 Field 定义输入模式

# “currency_converter” 工具的模式
class CurrencyConverterInput(BaseModel):
    amount: float = Field(description="要转换的金额。")
    from_currency: str = Field(description="要转换的货币（例如 USD）。")
    to_currency: str = Field(description="目标货币（例如 EUR）。")

# “weather_lookup” 工具的模式
class WeatherLookupInput(BaseModel):
    location: str = Field(description="要查询天气的地点（例如 New York）。")

# “text_summarizer” 工具的模式
class TextSummarizerInput(BaseModel):
    text: str = Field(description="要总结的文本。")
    max_length: int = Field(default=100, description="总结的最大长度。")

# 第 2 步：为每个工具定义函数

# “currency_converter” 工具的函数
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """将金额从一种货币转换为另一种货币。"""
    # 模拟货币转换（在现实场景中会调用 API）
    conversion_rates = {
        "USD": {"EUR": 0.93, "GBP": 0.80, "JPY": 148.50},
        "EUR": {"USD": 1.07, "GBP": 0.86, "JPY": 159.50},
        "GBP": {"USD": 1.25, "EUR": 1.16, "JPY": 185.00},
        "JPY": {"USD": 0.0067, "EUR": 0.0063, "GBP": 0.0054},
    }
    if from_currency not in conversion_rates or to_currency not in conversion_rates[from_currency]:
        return f"不支持从 {from_currency} 到 {to_currency} 的转换。"
    rate = conversion_rates[from_currency][to_currency]
    converted_amount = amount * rate
    return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency}"

# “weather_lookup” 工具的函数
def weather_lookup(location: str) -> str:
    """获取给定地点的当前天气。"""
    # 模拟天气查询（在现实场景中会调用 API）
    weather_data = {
        "New York": "晴，72°F",
        "London": "多云，55°F",
        "Tokyo": "雨，65°F",
        "Paris": "局部多云，68°F",
    }
    if location not in weather_data:
        return f"{location} 的天气数据不可用。"
    return f"{location} 的天气是 {weather_data[location]}。"

# “text_summarizer” 工具的函数
def text_summarizer(text: str, max_length: int = 100) -> str:
    """将给定文本总结为指定最大长度。"""
    # 模拟文本总结（在现实场景中会使用 NLP 模型）
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

# 第 3 步：使用 StructuredTool.from_function 创建工具

# 创建 “currency_converter” 工具
currency_converter_tool = StructuredTool.from_function(
    func=currency_converter,
    name="currency_converter",
    description="将金额从一种货币转换为另一种货币。",
    args_schema=CurrencyConverterInput,
)

# 创建 “weather_lookup” 工具
weather_lookup_tool = StructuredTool.from_function(
    func=weather_lookup,
    name="weather_lookup",
    description="获取给定地点的当前天气。",
    args_schema=WeatherLookupInput,
)

# 创建 “text_summarizer” 工具
text_summarizer_tool = StructuredTool.from_function(
    func=text_summarizer,
    name="text_summarizer",
    description="将给定文本总结为指定最大长度。",
    args_schema=TextSummarizerInput,
)

# 第 4 步：使用工具

# 使用 “currency_converter” 工具
conversion_result = currency_converter_tool.run({"amount": 100, "from_currency": "USD", "to_currency": "EUR"})
print("货币转换结果：", conversion_result)  # 输出：100 USD = 93.00 EUR

# 使用 “weather_lookup” 工具
weather_result = weather_lookup_tool.run({"location": "New York"})
print("天气查询结果：", weather_result)  # 输出：New York 的天气是晴，72°F

# 使用 “text_summarizer” 工具
text = "The quick brown fox jumps over the lazy dog. This is a long text that needs to be summarized."
summary_result = text_summarizer_tool.run({"text": text, "max_length": 20})
print("文本总结结果：", summary_result)  # 输出：The quick brown fox...

# 第 5 步：将工具绑定到语言模型以在链中使用
model_with_tools = model.bind_tools([currency_converter_tool, weather_lookup_tool, text_summarizer_tool])
```

```python
# 使用绑定工具的模型
ai_msg = model_with_tools.invoke("将 100 USD 转换为 EUR 并检查伦敦的天气。")
pretty_print(ai_msg)
```

```python
# 使用绑定工具的模型
ai_msg = model_with_tools.invoke("将 200 USD 转换为 JPY。")
pretty_print(ai_msg)
```

```python
# 使用绑定工具的模型
ai_msg = model_with_tools.invoke("东京的天气如何？")
pretty_print(ai_msg)
```

```python
# 使用绑定工具的模型
ai_msg = model_with_tools.invoke("总结这段文本：'The quick brown fox jumps over the lazy dog. This is a long text that needs to be summarized.'")
pretty_print(ai_msg)
```

```python
# 使用绑定工具的模型
ai_msg = model_with_tools.invoke("讲个笑话。")
pretty_print(ai_msg)
```

## 结论

`StructuredTool.from_function()` 是开发人员扩展语言模型功能的基石，通过自定义结构化功能实现这一目标。通过提供的全面示例，我们探索了此工具如何促进创建健壮的工具，涵盖了使用详细模式的初始化、直接和批量执行、复杂的错误处理以及动态配置。通过利用 Pydantic 进行输入验证并集成元数据和标签以增强工具管理，开发人员可以打造针对特定任务的精确且可靠的工具。此外，运行时绑定参数和配置字段的能力增添了灵活性，使 `StructuredTool.from_function()` 成为构建可扩展且可维护 AI 应用中不可或缺的资产。随着对更具交互性和智能系统的需求增长，掌握像 `StructuredTool.from_function()` 这样的工具将在推动 AI 开发的创新和效率中发挥关键作用。