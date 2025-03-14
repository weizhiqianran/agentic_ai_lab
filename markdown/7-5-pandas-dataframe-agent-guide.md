# 使用 LangChain 的 Pandas DataFrame 代理

## 引言

`langchain_experimental.agents.agent_toolkits.pandas.base` 模块中的 `create_pandas_dataframe_agent` 函数是一个强大的工具，旨在使语言模型（LLM）能够与存储在 Pandas DataFrame 中的数据进行交互和分析。通过将 LLM 的能力与 Pandas 的灵活性相结合，该函数允许用户通过自然语言查询执行复杂的数据分析任务。该代理可以处理多种操作，例如过滤、聚合和可视化，使其成为数据科学家、分析师和开发人员简化工作流程的宝贵工具。

然而，需要注意的是，此功能伴随着重要的安全考虑。代理依赖于 Python REPL（读取-求值-打印循环）工具，可以执行任意代码。这一功能虽然强大，但如果未在适当的沙盒环境中使用，则会引入潜在风险，例如任意代码执行漏洞。用户必须通过将 `allow_dangerous_code` 参数设置为 `True` 来明确选择启用此功能，承认风险并确保采取适当的保护措施。

---

## 准备工作

### 安装所需库
本节安装使用 LangChain、OpenAI 嵌入、Anthropic 模型和其他实用工具所需的 Python 库。这些库包括：
- `langchain-openai`：提供与 OpenAI 嵌入模型和 API 的集成。
- `langchain-anthropic`：支持与 Anthropic 模型和 API 的集成。
- `langchain_community`：包含 LangChain 的社区贡献模块和工具。
- `langchain_experimental`：包括 LangChain 的实验性功能和实用工具。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```

### 初始化 OpenAI 和 Anthropic 聊天模型
本节展示如何使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI 和 Anthropic 的 API 密钥，并初始化它们的聊天模型。`ChatOpenAI` 和 `ChatAnthropic` 类用于创建这些模型的实例，可用于自然语言处理任务，如文本生成和对话 AI。

**关键步骤：**
1. **获取 API 密钥**：使用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI 和 Anthropic 的 API 密钥。
2. **初始化聊天模型**：
   - 使用 `gpt-4o-mini` 模型和获取的 OpenAI API 密钥初始化 `ChatOpenAI` 类。
   - 使用 `claude-3-5-sonnet-latest` 模型和获取的 Anthropic API 密钥初始化 `ChatAnthropic` 类。

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化 LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
#model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))

# 加载泰坦尼克号数据集
df = pd.read_csv("/kaggle/input/titanic-dataset/titanic.csv")
```

提供的代码定义了一个实用函数 `pretty_print_response`，用于以清晰且易读的方式格式化和显示代理的响应输出。

```python
def pretty_print_response(question, response):
    """
    以结构化和易读的方式格式化并打印代理的响应。

    该函数接受一个问题和代理的响应，从响应中提取 'output' 键，并以清晰的分隔符打印以提高可读性。
    它对于调试、记录或以用户友好的格式呈现结果非常有用。

    参数：
        question (str)：向代理提出的问题或查询。
        response (dict)：代理的响应，预期包含带有查询结果的 'output' 键。

    返回：
        None：此函数直接将格式化的输出打印到控制台。
    """
    print(f"问题：{question}")
    print("\n" + "=" * 80 + "\n")
    print(response["output"])  # 提取 'output' 键以进行美观打印
    print("\n" + "=" * 80 + "\n")
```

---

## 示例

`create_pandas_dataframe_agent` 函数是一个强大的工具，可以使用语言模型（LLM）分析和交互 Pandas DataFrame。它允许您通过自然语言查询执行复杂的数据分析任务，例如过滤、聚合和可视化。以下是展示 `create_pandas_dataframe_agent` 中关键参数使用的示例。

### **示例 1：使用默认参数的基本用法**
此示例展示如何使用默认参数创建 Pandas 代理。代理使用 DataFrame 和语言模型（`llm`）初始化。它可以回答有关数据集的问题，例如列名、数据类型和基本统计信息。

```python
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

# 加载泰坦尼克号数据集
df = pd.read_csv("/kaggle/input/titanic-dataset/titanic.csv")

# 使用默认参数创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",  # 使用现代的 "tool-calling" 代理类型
    allow_dangerous_code=True,  # 允许执行 Python 代码（需谨慎使用）
    verbose=True,               # 启用详细日志以便调试
)

# 向代理提问
response = agent.invoke("数据集中的列及其数据类型是什么？")
print(response["output"])
```

### **示例 2：自定义提示前缀和后缀**
此示例展示如何使用 `prefix` 和 `suffix` 参数自定义代理的行为。`prefix` 为代理提供上下文，而 `suffix` 指定输出的格式。

```python
# 使用自定义前缀和后缀创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    prefix="你是一名数据分析师。分析泰坦尼克号数据集并提供简洁的答案。",
    suffix="以清晰且结构化的格式提供最终答案。",
    allow_dangerous_code=True,
    verbose=True,
)

# 向代理提问
response = agent.invoke("乘客的平均年龄是多少？")
print(response["output"])
```

### **示例 3：在提示中包含 DataFrame 头部**
此示例展示如何使用 `include_df_in_prompt` 和 `number_of_head_rows` 参数在提示中包含 DataFrame 的前几行。这有助于代理理解数据的结构。

```python
# 在提示中包含前 5 行创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    include_df_in_prompt=True,  # 在提示中包含 DataFrame 头部
    number_of_head_rows=5,      # 包含的行数
    allow_dangerous_code=True,
    verbose=True,
)

# 向代理提问
response = agent.invoke("'Pclass' 列中的唯一值是什么？")
print(response["output"])
```

### **示例 4：添加额外工具**
此示例展示如何使用 `extra_tools` 参数为代理添加额外工具。这些工具可以扩展代理的功能，例如执行自定义计算或与外部 API 交互。

```python
from langchain.tools import Tool

# 定义自定义工具
def custom_calculation_tool(input: str) -> str:
    return f"自定义计算结果为：{input}"

# 将自定义工具添加到代理
extra_tools = [
    Tool(
        name="custom_calculation",
        func=custom_calculation_tool,
        description="一个用于执行自定义计算的工具。"
    )
]

# 使用额外工具创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    extra_tools=extra_tools,  # 添加自定义工具
    allow_dangerous_code=True,
    verbose=True,
)

# 要求代理使用自定义工具
response = agent.invoke("使用 custom_calculation 工具处理 'example input'。")
print(response["output"])
```

### **示例 5：限制执行时间和迭代次数**
此示例展示如何使用 `max_execution_time` 和 `max_iterations` 参数限制代理的执行时间和迭代次数。这对于控制资源使用非常有用。

```python
# 使用执行限制创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    max_execution_time=10,  # 将执行时间限制为 10 秒
    max_iterations=5,       # 将迭代次数限制为 5
    allow_dangerous_code=True,
    verbose=True,
)

# 向代理提问
response = agent.invoke("乘客的存活率是多少？")
print(response["output"])
```

### **示例 6：处理缺失数据**
此示例展示如何使用代理识别和处理 DataFrame 中的缺失数据。

```python
# 创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    allow_dangerous_code=True,
    verbose=True,
)

# 要求代理识别缺失数据
response = agent.invoke("哪些列有缺失数据，我们应该如何处理它们？")
print(response["output"])
```

### **示例 7：自定义提前停止**
此示例展示如何使用 `early_stopping_method` 参数自定义提前停止行为。如果代理遇到错误或达到停止条件，它将停止处理。

```python
# 使用自定义提前停止创建代理
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    early_stopping_method="force",  # 在错误时强制停止
    allow_dangerous_code=True,
    verbose=True,
)

# 向代理提问
response = agent.invoke("每个乘客等级的平均票价是多少？")
print(response["output"])
```

---

## 最佳实践

### 示例 1：使用 `create_pandas_dataframe_agent` 分析泰坦尼克号数据集
此示例展示如何使用 `create_pandas_dataframe_agent` 分析泰坦尼克号数据集。代理能够执行各种数据分析任务，例如数据探索、过滤、聚合和可视化，使用自然语言查询。

#### **步骤 1：加载数据集并初始化代理**
此代码块加载泰坦尼克号数据集并初始化 `create_pandas_dataframe_agent`。代理配置为使用语言模型（`model`），并设置为允许执行潜在危险的代码（例如 Python 代码执行）。`verbose` 标志启用代理操作的详细日志。

```python
# 加载泰坦尼克号数据集
df = pd.read_csv("/kaggle/input/titanic-dataset/titanic.csv")

# 创建代理
agent_executor = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    allow_dangerous_code=True,
    verbose=True
)
```

#### **步骤 2：使用代理执行数据分析**
此代码块展示如何使用代理回答有关泰坦尼克号数据集的各种问题。问题范围从基本探索到复杂分析，代理动态处理数据以提供洞察。`pretty_print_response` 函数用于以清晰且易读的方式格式化和打印响应。

```python
# 问题 1：基本数据探索
response_1 = agent_executor.invoke("数据集中的列及其数据类型是什么？")
pretty_print_response("1. 数据集中的列及其数据类型是什么？", response_1)
```

```python
# 问题 2：过滤数据
response_2 = agent_executor.invoke("找出所有存活的乘客。")
pretty_print_response("2. 找出所有存活的乘客。", response_2)
```

```python
# 问题 3：聚合
response_3 = agent_executor.invoke("乘客的平均年龄是多少？")
pretty_print_response("3. 乘客的平均年龄是多少？", response_3)
```

```python
# 问题 4：分组和聚合
response_4 = agent_executor.invoke("每个乘客等级的平均票价是多少？")
pretty_print_response("4. 每个乘客等级的平均票价是多少？", response_4)
```

```python
# 问题 5：可视化
response_5 = agent_executor.invoke("创建显示每个等级乘客数量的条形图。")
pretty_print_response("5. 创建显示每个等级乘客数量的条形图。", response_5)
```

```python
# 问题 6：条件分析
response_6 = agent_executor.invoke("女性乘客的存活率是多少？")
pretty_print_response("6. 女性乘客的存活率是多少？", response_6)
```

```python
# 问题 7：处理缺失数据
response_7 = agent_executor.invoke("哪些列有缺失数据，我们应该如何处理它们？")
pretty_print_response("7. 哪些列有缺失数据，我们应该如何处理它们？", response_7)
```

```python
# 问题 8：复杂查询
response_8 = agent_executor.invoke("找出所有存活、头等舱且年龄超过 30 岁的乘客姓名。")
pretty_print_response("8. 找出所有存活、头等舱且年龄超过 30 岁的乘客姓名。", response_8)
```

#### **步骤 3：自定义提示进行高级分析**
此代码块展示如何使用自定义提示指导代理的行为。代理使用前缀初始化，指示其作为数据分析师并提供简洁的答案。这对于定制代理的响应以适应特定任务或受众非常有用。

```python
# 问题 9：自定义提示
agent_executor_custom = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="tool-calling",
    prefix="你是一名数据分析师。分析泰坦尼克号数据集并提供简洁的答案。",
    allow_dangerous_code=True,
    verbose=True
)
response_9 = agent_executor_custom.invoke("按性别分布的乘客情况如何？")
pretty_print_response("9. 按性别分布的乘客情况如何？", response_9)
```

### 示例 2：使用 `create_pandas_dataframe_agent` 进行数据分析

此示例展示如何使用 `create_pandas_dataframe_agent` 分析多个 DataFrame，而无需事先合并它们。代理能够动态执行操作，如跨多个 DataFrame 进行过滤、聚合和条件分析。

#### **步骤 1：创建 DataFrame**
此代码块创建三个 DataFrame：
1. **`df_titanic`**：包含乘客详细信息，如 `PassengerId`、`Name`、`Pclass` 和 `Fare`。
2. **`df_fare_category`**：将 `Pclass` 映射到 `FareCategory`（例如头等舱、二等舱、经济舱）。
3. **`df_discount`**：包含 `PassengerId` 和 `Discount` 百分比。

```python
import pandas as pd

# 创建第一个 DataFrame：泰坦尼克号乘客数据
data_titanic = {
    "PassengerId": [1, 2, 3, 4, 5],
    "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina", "Futrelle, Mrs. Jacques Heath", "Allen, Mr. William Henry"],
    "Pclass": [3, 1, 3, 1, 3],
    "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
}

df_titanic = pd.DataFrame(data_titanic)

# 创建第二个 DataFrame：票价类别数据
data_fare_category = {
    "Pclass": [1, 2, 3],
    "FareCategory": ["First Class", "Second Class", "Economy"],
}

df_fare_category = pd.DataFrame(data_fare_category)

# 创建第三个 DataFrame：折扣数据
data_discount = {
    "PassengerId": [1, 2, 3, 4, 5],
    "Discount": [0.0, 5.0, 0.0, 10.0, 0.0],  # 折扣百分比
}

df_discount = pd.DataFrame(data_discount)
```

#### **步骤 2：初始化代理**
此代码块使用多个 DataFrame 初始化 `create_pandas_dataframe_agent`。代理可以动态连接、过滤和分析这些 DataFrame 中的数据。

```python
from langchain_experimental.agents import create_pandas_dataframe_agent

# 使用多个 DataFrame 创建代理
agent = create_pandas_dataframe_agent(
    model,
    [df_titanic, df_fare_category, df_discount],  # 以列表形式传递多个 DataFrame
    agent_type="tool-calling",
    allow_dangerous_code=True,
    verbose=True
)
```

#### **步骤 3：执行数据分析**
此代码块展示如何使用代理回答有关数据的各种问题。问题范围从基本探索到复杂分析，代理动态处理 DataFrame 以提供洞察。

```python
# 问题 1：基本数据探索
response_1 = agent.invoke("每个 DataFrame 的列及其数据类型是什么？")
pretty_print_response("1. 每个 DataFrame 的列及其数据类型是什么？", response_1)
```

```python
# 问题 2：过滤数据
response_2 = agent.invoke("通过连接 df_titanic 和 df_fare_category 找出所有头等舱乘客。")
pretty_print_response("2. 通过连接 df_titanic 和 df_fare_category 找出所有头等舱乘客。", response_2)
```

```python
# 问题 3：聚合
response_3 = agent.invoke("每个票价类别的平均票价是多少？使用 df_titanic 和 df_fare_category。")
pretty_print_response("3. 每个票价类别的平均票价是多少？使用 df_titanic 和 df_fare_category。", response_3)
```

```python
# 问题 4：条件分析
response_4 = agent.invoke("获得折扣的乘客总票价是多少？使用 df_titanic 和 df_discount。")
pretty_print_response("4. 获得折扣的乘客总票价是多少？使用 df_titanic 和 df_discount。", response_4)
```

```python
# 问题 5：复杂查询
response_5 = agent.invoke("找出总票价超过 50 美元的乘客姓名。使用所有 DataFrame。")
pretty_print_response("5. 找出总票价超过 50 美元的乘客姓名。使用所有 DataFrame。", response_5)
```

```python
# 问题 6：处理缺失数据
response_6 = agent.invoke("df_titanic 中有缺失值吗？")
pretty_print_response("6. df_titanic 中有缺失值吗？", response_6)
```

```python
# 问题 7：自定义提示
response_7 = agent.invoke("你是一名数据分析师。分析 DataFrame 并提供有关票价分布的洞察。")
pretty_print_response("7. 分析 DataFrame 并提供有关票价分布的洞察。", response_7)
```

---

## 结论
`create_pandas_dataframe_agent` 函数弥合了自然语言处理与数据分析之间的差距，使用户能够以直观且高效的方式与 Pandas DataFrame 交互。通过利用 LLM 的优势，此工具简化了复杂的数据任务，使其对更广泛的受众可访问。然而，这一功能的强大之处伴随着安全使用的责任。用户必须确保在安全的沙盒环境中操作，并了解执行任意代码相关的风险。如果负责任地使用，此工具可以显著提高生产力并为数据驱动的决策开启新的可能性。