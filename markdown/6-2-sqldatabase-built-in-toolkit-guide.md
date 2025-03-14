# LangChain SQLDatabase工具包快速参考

## **引言**  
在数据驱动的应用领域中，能够高效且准确地与SQL数据库交互至关重要。LangChain库提供了一套强大的工具，旨在简化和增强数据库交互体验。其中包括 **InfoSQLDatabaseTool**、**ListSQLDatabaseTool**、**QuerySQLCheckerTool** 和 **QuerySQLDatabaseTool**，每种工具在SQL工作流程中都有其独特用途。  

- **InfoSQLDatabaseTool** 帮助用户检索特定表的模式和样本行，便于理解数据库的结构和内容。  
- **ListSQLDatabaseTool** 提供了一种快速列出数据库中所有表的方法，使用户能够探索可用数据源。  
- **QuerySQLCheckerTool** 通过验证和修复常见错误，确保SQL查询的正确性，降低执行过程中的错误风险。  
- **QuerySQLDatabaseTool** 执行SQL查询并返回结果，是数据检索和分析的多功能工具。  

这些工具结合使用时，构成了一个与SQL数据库交互的强大框架，适用于数据探索、查询验证或执行。本文将探讨这些工具的功能和实际应用，提供示例和见解，帮助您在项目中有效利用它们。

### **SQL工具比较**

| **功能**                | **InfoSQLDatabaseTool**                                                                 | **ListSQLDatabaseTool**                                                      | **QuerySQLCheckerTool**                                                                 | **QuerySQLDatabaseTool**                                                      |
|-------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **用途**                | 检索指定表的模式和样本行。                                                             | 列出数据库中的所有表。                                                       | 使用大型语言模型（LLM）验证SQL查询的正确性。                                             | 对数据库执行SQL查询并返回结果。                                                |
| **输入**                | 逗号分隔的表名列表。                                                                   | 通常为空字符串（无需输入）。                                                 | 需要验证的SQL查询。                                                                     | 需要执行的SQL查询。                                                            |
| **输出**                | 指定表的模式和样本行。                                                                 | 数据库中表的逗号分隔列表。                                                   | 经过验证的SQL查询或查询错误时的错误消息。                                               | 查询结果或查询失败时的错误消息。                                                |
| **主要用例**            | 了解特定表的结构和样本数据。                                                           | 发现数据库中所有可用表。                                                     | 在执行前确保SQL查询正确。                                                               | 运行SQL查询并检索结果。                                                        |
| **依赖项**              | 需要一个`SQLDatabase`对象。                                                            | 需要一个`SQLDatabase`对象。                                                  | 需要一个`SQLDatabase`对象和一个LLM（如GPT）进行验证。                                    | 需要一个`SQLDatabase`对象。                                                    |
| **常用方法**            | `run`、`arun`、`invoke`、`ainvoke`、`batch`、`abatch`、`stream`、`astream`。            | `run`、`arun`、`invoke`、`ainvoke`、`batch`、`abatch`、`stream`、`astream`。 | `run`、`arun`、`invoke`、`ainvoke`、`batch`、`abatch`、`stream`、`astream`。            | `run`、`arun`、`invoke`、`ainvoke`、`batch`、`abatch`、`stream`、`astream`。   |
| **错误处理**            | 返回模式和样本行；如果表不存在则报错。                                                  | 返回表名；如果数据库无法访问则报错。                                          | 返回修正后的查询或查询无效时的错误消息。                                                | 返回查询结果或查询失败时的错误消息。                                            |
| **异步支持**            | 是（`arun`、`ainvoke`、`abatch`、`astream`）。                                          | 是（`arun`、`ainvoke`、`abatch`、`astream`）。                               | 是（`arun`、`ainvoke`、`abatch`、`astream`）。                                          | 是（`arun`、`ainvoke`、`abatch`、`astream`）。                                 |
| **配置选项**            | 支持`with_config`、`with_retry`、`with_listeners`、`with_fallbacks`。                   | 支持`with_config`、`with_retry`、`with_listeners`、`with_fallbacks`。        | 支持`with_config`、`with_retry`、`with_listeners`、`with_fallbacks`。                   | 支持`with_config`、`with_retry`、`with_listeners`、`with_fallbacks`。          |
| **示例输入**            | `"Customer, Invoice"`                                                                  | `""`（空字符串）                                                            | `"SELECT * FROM Customers WHERE Country = 'USA'"`                                      | `"SELECT * FROM Customers WHERE Country = 'USA'"`                             |
| **示例输出**            | `Customer`和`Invoice`表的模式和样本行。                                                | `"Customer, Invoice, Order"`（表名列表）。                                  | `"SELECT * FROM Customers WHERE Country = 'USA'"`（经过验证的查询）。                  | 查询结果（例如`Customers`表中的行）。                                           |

### **何时使用哪种工具？**
- 当需要了解特定表的结构和样本数据时，使用 **InfoSQLDatabaseTool**。
- 当希望发现数据库中的所有表时，使用 **ListSQLDatabaseTool**。
- 当需要在执行前验证SQL查询时，使用 **QuerySQLCheckerTool**。
- 当需要执行SQL查询并检索结果时，使用 **QuerySQLDatabaseTool**。

---

## 准备工作

### 安装必要的库
本节将安装使用LangChain、OpenAI嵌入、Anthropic模型和其他实用工具所需的Python库。这些库包括：
- `langchain-openai`：提供与OpenAI嵌入模型和API的集成。
- `langchain-anthropic`：支持与Anthropic模型和API的集成。
- `langchain_community`：包含LangChain的社区贡献模块和工具。
- `langchain_experimental`：包括LangChain的实验性功能和实用工具。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
!pip install -qU langchainhub
```

### 初始化OpenAI和Anthropic聊天模型
本节展示如何使用Kaggle的`UserSecretsClient`安全获取OpenAI和Anthropic的API密钥，并初始化它们的聊天模型。使用`ChatOpenAI`和`ChatAnthropic`类创建这些模型实例，可用于文本生成和对话AI等自然语言处理任务。

**关键步骤：**
1. **获取API密钥**：通过Kaggle的`UserSecretsClient`安全检索OpenAI和Anthropic的API密钥。
2. **初始化聊天模型**：
   - 使用`gpt-4o-mini`模型和获取的OpenAI API密钥初始化`ChatOpenAI`。
   - 使用`claude-3-5-sonnet-latest`模型和获取的Anthropic API密钥初始化`ChatAnthropic`。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from kaggle_secrets import UserSecretsClient

# 安全获取API密钥
user_secrets = UserSecretsClient()

# 初始化模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=user_secrets.get_secret("my-openai-api-key"))
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, api_key=user_secrets.get_secret("my-anthropic-api-key"))
```

---

## **InfoSQLDatabaseTool 示例**

### 示例1：检索特定表的模式和样本行
获取`Customer`和`Invoice`表的模式和样本行，以了解其结构和数据。

```python
from langchain_community.tools.sql_database.tool import InfoSQLDatabaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine

# 创建SQLAlchemy引擎和SQLDatabase对象
engine = create_engine("sqlite:////kaggle/input/chinook-database/chinook.db")
db = SQLDatabase(engine)

# 初始化工具
info_tool = InfoSQLDatabaseTool(db=db)

# 检索Customer和Invoice表的模式和样本行
result = info_tool.run("Customer, Invoice")
print(result)
```

### 示例2：为数据分析探索表结构
分析`Track`和`Album`表的模式和样本行，为数据分析任务做准备。

```python
# 初始化工具
info_tool = InfoSQLDatabaseTool(db=db)

# 检索Track和Album表的模式和样本行
result = info_tool.run("Track, Album")
print(result)
```

### 示例3：调试表关系
检查`PlaylistTrack`和`Playlist`表的模式，了解它们之间的关系。

```python
# 初始化工具
info_tool = InfoSQLDatabaseTool(db=db)

# 检索PlaylistTrack和Playlist表的模式和样本行
result = info_tool.run("PlaylistTrack, Playlist")
print(result)
```

---

## **ListSQLDatabaseTool 示例**

### 示例1：列出数据库中的所有表
检索`chinook.db`数据库中所有表的列表，以了解其结构。

```python
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool

# 创建SQLAlchemy引擎和SQLDatabase对象
engine = create_engine("sqlite:////kaggle/input/chinook-database/chinook.db")
db = SQLDatabase(engine)

# 初始化工具
list_tool = ListSQLDatabaseTool(db=db)

# 列出数据库中的所有表
tables = list_tool.run("")
print(tables)
```

### 示例2：在查询前验证表是否存在
在运行查询前，检查特定表（例如`Customer`、`Invoice`）是否存在。

```python
# 初始化工具
list_tool = ListSQLDatabaseTool(db=db)

# 列出所有表并检查特定表
tables = list_tool.run("")
if "Customer" in tables and "Invoice" in tables:
    print("表存在。可以继续查询。")
else:
    print("所需表不存在。")
```

### 示例3：根据表列表动态生成查询
使用表列表为每个表动态生成查询。

```python
# 初始化工具
list_tool = ListSQLDatabaseTool(db=db)

# 列出所有表并为每个表生成查询
tables = list_tool.run("").split(", ")
for table in tables:
    print(f"SELECT * FROM {table} LIMIT 5;")
```

---

## **QuerySQLCheckerTool 示例**

### 示例1：验证简单查询
使用`run`方法验证从`Customer`表中获取所有行的简单SQL查询。

```python
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain import hub
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine

# 设置数据库连接
engine = create_engine("sqlite:////kaggle/input/chinook-database/chinook.db")
db = SQLDatabase(engine)

# 初始化SQLDatabaseToolkit（必须先创建工具包）
toolkit = SQLDatabaseToolkit(db=db, llm=model)

# 初始化QuerySQLCheckerTool
checker_tool = QuerySQLCheckerTool(db=db, llm=model)

# 定义SQL查询
query = "SELECT * FROM Customer"

# 使用QuerySQLCheckerTool验证查询
validated_query = checker_tool.run(query)
print("验证后的查询:", validated_query)
```

### 示例2：修复常见错误的查询
使用`run`方法验证并修复带有常见错误的查询（例如字符串缺少引号）。

```python
# 初始化QuerySQLCheckerTool
checker_tool = QuerySQLCheckerTool(db=db, llm=model)

# 定义带有错误的SQL查询
query = "SELECT * FROM Customer WHERE Country = USA"

# 使用QuerySQLCheckerTool验证并修复查询
validated_query = checker_tool.run(query)
print("验证后的查询:", validated_query)
```

### 示例3：验证包含联接的复杂查询
使用`run`方法验证涉及联接和聚合的复杂查询。

```python
# 初始化QuerySQLCheckerTool
checker_tool = QuerySQLCheckerTool(db=db, llm=model)

# 定义SQL查询
query = """
    SELECT c.FirstName, c.LastName, SUM(i.Total) AS TotalSpent
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY c.CustomerId
    ORDER BY TotalSpent DESC
"""

# 使用QuerySQLCheckerTool验证查询
validated_query = checker_tool.run(query)
print("验证后的查询:", validated_query)
```

---

## **QuerySQLDatabaseTool 示例**

### 示例1：聚合数据
使用`run`方法计算每个客户的总销售额。

```python
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine

# 创建SQLAlchemy引擎和SQLDatabase对象
engine = create_engine("sqlite:////kaggle/input/chinook-database/chinook.db")
db = SQLDatabase(engine)

# 初始化工具
query_tool = QuerySQLDatabaseTool(db=db)

# 执行查询
result = query_tool.run("""
    SELECT CustomerId, SUM(Total) AS TotalSpent 
    FROM Invoice 
    GROUP BY CustomerId
""")
print(result)
```

### 示例2：联接表
使用`run`方法获取客户姓名及其发票总额。

```python
# 初始化工具
query_tool = QuerySQLDatabaseTool(db=db)

# 执行查询
result = query_tool.run("""
    SELECT c.FirstName, c.LastName, SUM(i.Total) AS TotalSpent
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY c.CustomerId
""")
print(result)
```

### 示例3：使用`invoke`执行查询
使用`invoke`方法执行查询并从`Employee`表中获取所有行。

```python
# 初始化工具
query_tool = QuerySQLDatabaseTool(db=db)

# 使用invoke执行查询
result = query_tool.invoke("SELECT * FROM Employee")
print(result)
```

### 示例4：使用`batch`执行多个查询
使用`batch`方法在一次调用中执行多个查询。

```python
# 初始化工具
query_tool = QuerySQLDatabaseTool(db=db)

# 使用batch执行多个查询
queries = [
    "SELECT * FROM Customer LIMIT 5",
    "SELECT * FROM Invoice LIMIT 5"
]
results = query_tool.batch(queries)
for result in results:
    print(result)
```

### 示例5：使用`stream`获取增量结果
使用`stream`方法为大型查询增量获取结果。

```python
# 初始化工具
query_tool = QuerySQLDatabaseTool(db=db)

# 使用stream执行查询
for chunk in query_tool.stream("SELECT * FROM Track LIMIT 10"):
    print(chunk)
```

### 示例6：使用`with_config`进行自定义配置
使用`with_config`方法为工具添加元数据或标签。

```python
# 使用自定义配置初始化工具
configured_tool = query_tool.with_config({"tags": ["query_execution"], "metadata": {"purpose": "data_analysis"}})

# 执行查询
result = configured_tool.run("SELECT * FROM Genre")
print(result)
```

### 示例7：使用`with_retry`处理错误
使用`with_retry`方法在查询执行出错时重试。

```python
# 使用重试逻辑初始化工具
retry_tool = query_tool.with_retry(retry_if_exception_type=(Exception,), stop_after_attempt=3)

# 使用invoke执行查询
result = retry_tool.invoke("SELECT * FROM NonExistentTable")  # 失败时将重试
print(result)
```

---

## 最佳实践

### 示例1：使用SQL代理回答业务问题

在此示例中，我们展示如何使用**SQLDatabaseToolkit**和**create_react_agent**通过查询SQL数据库回答业务问题。目标是在`chinook.db`数据库中找到**按总支出排名的前5位客户**。此示例突出显示了将语言模型（LLM）与SQL工具结合使用的强大功能，以自动化复杂的数据检索任务。

#### **示例的关键组件**

1. **SQLDatabaseToolkit**：
   - 该工具包提供`QuerySQLDatabaseTool`、`InfoSQLDatabaseTool`和`QuerySQLCheckerTool`等工具，以便与SQL数据库交互。
   - 它使代理能够生成、验证和执行SQL查询。

2. **create_react_agent**：
   - 该函数创建一个代理，使用LLM和`SQLDatabaseToolkit`提供的工具回答用户问题。
   - 代理遵循推理过程以确定正确的SQL查询并解释结果。

3. **业务问题**：
   - 问题**“按总支出排名的前5位客户是谁？”**要求代理：
     - 识别相关表（`Customer`和`Invoice`）。
     - 在`CustomerId`字段上联接表。
     - 聚合`Invoice`表中的`Total`字段以计算每个客户的总支出。
     - 按总支出降序排序并返回前5位客户。

#### **此示例的用处**

- **自动化**：代理自动编写和执行SQL查询，节省时间并减少错误风险。
- **自然语言界面**：用户可以用简单的英语提出问题，代理将其翻译成SQL查询。
- **可扩展性**：此方法可扩展以回答各种业务问题，例如收入分析、客户细分和库存管理。

```python
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain import hub
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine

# 设置数据库连接
engine = create_engine("sqlite:////kaggle/input/chinook-database/chinook.db")
db = SQLDatabase(engine)

# 初始化SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)

# 拉取SQL代理系统提示
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

# 格式化系统消息
system_message = prompt_template.format(dialect="SQLite", top_k=5)

# 创建代理
agent_executor = create_react_agent(model, tools=toolkit.get_tools(), state_modifier=system_message)

# 示例1：向代理提出业务问题
question = "按总支出排名的前5位客户是谁？"
response = agent_executor.invoke({"messages": [("user", question)]})
print(response["messages"][-1].content)
```

```python
# 示例2：按国家查找总收入
question = "根据售出的曲目数量，哪些是前3位畅销艺术家？"
response = agent_executor.invoke({"messages": [("user", question)]})
print(response["messages"][-1].content)
```

```python
# 示例3：查找最受欢迎的流派
question = "根据售出的曲目数量，哪种音乐流派最受欢迎？"
response = agent_executor.invoke({"messages": [("user", question)]})
print(response["messages"][-1].content)
```

```python
# 示例4：查找销售额最高的员工
question = "哪位员工的总销售额最高？"
response = agent_executor.invoke({"messages": [("user", question)]})
print(response["messages"][-1].content)
```

```python
# 示例5：查找最赚钱的曲目
question = "哪首曲目产生了最高的总收入？"
response = agent_executor.invoke({"messages": [("user", question)]})
print(response["messages"][-1].content)
```

### 示例2：验证并执行SQL查询

在此示例中，我们展示如何使用**QuerySQLCheckerTool**和**QuerySQLDatabaseTool**验证并执行SQL查询。目标是在`chinook.db`数据库中计算**每个国家的总销售额**。此示例强调了在执行前确保SQL查询正确性的重要性，并展示了LangChain工具如何简化此过程。

#### **示例的关键组件**

1. **QuerySQLCheckerTool**：
   - 该工具使用语言模型（LLM）验证SQL查询的正确性。
   - 它检查常见错误，例如语法错误、缺少引号或错误的表/列引用。
   - 如果查询无效，它会建议更正或重写查询。

2. **QuerySQLDatabaseTool**：
   - 该工具对数据库执行经过验证的SQL查询并返回结果。
   - 它确保仅执行正确且安全的查询，降低错误或数据损坏的风险。

3. **SQL查询**：
   - 查询**"SELECT c.Country, SUM(i.Total) AS TotalSales FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.Country ORDER BY TotalSales DESC"**执行以下操作：
     - 在`CustomerId`字段上联接`Customer`和`Invoice`表。
     - 按`Country`分组结果。
     - 为每个国家计算总销售额（`SUM(i.Total)`）。
     - 按总销售额降序排序结果。

#### **此示例的用处**

- **错误预防**：`QuerySQLCheckerTool`确保SQL查询在执行前正确，避免运行时错误和数据不一致。
- **效率**：通过自动化查询验证和执行，此方法节省时间并减少手动干预的需要。
- **灵活性**：这些工具可以处理从简单SELECT语句到复杂联接和聚合的各种SQL查询。

```python
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool, QuerySQLDatabaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
import os

# 设置数据库连接
engine = create_engine("sqlite:////kaggle/input/chinook-database/chinook.db")
db = SQLDatabase(engine)

# 初始化QuerySQLCheckerTool和QuerySQLDatabaseTool
checker_tool = QuerySQLCheckerTool(db=db, llm=model)
query_tool = QuerySQLDatabaseTool(db=db)

# 定义SQL查询
query = """
    SELECT c.Country, SUM(i.Total) AS TotalSales
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY c.Country
    ORDER BY TotalSales DESC
"""

# 使用QuerySQLCheckerTool验证查询
validated_query = checker_tool.run(query)
print("验证后的查询:", validated_query)

# 使用QuerySQLDatabaseTool执行经过验证的查询
result = query_tool.run(validated_query)
print("查询结果:", result)
```

---

## **结论**  
**InfoSQLDatabaseTool**、**ListSQLDatabaseTool**、**QuerySQLCheckerTool** 和 **QuerySQLDatabaseTool** 是LangChain生态系统中与SQL数据库交互不可或缺的组件。每种工具都针对特定需求，从探索数据库结构到验证和执行查询。  

通过使用这些工具，开发人员和数据专业人士可以简化工作流程，减少错误，并深入了解数据。无论您是构建数据分析管道、开发具有数据库访问权限的聊天机器人，还是简单地探索数据集，这些工具都提供了与SQL数据库有效工作所需的灵活性和可靠性。  

随着对智能高效数据处理的需求不断增长，掌握这些工具将使您能够构建更健壮和可扩展的应用程序。从今天开始将它们集成到您的项目中，释放SQL数据库的全部潜力。