# 使用LangGraph为自定义库构建的AI编码器工作流程

## 引言

本笔记本展示了如何使用**LangGraph**、**LangChain**和**Pydantic**为您的自定义库构建一个**AI驱动的代码生成工作流程**。该工作流程旨在处理用户查询，生成可靠的Python代码解决方案，并通过错误处理和反思迭代改进输出。通过利用LangGraph，系统将代码生成的不同阶段连接成一个动态且可重用的工作流程。这对于自动化与自定义库相关的开发任务特别有用，确保生成准确且可执行的解决方案。

以下是流程的概要：

1. **导入所需库**：设置必要的库和依赖项。
2. **常量和配置**：定义常量，如模型名称、最大迭代次数和文档URL。
3. **加载自定义库文档**：实现一个函数，从指定URL获取并整合库文档，确保工作流程所需的所有信息都可访问，使用递归加载器检索文档的多个页面或部分。
4. **定义数据模型**：使用Pydantic创建数据模型，以结构化工作流程的状态和代码解决方案。
5. **初始化语言模型**：设置用于代码生成的语言模型（LLM）。
6. **创建提示模板**：设计提示，以指导LLM生成结构化的代码解决方案。
7. **定义工作流程节点**：实现处理代码生成、代码检查和错误反思的函数。
8. **构建和编译工作流程**：组装工作流程图并编译以供执行。
9. **执行示例查询**：通过示例用户问题展示工作流程，并显示生成的代码解决方案。
10. **结论**：总结工作流程及其功能。

## 安装所需包

首先，我们需要安装层次代理系统所需的必要包。这些包包括LangChain和LangGraph的各种组件。

```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU langgraph
```

## 导入所需库

首先，我们导入工作流程所需的所有必要库和模块，包括用于网页抓取、语言模型交互、数据建模和工作流程管理的库。

```python
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import List, Optional
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import SystemMessage, HumanMessage
from kaggle_secrets import UserSecretsClient
```

## 常量和配置

本节定义了整个工作流程中使用的关键常量，例如模型名称、代码生成尝试的最大迭代次数和您的私有库文档的URL（此处使用LCEL文档）。

```python
# 代码生成尝试的最大迭代次数
MAX_ITERATIONS = 3

# 您的私有库的URL（此处使用LCEL文档）
LCEL_DOCS_URL = "https://python.langchain.com/docs/concepts/lcel/"
```

## 加载自定义库文档

本节提供了一个函数，用于从指定URL加载并合并自定义库文档（此处以LCEL文档为例）。使用递归URL加载器来获取和解析内容，确保包含所有相关文档页面。

```python
# 加载LCEL文档
def load_lcel_docs(url: str) -> str:
    """
    从给定URL加载并拼接LCEL文档。

    参数:
        url (str): 加载文档的URL。

    返回:
        str: 所有文档页面的拼接内容。
    """
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    # 按源对文档进行逆序排序，以确保一致的顺序
    sorted_docs = sorted(docs, key=lambda x: x.metadata["source"], reverse=True)
    # 使用分隔符连接所有文档内容
    return "\n\n\n --- \n\n\n".join(doc.page_content for doc in sorted_docs)
```

## 定义数据模型

使用Pydantic，我们定义数据模型以结构化工作流程的状态和生成的代码解决方案，确保数据处理中的类型安全和清晰度。

### CodeSolution模型

表示代码解决方案的结构，包括描述（`prefix`）、导入语句（`imports`）和主代码块（`code`）。

```python
# 代码解决方案的数据模型
class CodeSolution(BaseModel):
    """
    关于LCEL问题的代码解决方案的模式。

    属性:
        prefix (str): 问题和方法的描述。
        imports (str): 包含导入语句的代码块。
        code (str): 不包括导入语句的代码块。
    """
    prefix: str = Field(description="问题和方法的描述")
    imports: str = Field(description="代码块导入语句")
    code: str = Field(description="不包括导入语句的代码块")
```

### GraphState模型

表示工作流程图的状态，跟踪错误、消息、生成的代码解决方案和迭代次数。

```python
# 使用Pydantic定义图状态
class GraphState(BaseModel):
    """
    表示图的状态。

    属性:
        error (str): 指示是否发生错误（'yes'或'no'）。
        messages (List): 消息列表（用户问题、错误消息等）。
        generation (Optional[CodeSolution]): 生成的代码解决方案。
        iterations (int): 已尝试的次数。
    """
    error: str = Field(default="no", description="'yes'或'no'指示是否发生错误")
    messages: List = Field(default_factory=list, description="消息列表（用户问题、错误消息等）")
    generation: Optional[CodeSolution] = Field(default=None, description="生成的代码解决方案")
    iterations: int = Field(default=0, description="已尝试的次数")
```

## 初始化语言模型

我们初始化用于生成代码解决方案的语言学习模型（LLM）。在此案例中，我们使用DeepSeek的`deepseek-chat`模型，但也可以配置Anthropic的Claude或OpenAI的GPT-4等替代方案。

```python
user_secrets = UserSecretsClient()

# 初始化LLM
# Anthropic
#llm = ChatAnthropic(temperature=0, model="claude-3-5-sonnet-latest", api_key=user_secrets.get_secret("my-anthropic-api-key"))

# OpenAI
# llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=user_secrets.get_secret("my-openai-api-key"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://20.243.34.136:2999/v1",
                        openai_api_key="sk-j8r3Pxztstd3wBjF8fEe44E63f69486bAdC2C4562bD1E1F3")

# DeepSeek-V3
#llm = ChatOpenAI(temperature=0, model="deepseek-chat", api_key=user_secrets.get_secret("my-deepseek-api-key"),
#                 base_url="https://api.deepseek.com/v1")
```

## 创建提示模板

我们创建一个提示模板，指导LLM生成结构化的代码解决方案。提示包括系统消息和上下文及用户消息的占位符。

```python
# 代码生成的提示模板
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            """您是一个在LCEL（LangChain表达式语言）方面具有专长的编码助手。
以下是LCEL文档的完整集合：
------------------------------------------
{context}
------------------------------------------
根据上面提供的文档回答用户问题。确保您提供的任何代码都可以执行，包括所有必需的导入和定义的变量。以代码解决方案的描述结构化您的回答。
然后列出导入语句。最后列出可运行的代码块。以下是用户问题："""
        ),
        ("placeholder", "{messages}"),
    ]
)
```

## 定义工作流程节点

工作流程节点是表示工作流程中不同阶段的函数。在此，我们定义了三个主要节点：`gen_code_node`、`check_code_node`和`reflect_code_node`。

### 代码生成节点

根据当前状态生成代码解决方案，使用适当的上下文和消息调用LLM。

```python
# 节点
def gen_code_node(state: GraphState) -> GraphState:
    """
    根据当前状态生成代码解决方案。

    参数:
        state (GraphState): 图的当前状态。

    返回:
        GraphState: 更新了生成代码解决方案的状态。
    """
    print("---生成代码解决方案---")
    messages = state.messages
    iterations = state.iterations
    error = state.error

    # 如果发生错误，添加重试消息
    if error == "yes":
        messages.append(HumanMessage("现在，再试一次。调用代码工具以结构化输出，包括前缀、导入和代码块。"))

    # 使用代码生成链生成代码解决方案
    code_solution = code_gen_chain.invoke({"context": concatenated_content, "messages": messages})
    # 将生成解决方案附加到消息中
    messages.append(HumanMessage(f"{code_solution.prefix} \n 导入: {code_solution.imports} \n 代码: {code_solution.code}"))

    # 增加迭代计数并返回更新状态
    return GraphState(
        error="no",
        messages=messages,
        generation=code_solution,
        iterations=iterations + 1,
    )
```

### 代码检查节点

通过尝试执行导入语句和主代码块检查生成的代码是否存在错误。根据这些执行的成功或失败更新状态。

```python
def check_code_node(state: GraphState) -> GraphState:
    """
    检查生成的代码是否存在错误。

    参数:
        state (GraphState): 图的当前状态。

    返回:
        GraphState: 更新状态，指示代码是否通过检查。
    """
    print("---检查代码---")
    messages = state.messages
    code_solution = state.generation
    iterations = state.iterations

    # 通过尝试执行导入语句检查导入
    try:
        exec(code_solution.imports)
    except Exception as e:
        print("---代码导入检查：失败---")
        messages.append(HumanMessage(f"您的解决方案未通过导入测试：{e}"))
        return GraphState(
            error="yes",
            messages=messages,
            generation=code_solution,
            iterations=iterations,
        )

    # 通过尝试运行完整代码（导入+代码）检查执行
    try:
        exec(code_solution.imports + "\n" + code_solution.code)
    except Exception as e:
        print("---代码块检查：失败---")
        messages.append(HumanMessage(f"您的解决方案未通过代码执行测试：{e}"))
        return GraphState(
            error="yes",
            messages=messages,
            generation=code_solution,
            iterations=iterations,
        )

    # 如果没有错误，返回错误设置为'no'的状态
    print("---无代码测试失败---")
    return GraphState(
        error="no",
        messages=messages,
        generation=code_solution,
        iterations=iterations,
    )
```

### 反思节点

反思代码生成或执行过程中遇到的任何错误，提供改进的见解。

```python
def reflect_code_node(state: GraphState) -> GraphState:
    """
    反思错误并提供改进见解。

    参数:
        state (GraphState): 图的当前状态。

    返回:
        GraphState: 更新了错误反思的状态。
    """
    print("---反思错误---")
    messages = state.messages
    code_solution = state.generation

    # 使用代码生成链生成反思
    reflections = code_gen_chain.invoke({"context": concatenated_content, "messages": messages})
    messages.append(HumanMessage(f"以下是错误的反思：{reflections}"))

    return GraphState(
        error="yes",
        messages=messages,
        generation=code_solution,
        iterations=state.iterations,
    )
```

## 构建和编译工作流程

我们通过添加定义的节点并根据工作流程状态指定它们之间的转换来构建工作流程图。工作流程从代码生成开始，然后检查代码，并在重试之前可选地反思错误。

```python
# 代码生成链
code_gen_chain = code_gen_prompt | llm.with_structured_output(CodeSolution)

# 边缘
def decide_to_finish(state: GraphState) -> str:
    """
    根据状态确定是完成还是重试。

    参数:
        state (GraphState): 图的当前状态。

    返回:
        str: 决定是完成还是重试代码生成过程。
    """
    error = state.error
    iterations = state.iterations

    # 如果没有错误或达到最大迭代次数，则完成
    if error == "no" or iterations == MAX_ITERATIONS:
        print("---决定：完成---")
        return "end"
    else:
        # 否则，根据错误决定重试或反思
        print("---决定：重试解决方案---")
        return "reflect_code_node" if error == "yes" else "gen_code_node"

# 构建和编译工作流程
workflow = StateGraph(GraphState)
workflow.add_node("gen_code_node", gen_code_node)  # 添加代码生成节点
workflow.add_node("check_code_node", check_code_node)  # 添加代码检查节点
workflow.add_node("reflect_code_node", reflect_code_node)  # 添加反思节点

workflow.add_edge(START, "gen_code_node")  # 从代码生成开始
workflow.add_edge("gen_code_node", "check_code_node")  # 检查生成的代码
workflow.add_conditional_edges(
    "check_code_node",
    decide_to_finish,
    {"end": END, "reflect_code_node": "reflect_code_node", "gen_code_node": "gen_code_node"},
)
workflow.add_edge("reflect_code_node", "gen_code_node")  # 反思后重试
app = workflow.compile()  # 编译工作流程
```

### 可选：显示工作流程图

如果需要，可以可视化工作流程图。这需要额外的依赖项，是可选的。

```python
# 可选：显示工作流程图（需要额外依赖项）
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # 这需要一些额外的依赖项，是可选的
    pass
```

## 加载LCEL文档内容

我们使用之前定义的`load_lcel_docs`函数加载LCEL文档内容。此内容作为LLM生成代码解决方案时的上下文。

```python
# 加载LCEL文档
concatenated_content = load_lcel_docs(LCEL_DOCS_URL)
```

## 提取和打印代码的函数

定义了一个实用函数，从工作流程的解决方案中提取生成的Python代码并以可读格式打印。

```python
# 提取和打印代码的函数
def extract_and_print_code(solution: dict) -> None:
    """
    从解决方案中提取Python代码并打印。

    参数:
        solution (dict): 工作流程返回的解决方案字典。
    """
    # 将解决方案字典转换为GraphState对象
    graph_state = GraphState(**solution)

    if graph_state.generation is None:
        print("解决方案中未找到代码生成。")
        return

    # 从CodeSolution对象中提取代码
    code_solution = graph_state.generation
    print()
    print("# " + "-"*80)
    print("# 提取的Python代码")
    print("# " + "-"*80)
    print(code_solution.code)
```

## 执行示例查询

我们通过执行几个示例用户查询来展示工作流程的功能。对于每个查询，我们初始化工作流程的状态，调用工作流程，并显示生成的代码解决方案。

```python
# 示例用法1
question = "如何直接将字符串传递给可运行对象，并使用它构建我的提示所需的输入？"

initial_state = GraphState(
    messages=[HumanMessage(question)],
    iterations=0,
    error="no",
    generation=None,
)
solution = app.invoke(initial_state)

# 输出解决方案
extract_and_print_code(solution)
```

```python
# 示例用法2
question = """如何创建一个简单的LCEL链，接受字符串输入，将其转换为大写，
然后将文本' - Processed'附加到结果中？使用管道操作符链接步骤。"""

initial_state = GraphState(
    messages=[HumanMessage(question)],
    iterations=0,
    error="no",
    generation=None,
)
solution = app.invoke(initial_state)

# 输出解决方案
extract_and_print_code(solution)
```

```python
# 示例用法3
question = """如何创建一个LCEL链，根据条件将输入路由到两个可运行对象之一？
例如，如果输入是大于10的数字，则将其路由到将其乘以2的可运行对象；
否则，路由到将其加5的可运行对象。"""

initial_state = GraphState(
    messages=[HumanMessage(question)],
    iterations=0,
    error="no",
    generation=None,
)
solution = app.invoke(initial_state)

# 输出解决方案
extract_and_print_code(solution)
```

```python
# 示例用法4
question = """如何创建一个LCEL链，接受带有name和age键的字典，
格式化一个像'Name: {name}, Age: {age}'的字符串，然后将结果转换为大写？
使用管道操作符链接步骤。"""

initial_state = GraphState(
    messages=[HumanMessage(question)],
    iterations=0,
    error="no",
    generation=None,
)
solution = app.invoke(initial_state)

# 输出解决方案
extract_and_print_code(solution)
```

## 结论

在本笔记本中，我们为您的自定义库构建了一个强大的**AI编码器工作流程**，使用了**LangGraph**。此工作流程通过结构化提示、错误处理和迭代增强，自动化了解释用户查询、生成Python代码和优化解决方案的过程。通过将LangGraph的模块化工作流程设计与全面的库文档集成，此系统确保生成的代码既准确又可执行。

这种方法不仅简化了开发任务，还为扩展编码过程中的自动化提供了灵活的基础。有了这个工作流程，开发人员可以自信地应对复杂的编码需求，同时保持自定义库所需的可靠性和精确性。