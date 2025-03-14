```python
!pip install -qU langchain_community
!pip install -qU langchain_experimental
!pip install -qU grandalf
```

---
#

## **1. 核心组成和链接**
这些示例重点介绍 LCEL 的基本构建块，例如链接和组合 Runnables。
#

### **1.1. 基本 Runnable 组合**
使用管道 (`|`) 运算符将两个简单的函数链接在一起。


```python
from langchain_core.runnables import RunnableLambda

# 定义两个简单的函数
add_five = RunnableLambda(lambda x: x + 5)
multiply_by_two = RunnableLambda(lambda x: x * 2)

# 将它们链接在一起
chain = add_five | multiply_by_two
print(chain.invoke(3))
```

    16


### **1.2. 并行执行**
并行运行多个 Runnables 并组合它们的输出。


```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

# 定义两个函数
add_five = RunnableLambda(lambda x: x + 5)
multiply_by_two = RunnableLambda(lambda x: x * 2)

# 并行运行它们
chain = RunnableParallel(add=add_five, multiply=multiply_by_two)
print(chain.invoke(3))
```

    {'add': 8, 'multiply': 6}


### **1.3. 条件逻辑**
根据输入动态选择要执行的 Runnable。


```python
from langchain_core.runnables import RunnableLambda

# 定义两个函数
add_five = RunnableLambda(lambda x: x + 5)
multiply_by_two = RunnableLambda(lambda x: x * 2)

# 根据输入选择要运行的函数
chain = RunnableLambda(lambda x: add_five if x > 10 else multiply_by_two)
print(chain.invoke(15))
print(chain.invoke(5))
```

    20
    10


### **1.4. 动态链构建**
根据输入动态构建链。


```python
from langchain_core.runnables import RunnableLambda

# 定义函数
add_five = RunnableLambda(lambda x: x + 5)
multiply_by_two = RunnableLambda(lambda x: x * 2)

# 动态构建链
chain = RunnableLambda(lambda x: add_five if x > 10 else multiply_by_two)
print(chain.invoke(15))
print(chain.invoke(5))
```

    20
    10


---
#
## **2. 错误处理和鲁棒性**
这些示例演示如何处理错误并使工作流程更健壮。
#
### **2.1. 回退机制**
提供回退 Runnables，以防主要的 Runnable 失败。

### **2.2. 重试机制**
如果 Runnable 失败，则重试指定的次数。


```python
from langchain_core.runnables import RunnableLambda

# 定义一个在第一次尝试时失败的函数
counter = 0
def faulty_function(x):
    global counter
    counter += 1
    if counter < 2:
        raise ValueError("Failed!")
    return x + 5

# 创建一个带有重试的 Runnable
runnable = RunnableLambda(faulty_function).with_retry(stop_after_attempt=3)
print(runnable.invoke(10))  # (第二次尝试成功)
```

    15


---
#
## **3. 输入和输出操作**
这些示例展示如何在链中操作输入和输出。
#
### **3.1. 并行执行和输入操作**
并行运行多个 Runnables，将输入不变地传递给一个，并修改它以传递给另一个。


```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 创建一个 RunnableParallel 实例
runnable = RunnableParallel(
    passed=RunnablePassthrough(),  # 不变地传递输入
    modified=lambda x: x["num"] + 1  # 将输入的 "num" 键加 1
)

# 使用输入调用 runnable
result = runnable.invoke({"num": 1})
print(result)
```

    {'passed': {'num': 1}, 'modified': 2}


### **3.2. 动态输入修改**
在将输入传递给下一个 Runnable 之前动态修改输入。


```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 定义一个将数字输入加 5 的函数
add_five = RunnableLambda(lambda x: x + 5)

# 动态修改输入并将修改后的输入传递给 add_five
chain = (
    RunnablePassthrough.assign(modified_input=lambda x: x["input"] * 2)  # 将输入乘以 2
    | RunnableLambda(lambda x: add_five.invoke(x["modified_input"]))  # 将修改后的输入加 5
)

# 调用链
result = chain.invoke({"input": 3})
print(result)
```

    11


### **3.3. 输出子集**
提取输出字典的特定子集。


```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 定义一个将数字输入加 5 的函数
add_five = RunnableLambda(lambda x: x + 5)

# 仅从输出中提取 "result" 键
chain = RunnablePassthrough.assign(result=lambda x: add_five.invoke(x["input"])).pick(["result"])

# 使用字典输入调用链
result = chain.invoke({"input": 10})
print(result)
```

    {'result': 15}


### **3.4. 嵌套输入操作**
修改输入并将新键添加到输出字典。


```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 创建一个 RunnableParallel 实例
runnable = RunnableParallel(
    extra=RunnablePassthrough.assign(multi=lambda x: x["num"] * 3),  # 添加一个新键 'multi'
    modified=lambda x: x["num"] + 1  # 将输入的 "num" 键加 1
)

# 使用输入调用 runnable
result = runnable.invoke({"num": 1})
print(result)
```

    {'extra': {'num': 1, 'multi': 3}, 'modified': 2}


---
#
## **4. 批量和流式传输**
这些示例重点介绍处理多个输入或流式传输输出。
#
### **4.1. 流式传输输出**
从 Runnable 中流式传输结果，因为它们是生成的。


```python
from langchain_core.runnables import RunnableLambda

# 定义一个递增地产生结果的函数
def generate_numbers(x):
    for i in range(x):
        yield i

# 创建一个 Runnable
runnable = RunnableLambda(generate_numbers)

# 流式传输结果
for chunk in runnable.stream(5):
    print(chunk)
```

    0
    1
    2
    3
    4


### **4.2. 批量处理**
并行处理一批输入。


```python
from langchain_core.runnables import RunnableLambda

# 定义一个函数
add_five = RunnableLambda(lambda x: x + 5)

# 处理一批输入
results = add_five.batch([1, 2, 3])
print(results)
```

    [6, 7, 8]


### **4.3. 事件流式传输**
从 Runnable 的执行中流式传输事件。


```python
import nest_asyncio
nest_asyncio.apply()

from langchain_core.runnables import RunnableLambda

# 定义一个函数
async def generate_numbers(x):
    for i in range(x):
        yield i

# 创建一个 Runnable
runnable = RunnableLambda(generate_numbers)

# 流式传输事件
async for event in runnable.astream_events(5, version="v2"):
    print(event)
```

    {'event': 'on_chain_start', 'data': {'input': 5}, 'name': 'generate_numbers', 'tags': [], 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'metadata': {}, 'parent_ids': []}
    {'event': 'on_chain_stream', 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'name': 'generate_numbers', 'tags': [], 'metadata': {}, 'data': {'chunk': 0}, 'parent_ids': []}
    {'event': 'on_chain_stream', 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'name': 'generate_numbers', 'tags': [], 'metadata': {}, 'data': {'chunk': 1}, 'parent_ids': []}
    {'event': 'on_chain_stream', 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'name': 'generate_numbers', 'tags': [], 'metadata': {}, 'data': {'chunk': 2}, 'parent_ids': []}
    {'event': 'on_chain_stream', 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'name': 'generate_numbers', 'tags': [], 'metadata': {}, 'data': {'chunk': 3}, 'parent_ids': []}
    {'event': 'on_chain_stream', 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'name': 'generate_numbers', 'tags': [], 'metadata': {}, 'data': {'chunk': 4}, 'parent_ids': []}
    {'event': 'on_chain_end', 'data': {'output': 10}, 'run_id': 'b71fc9f8-f828-460b-827d-e5f3f15baa95', 'name': 'generate_numbers', 'tags': [], 'metadata': {}, 'parent_ids': []}


---
#
## **5. 配置和自定义**
这些示例突出显示如何配置和自定义 Runnables。
#
### **5.1. 可配置的 Runnables**
使 Runnable 在运行时可配置。


```python
from langchain_core.runnables import RunnableLambda, ConfigurableField

# 定义一个可配置的函数
def configurable_function(x, multiplier=1):
    return x * multiplier

# 创建一个带有包装函数的 RunnableLambda
def create_runnable(multiplier=1):
    return RunnableLambda(lambda x: configurable_function(x, multiplier))

# 使 Runnable 可配置
runnable = RunnableLambda(lambda x: create_runnable(x["multiplier"]).invoke(x["input"]))

# 使用配置调用
result = runnable.invoke({"input": 5, "multiplier": 3})
print(result)  # 输出：15
```

    15



```python
import time
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run

# 定义一个函数
def on_start(run_obj: Run):
    print(f"Started at: {run_obj.start_time}")

def on_end(run_obj: Run):
    print(f"Ended at: {run_obj.end_time}")

# 创建一个带有监听器的 Runnable
runnable = RunnableLambda(lambda x: time.sleep(x)).with_listeners(on_start=on_start, on_end=on_end)
print(runnable.invoke(2))  # 输出：打印开始和结束时间
```

    Started at: 2025-02-25 16:01:16.499425+00:00
    Ended at: 2025-02-25 16:01:18.503105+00:00
    None


---
#
## **6. 可视化和调试**
这些示例有助于可视化和调试工作流程。
#
### **6.1. 图形表示**
可视化 Runnable 链的结构。


```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

# 定义函数
add_five = RunnableLambda(lambda x: x + 5)
multiply_by_two = RunnableLambda(lambda x: x * 2)

# 创建一个链
chain = add_five | RunnableParallel(add=add_five, multiply=multiply_by_two)

# 打印图形
chain.get_graph().print_ascii()
```

            +-------------+          
            | LambdaInput |          
            +-------------+          
                    *                
                    *                
                    *                
               +--------+            
               | Lambda |            
               +--------+            
                    *                
                    *                
                    *                
    +-----------------------------+  
    | Parallel<add,multiply>Input |  
    +-----------------------------+  
               *         *           
             **           **         
            *               *        
     +--------+          +--------+  
     | Lambda |          | Lambda |  
     +--------+          +--------+  
               *         *           
                **     **            
                  *   *              
    +------------------------------+ 
    | Parallel<add,multiply>Output | 
    +------------------------------+ 


---
#
## **7. 提示和 LLM 集成**
这些示例重点介绍将提示和 LLM 集成到工作流程中。
#
### **7.1. 提示模板**
使用 LCEL 链接提示和 LLM 调用。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# 定义一个提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("human", "给我讲一个关于 {topic} 的笑话。"),
])

# 定义一个虚假的 LLM
fake_llm = RunnableLambda(lambda prompt: "为什么鸡要过马路？为了到马路对面！")

# 链接提示和 LLM
chain = prompt | fake_llm
result = chain.invoke({"topic": "chickens"})  # 输出："为什么鸡要过马路？为了到马路对面！"
print(result)
```

    为什么鸡要过马路？为了到马路对面！


### **7.2. 多步骤提示链接**
将多个提示链接在一起以创建更复杂的工作流程。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# 定义第一个提示模板
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("human", "生成一个关于 {topic} 的短篇故事。"),
])

# 定义第二个提示模板
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("human", "用一句话概括以下故事：{story}"),
])

# 定义一个虚假的 LLM
fake_llm = RunnableLambda(lambda prompt: "从前，有一个勇敢的骑士，他从一条龙手中拯救了一个村庄。")

# 链接提示和 LLM
chain = (
    prompt1  # 生成一个故事
    | fake_llm  # 使用虚假的 LLM 生成故事
    | {"story": RunnablePassthrough()}  # 将故事传递到下一步
    | prompt2  # 概括故事
    | fake_llm  # 使用虚假的 LLM 进行概括
)

# 调用链
result = chain.invoke({"topic": "a brave knight"})
print(result)
```

    从前，有一个勇敢的骑士，他从一条龙手中拯救了一个村庄。


## **结论**
#
本备忘单中提供的示例演示了 **LangChain Expression Language (LCEL)** 在构建复杂工作流程方面的多功能性和强大功能。从基本组成和链接到错误处理、动态输入操作和提示模板等高级功能，LCEL 提供了一种灵活且直观的方式来创建强大且可扩展的应用程序。以下是每个类别的关键要点的总结：
#
1. **核心组成和链接**：
   - LCEL 允许您使用管道 (`|`) 运算符链接 Runnables，从而实现函数的无缝组合。
   - 使用 `RunnableParallel` 的并行执行使您可以同时运行多个 Runnables 并组合它们的输出。
   - 条件逻辑和动态链构建使工作流程能够根据输入条件进行调整。
   #
2. **错误处理和鲁棒性**：
   - 回退机制和重试逻辑确保工作流程可以优雅地处理错误并从故障中恢复。
   #
3. **输入和输出操作**：
   - 动态输入修改和子集提取使您可以操作数据，因为它在链中流动。
   - 嵌套输入操作演示了如何在复杂的工作流程中添加新键或修改输入。
   #
4. **批量和流式传输**：
   - 批量处理和流式传输功能使您可以轻松处理多个输入或递增地处理数据。
   - 事件流式传输提供了对 Runnables 执行的可视性，从而可以更好地调试和监视。
   #
5. **配置和自定义**：
   - 可配置的 Runnables 允许您在运行时自定义行为，从而使工作流程更具适应性。
   - 生命周期监听器可以跟踪执行开始和结束时间，从而为您的工作流程添加可观察性。
   #
6. **可视化和调试**：
   - 图形表示工具可帮助可视化 Runnable 链的结构，从而更容易理解和调试复杂的工作流程。
   #
7. **提示和 LLM 集成**：
   - 提示模板和多步骤提示链接允许您动态地将 LLM 集成到工作流程中。
   - 示例演示了如何根据输入条件生成和格式化提示，从而实现与 LLM 的自适应交互。
   #
   这些示例突出了 LCEL 的灵活性和强大功能，使其成为构建复杂的 LangChain 应用程序的开发人员的重要工具。无论您是创建简单的管道还是复杂的工作流程，LCEL 都提供了构建块，可以高效且有效地实现它。通过掌握这些概念，您可以释放 LangChain 的全部潜力并构建既强大又可扩展的应用程序。
