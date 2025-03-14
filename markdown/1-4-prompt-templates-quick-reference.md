# **LangChain 中的提示模板参考**

## **介绍**

在会话式 AI 和语言模型的领域中，制作有效的提示语是一项关键技能。无论您是构建聊天机器人、虚拟助手还是交互式应用程序，动态生成和管理提示语的能力都可以显著提升用户体验。本文探讨了 LangChain 框架中 **PromptTemplate** 和 **ChatPromptTemplate** 的强大功能，以及多功能的 **MessagesPlaceholder**，以创建适用于各种用例的灵活且动态的提示语。

从简单的文本补全到多轮对话，这些工具使开发人员能够轻松地构建提示语、注入动态内容和管理对话历史记录。通过利用这些模板，您可以构建强大且交互式的 AI 系统，这些系统可以适应用户输入并保持跨交互的上下文。本文提供了按用例分类的 **15 个实用示例**，演示了如何在实际场景中有效地使用这些工具。


```python
!pip install -qU langchain-openai
!pip install -qU langchain-anthropic
!pip install -qU langchain_community
!pip install -qU langchain_experimental
```


```python
import json

def pretty_print(data):
    """
    将字典、列表或对象转换为 JSON 并以漂亮的格式打印。

    Args:
        data: 要漂亮打印的数据（dict、list 或 object）。
    """
    # 如果需要，将对象转换为字典
    if not isinstance(data, (dict, list)):
        try:
            data = vars(data)  # 将对象转换为字典
        except TypeError:
            pass  # 如果 vars() 失败，则按原样处理

    # 转换为 JSON 并打印，带有缩进
    pretty_json = json.dumps(data, indent=4, default=str)  # 使用 default=str 来处理不可序列化的对象
    print(pretty_json)
```

## **1. PromptTemplate 示例**

### **1.1. 简单的文本补全**
使用 `PromptTemplate` 为给定的输入生成补全。


```python
from langchain_core.prompts import PromptTemplate

# 定义一个简单的提示模板
template = PromptTemplate.from_template("写一个关于 {topic} 的短篇故事。")
formatted_prompt = template.format(topic="一个学习绘画的机器人")

print(formatted_prompt)
```

---

## **2. ChatPromptTemplate 示例**

### **2.1. 单轮聊天**
使用 `SystemMessage`、`HumanMessage` 和 `AIMessage` 创建一个单轮聊天提示。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的 AI 助手。"),
    HumanMessage(content="给我讲一个关于 {topic} 的笑话。"),
])

prompt_value = template.invoke({"topic": "程序员"})
pretty_print(prompt_value.messages)
```

### **2.2. 多轮聊天**
使用 `SystemMessage`、`HumanMessage` 和 `AIMessage` 模拟多轮对话。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个多轮聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的 AI 助手。"),
    HumanMessage(content="嗨，我叫 {name}。"),
    AIMessage(content="你好，{name}！今天有什么可以帮你的吗？"),
    HumanMessage(content="我需要 {task} 方面的帮助。"),
])

prompt_value = template.invoke({"name": "Alice", "task": "写简历"})
pretty_print(prompt_value.messages)
```

### **2.3. 预填充变量**
使用 `partial` 在聊天模板中预填充系统指令。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# 定义一个带有部分变量的聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个 {role}。"),
    HumanMessage(content="{user_input}"),
])
partial_template = template.partial(role="数学辅导老师")

prompt_value = partial_template.invoke({"user_input": "解释一下勾股定理。"})
pretty_print(prompt_value.messages)
```

### **2.4. 角色扮演**
使用 `SystemMessage`、`HumanMessage` 和 `AIMessage` 模拟角色扮演场景。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# 定义一个角色扮演聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个海盗。像海盗一样说话。"),
    HumanMessage(content="告诉我你的冒险经历。"),
])

prompt_value = template.invoke({})
pretty_print(prompt_value.messages)
```

### **2.5. 与聊天模型集成**
将聊天提示模板与聊天模型一起使用。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from kaggle_secrets import UserSecretsClient

# 检索 OpenAI API 密钥
llm_api_key = UserSecretsClient().get_secret("api-key-openai")

# 定义一个聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的 AI 助手。"),
    HumanMessage(content="用简单的术语解释 {concept}。"),
])

# 与聊天模型集成
chat_model = ChatOpenAI(model="gpt-4o-mini", api_key=llm_api_key)
prompt_value = template.invoke({"concept": "量子计算"})

response = chat_model.invoke(prompt_value.messages)
pretty_print(response)
```

### **2.6. 多角色对话**
使用 `SystemMessage`、`HumanMessage` 和 `AIMessage` 模拟具有多个角色的对话。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个多角色聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的 AI 助手。"),
    HumanMessage(content="嗨，我叫 {name}。"),
    AIMessage(content="你好，{name}！我是来帮忙的。"),
    HumanMessage(content="你能向我解释一下 {topic} 吗？"),
    AIMessage(content="当然！这是 {topic} 的解释："),
])

prompt_value = template.invoke({"name": "Alice", "topic": "黑洞"})
pretty_print(prompt_value.messages)
```

### **2.7. 客户支持**
模拟客户支持对话。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个客户支持聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一名客户支持代理。"),
    HumanMessage(content="嗨，我的 {product} 有问题。"),
    AIMessage(content="很抱歉听到这个消息。你能描述一下这个问题吗？"),
    HumanMessage(content="问题是 {issue}。"),
])

prompt_value = template.invoke({"product": "我的手机", "issue": "它无法开机"})
pretty_print(prompt_value.messages)
```

---

## **3. MessagesPlaceholder 示例**

### **3.1. 注入对话历史记录**
使用 `MessagesPlaceholder` 将对话历史记录注入到聊天提示中。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个带有 MessagesPlaceholder 的聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}"),
])

# 模拟对话历史记录
history = [
    HumanMessage(content="5 + 2 是多少？"),
    AIMessage(content="5 + 2 等于 7。"),
]

# 使用历史记录和一个新问题调用模板
prompt_value = template.invoke({
    "history": history,
    "question": "现在将它乘以 4。"
})

pretty_print(prompt_value.messages)
```

### **3.2. 可选的 MessagesPlaceholder**
使用**可选的** `MessagesPlaceholder` 以允许空的对话历史记录。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# 定义一个带有可选 MessagesPlaceholder 的聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history", optional=True),
    HumanMessage(content="{question}"),
])

# 在不提供历史记录的情况下调用模板
prompt_value = template.invoke({
    "question": "法国的首都是哪里？"
})

pretty_print(prompt_value.messages)
```

### **3.3. 限制消息数量**
使用 `n_messages` 限制注入到提示中的消息数量。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个带有 MessagesPlaceholder 的聊天提示模板，该 MessagesPlaceholder 限制为 2 条消息
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history", n_messages=2),
    HumanMessage(content="{question}"),
])

# 模拟对话历史记录
history = [
    HumanMessage(content="5 + 2 是多少？"),
    AIMessage(content="5 + 2 等于 7。"),
    HumanMessage(content="现在将它乘以 4。"),
    AIMessage(content="7 * 4 等于 28。"),
]

# 使用历史记录和一个新问题调用模板
prompt_value = template.invoke({
    "history": history,
    "question": "它的平方根是多少？"
})

pretty_print(prompt_value.messages)
```

### **3.4. 动态对话上下文**
使用 `MessagesPlaceholder` 将上下文动态注入到对话中。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个带有 MessagesPlaceholder 的聊天提示模板，用于上下文
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{question}"),
])

# 模拟上下文（例如，之前的指令或背景信息）
context = [
    SystemMessage(content="用户是一个学习数学的学生。"),
    HumanMessage(content="我需要代数方面的帮助。"),
]

# 使用上下文和一个新问题调用模板
prompt_value = template.invoke({
    "context": context,
    "question": "我该如何解 2x + 5 = 15？"
})

pretty_print(prompt_value.messages)
```

### **3.5. 具有部分历史记录的多轮对话**
使用 `MessagesPlaceholder` 管理具有部分历史记录的多轮对话。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个带有 MessagesPlaceholder 的聊天提示模板，用于部分历史记录
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}"),
])

# 模拟部分对话历史记录
history = [
    HumanMessage(content="5 + 2 是多少？"),
    AIMessage(content="5 + 2 等于 7。"),
]

# 使用部分历史记录和一个新问题调用模板
prompt_value = template.invoke({
    "history": history,
    "question": "现在从中减去 3。"
})

pretty_print(prompt_value.messages)
```

### **3.6. 组合多个 MessagesPlaceholder**
使用多个 `MessagesPlaceholder` 实例来管理对话的不同部分。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义一个带有多个 MessagesPlaceholder 的聊天提示模板
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="context"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}"),
])

# 模拟上下文和历史记录
context = [
    SystemMessage(content="用户是一个学习数学的学生。"),
]
history = [
    HumanMessage(content="5 + 2 是多少？"),
    AIMessage(content="5 + 2 等于 7。"),
]

# 使用上下文、历史记录和一个新问题调用模板
prompt_value = template.invoke({
    "context": context,
    "history": history,
    "question": "现在将它乘以 4。"
})

pretty_print(prompt_value.messages)
```

## **结论**

本文中的示例展示了 **PromptTemplate**、**ChatPromptTemplate** 和 **MessagesPlaceholder** 在构建动态和交互式 AI 系统中的多功能性和强大功能。无论您是创建简单的文本补全、管理多轮对话，还是将动态上下文注入到提示中，这些工具都提供了一种灵活有效的方式来构建和管理您的提示。

通过掌握这些技术，您可以创建不仅响应迅速而且具有上下文意识的 AI 应用程序，从而为用户提供无缝且引人入胜的体验。从客户支持机器人到角色扮演场景，可能性是无限的。当您探索这些示例时，请考虑如何将它们调整到您的特定用例，并将您的会话 AI 项目提升到一个新的水平。
