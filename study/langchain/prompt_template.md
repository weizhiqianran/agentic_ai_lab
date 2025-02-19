# An Introduction to LangChain's PromptTemplate

LangChain offers a powerful suite of prompt template classes to help developers build dynamic and robust prompts for language models. In this article, we introduce the core classes and explain their functionalities, differences, and practical usage with examples.

---

### Table of Contents

1. [Introduction](#introduction)
2. [Overview of PromptTemplate Classes](#overview-of-prompttemplate-classes)
   - [PromptTemplate](#prompttemplate)
   - [FewShotPromptTemplate](#fewshotprompttemplate)
   - [ChatPromptTemplate](#chatprompttemplate)
   - [FewShotChatMessagePromptTemplate](#fewshotchatmessageprompttemplate)
   - [MessagesPlaceholder](#messagesplaceholder)
3. [Comparison Table](#comparison-table)
4. [Practical Examples](#practical-examples)
   - [Example 1: Dynamic Survey Question Generator](#example-1-dynamic-survey-question-generator)
   - [Example 2: Case Study Analysis with Examples](#example-2-case-study-analysis-with-examples)
   - [Example 3: Customer Support Chat Template](#example-3-customer-support-chat-template)
   - [Example 4: Contextual Conversation Handler](#example-4-contextual-conversation-handler)
   - [Example 5: Dynamic Survey Question Generator with Few-Shot](#example-5-dynamic-survey-question-generator-with-few-shot)
5. [Best Practices](#best-practices)
6. [Conclusion](#conclusion)

---

## Introduction

In today's rapidly evolving digital landscape, the way we interact with language models is becoming increasingly sophisticated. LangChain offers a robust suite of prompt templating classes that streamline the process of building dynamic, context-aware prompts. Whether you're generating customized survey questions, analyzing business cases, or managing multi-turn conversations in a chatbot, LangChain provides the tools to structure and manage your prompts efficiently.

This guide will introduce you to the core components of LangChain's prompt system—**PromptTemplate**, **FewShotPromptTemplate**, **ChatPromptTemplate**, **FewShotChatMessagePromptTemplate**, and **MessagesPlaceholder**—and demonstrate how they can be applied to solve real-world problems. By leveraging these templates, you can create secure, modular, and adaptive prompts that enhance the performance and usability of your language model applications.

---

## Overview of PromptTemplate Classes

Below is an in-depth look at the major prompt template classes provided by LangChain. For each class, we introduce the key member functions and provide at least two examples—complete with code and explanations—to illustrate their usage.

### PromptTemplate

The **PromptTemplate** class is the foundational component for creating string-based prompts. It allows you to define a template with placeholders and then generate a complete prompt by filling in these placeholders.

#### Major Member Functions

- **`format(**kwargs)`**  
  Formats the prompt by substituting the provided variables into the template.

- **`from_template(template: str, ...)`**  
  A class method that instantiates a `PromptTemplate` directly from a template string, automatically inferring input variables.

- **`__add__(other)`**  
  Combines two `PromptTemplate` instances (only supported for f-string formatted templates) or concatenates with a string.

---

#### Example 1: Using `format()`

Create a basic prompt by filling in placeholders using the `format()` method.

```python
from langchain_core.prompts import PromptTemplate

# Create a prompt template using f-string format.
prompt = PromptTemplate.from_template("Hello, {name}! Welcome to {place}.")
formatted_prompt = prompt.format(name="Alice", place="Wonderland")
print(formatted_prompt)
```

**Output:**
```
Hello, Alice! Welcome to Wonderland.
```

**Explanation:**  
This example creates a prompt template with placeholders for `name` and `place`. The `format()` method is then used to substitute these placeholders with "Alice" and "Wonderland", resulting in the output:  
`"Hello, Alice! Welcome to Wonderland."`

---

#### Example 2: Combining Templates with `__add__`

Combine two prompt templates using the `+` operator to form a single, concatenated prompt.

```python
from langchain_core.prompts import PromptTemplate

# Define two prompt templates.
greeting_prompt = PromptTemplate.from_template("Hello, {name}!")
followup_prompt = PromptTemplate.from_template(" How are you feeling today?")

# Combine the two templates.
combined_prompt = greeting_prompt + followup_prompt
formatted_combined = combined_prompt.format(name="Bob")
print(formatted_combined)
```

**Output:**
```
Hello, Bob! How are you feeling today?
```

**Explanation:**  
This example demonstrates how to combine two `PromptTemplate` instances. The first template greets the user, and the second asks a follow-up question. When combined and formatted with the variable `name` set to "Bob", the output becomes:  
`"Hello, Bob! How are you feeling today?"`

---

#### Example 3: Using Partial Variables

This example demonstrates how to create a `PromptTemplate` with partial variables. Partial variables allow you to pre-fill certain placeholders in the template so that they do not need to be provided every time you format the template. In this case, the `greeting` variable is set to `"Hello"` when the template is created. Later, when the `format()` method is called, you only need to supply the remaining variable (`name`).

```python
from langchain_core.prompts import PromptTemplate

# Create a prompt template with partial variables
template = PromptTemplate.from_template("{greeting} {name}", partial_variables={"greeting": "Hello"})
formatted_prompt = template.format(name="Alice")

print(formatted_prompt)
```

**Output:**
```
Hello Alice
```

**Explanation:**  
In this example, the `PromptTemplate` is instantiated with a template string containing two placeholders: `{greeting}` and `{name}`. By providing `partial_variables={"greeting": "Hello"}`, the `greeting` placeholder is automatically filled with `"Hello"`. When the `format()` method is later called with `name="Alice"`, the resulting output is `"Hello Alice"`, effectively combining the partial variable with the user-supplied variable.

---

### FewShotPromptTemplate

The **FewShotPromptTemplate** class extends `PromptTemplate` by incorporating few-shot examples into the prompt. This class is useful when you want to guide the language model using contextual examples.

#### Major Member Functions

- **`format(**kwargs)`**  
  Formats the prompt by inserting the few-shot examples, along with a prefix and suffix, into the overall prompt.

- **`from_examples(examples: list[str], suffix: str, input_variables: list[str], ...)`**  
  Creates a prompt template by dynamically generating a prompt that includes provided examples.

---

#### Example 1: Static Few-Shot Examples
 
Generate a prompt that includes a fixed set of examples to help the model understand the context.

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Create an example prompt for each few-shot example.
example_prompt = PromptTemplate.from_template("Q: {input}\nA: {output}")

# Define a few-shot prompt with static examples.
few_shot_prompt = FewShotPromptTemplate(
    examples=[
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+3?", "output": "6"}
    ],
    example_prompt=example_prompt,
    prefix="Please answer the following questions:",
    suffix="Q: {input}",
    input_variables=["input"],
)

formatted_prompt = few_shot_prompt.format(input="5+5")
print(formatted_prompt)
```

**Output:**
```
Please answer the following questions:

Q: What is 2+2?
A: 4

Q: What is 3+3?
A: 6

Q: 5+5
```

**Explanation:**  
This example builds a few-shot prompt using a static set of arithmetic questions. The prompt is structured with a prefix, a series of formatted examples, and a suffix where the new question is appended. When the `format()` method is called, it substitutes the input variable to complete the prompt.

---

#### Example 2: Dynamic Example Generation

Leverage the `from_examples()` method to dynamically create a few-shot prompt from a list of examples.

```python
from langchain import FewShotPromptTemplate

# Dynamically create a prompt with examples.
prompt = FewShotPromptTemplate(
    examples=[
        {"input": "Hello", "output": "Hola"},
        {"input": "Goodbye", "output": "Adiós"}
    ],
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Example: Translate '{input}' to Spanish -> '{output}'"
    ),
    suffix="Translate '{text}' to Spanish:",
    input_variables=["text"],
    prefix="Here are some translation examples:"
)

formatted_prompt = prompt.format(text="Thank you")
print(formatted_prompt)
```

**Output:**
```
Here are some translation examples:

Example: Translate 'Hello' to Spanish -> 'Hola'

Example: Translate 'Goodbye' to Spanish -> 'Adiós'

Translate 'Thank you' to Spanish:
```

**Explanation:**  
In this example, `from_examples()` constructs a prompt that includes multiple translation examples. The prompt is then dynamically extended with a new text ("Thank you") for translation. This method simplifies prompt creation when you have many examples to include.

---

#### Example 3: Dynamic Example Selection

This example demonstrates how to dynamically select few-shot examples based on semantic similarity rather than relying on a fixed set of examples. Instead of always including the same examples, a semantic similarity example selector is used to pick the most relevant example(s) from a pool based on the new input. In this scenario, when the input is `"3+3"`, the example selector retrieves the most similar example from the provided examples using embeddings and a vector store.

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Define a list of example Q&A pairs
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
]

# Create an example selector that uses semantic similarity
# This selector will use the OpenAIEmbeddings to compute similarity and store the vectors in a Chroma vector store.
# 'k=1' specifies that the top 1 most similar example will be selected.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small"),
    Chroma,
    k=1
)

# Define the template used to format each example
example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

# Create a FewShotPromptTemplate that uses the dynamic example selector.
# The suffix will be appended after the examples to prompt for a new answer.
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# Format the final prompt with a new input "3+3"
formatted_prompt = few_shot_prompt.format(input="3+3")

print(formatted_prompt)
```

**Output:**
```
Input: 2+3
Output: 5

Input: 3+3
Output:
```

**Explanation:**  
- **Semantic Example Selection:**  
  - The example selector uses semantic similarity to compare the new input `"3+3"` with the provided examples.
  - With `k=1`, it selects the top example (in this case, `{"input": "2+3", "output": "5"}`) that is most similar to the new input.
- **Example Formatting:**  
  - The selected example is formatted using the `example_prompt` template to produce a string like `Input: 2+3\nOutput: 5`.
- **Prompt Assembly:**  
  - The final prompt consists of the dynamically selected example(s) followed by the suffix, which includes the new input.
  - The final section `Input: 3+3\nOutput:` is where the model is expected to generate an answer.
- **Repeated Example:**  
  - Note that in this particular output, the selected example appears twice before the new input. This repetition might occur based on how the few-shot prompt template assembles its components; it serves to reinforce the context for the model.

---

### ChatPromptTemplate

The **ChatPromptTemplate** class is designed for building conversation flows with multiple messages. It supports various message types (system, human, AI) and can infer input variables from the conversation context.

#### Major Member Functions

- **`format_messages(**kwargs)`**  
  Formats the prompt into a list of finalized messages for a chat-based language model.

- **`invoke(inputs)`**  
  Directly processes input variables to generate a structured chat prompt (often wrapping `format_messages()`).

- **`from_messages(messages: Sequence)`**  
  A class method that instantiates a chat prompt template from various message representations.

---

#### Example 1: Basic Chat Conversation

Create a chat prompt using a sequence of message tuples, then generate the conversation flow.

```python
from langchain_core.prompts import ChatPromptTemplate

# Define a chat prompt with system, human, and AI messages.
# Each tuple represents a message in the conversation with a role and content.
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful AI assistant named {name}."),
    ("human", "Hello! How can I assist you today?"),
    ("ai", "I'm here to help!"),
    ("human", "{user_query}")
])

# Invoke the template with input variables to replace the placeholders.
chat_prompt = chat_template.invoke({
    "name": "Charlie",          # Replaces {name} in the system message.
    "user_query": "Can you tell me the weather forecast?"  # Replaces {user_query} in the human message.
})
print(chat_prompt)
```

**Output:**
```
messages = [
    SystemMessage(
        content="You are a helpful AI assistant named Charlie.",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="Hello! How can I assist you today?",
        additional_kwargs={},
        response_metadata={},
    ),
    AIMessage(content="I'm here to help!", additional_kwargs={}, response_metadata={}),
    HumanMessage(
        content="Can you tell me the weather forecast?",
        additional_kwargs={},
        response_metadata={},
    ),
]
```

**Explanation:**  
This example builds a simple chat conversation. The `ChatPromptTemplate` accepts a list of message tuples, and the `invoke()` method fills in the placeholders to produce a structured conversation ready for a chat model.

---

#### Example 2: Using `format_messages()`

Explicitly format messages into a finalized chat prompt list using the `format_messages()` method.

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a chat prompt template with mixed message types.
# This template includes a system message, a human inquiry, an AI response, and another human message with a placeholder.
chat_template = ChatPromptTemplate([
    ("system", "Your role is to provide weather updates."),
    ("human", "Hi, can you give me today's weather?"),
    ("ai", "Sure, please specify your location."),
    ("human", "{location}")
])

# Format the messages by supplying the 'location' variable.
messages = chat_template.format_messages(location="San Francisco")
# Print each formatted message from the conversation.
for message in messages:
    print(message)
```

**Output:**
```
content='Your role is to provide weather updates.' additional_kwargs={} response_metadata={}
content="Hi, can you give me today's weather?" additional_kwargs={} response_metadata={}
content='Sure, please specify your location.' additional_kwargs={} response_metadata={}
content='San Francisco' additional_kwargs={} response_metadata={}
```

**Explanation:**  
This example uses `format_messages()` to generate a list of finalized messages. Each message is formatted with the provided variable (`location`), creating a complete conversation context for the chat model.

---

#### Example 3: Using a MessagesPlaceholder

Demonstrate how to include dynamic conversation history in a chat prompt by using a `MessagesPlaceholder`. The placeholder allows you to inject a list of pre-formatted messages into the conversation dynamically.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a chat prompt template with a MessagesPlaceholder.
# The MessagesPlaceholder "history" will be replaced by the actual conversation history provided at runtime.
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),      # System message setting context.
    MessagesPlaceholder("history"),               # Placeholder for conversation history.
    ("human", "{input}"),                         # Human message with a placeholder for the new query.
])

# Format the template by providing conversation history and a new human input.
messages = template.format_messages(
    history=[
        ("human", "Hi!"),                         # Part of the conversation history.
        ("ai", "How can I assist you today?"),    # Part of the conversation history.
    ],
    input="Can you make me an ice cream sundae?"  # New user query replacing {input}.
)

# Print each formatted message from the conversation.
for message in messages:
    print(message)
```

**Output:**
```
content='You are a helpful AI bot.' additional_kwargs={} response_metadata={}
content='Hi!' additional_kwargs={} response_metadata={}
content='How can I assist you today?' additional_kwargs={} response_metadata={}
content='Can you make me an ice cream sundae?' additional_kwargs={} response_metadata={}
```

**Explanation:**  
- **Dynamic History Injection:**  
  - The `MessagesPlaceholder` named `"history"` is used to inject a list of pre-formatted messages into the prompt.  
  - At runtime, the placeholder is replaced with the messages provided under the `"history"` key.
- **Flexible Conversation Construction:**  
  - The template combines a static system message with dynamic conversation history and a new user query.  
  - The new query (replacing `{input}`) is appended after the injected history, creating a cohesive conversation flow.
- **Modular Design:**  
  - This approach allows for easy reuse of conversation history across different prompts without modifying the core template structure.

---

### FewShotChatMessagePromptTemplate

The **FewShotChatMessagePromptTemplate** class combines few-shot examples with chat templates. It formats a conversation that includes example exchanges before introducing the user's query.

#### Major Member Functions

- **`format_messages(**kwargs)`**  
  Formats few-shot examples into a sequence of messages.

- **`format(**kwargs)`**  
  Generates a string representation of the chat prompt by combining messages.

- **`aformat_messages(**kwargs)`**  
  An asynchronous version of `format_messages()` for use in async workflows.

---

#### Example 1: Fixed Few-Shot Examples in Chat
 
Construct a chat prompt that includes fixed few-shot examples to guide the conversation.

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# Define a chat template for few-shot examples.
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {input}?"),
    ("ai", "{output}")
])

# Create a few-shot chat prompt with static examples.
few_shot_chat = FewShotChatMessagePromptTemplate(
    examples=[
        {"input": "2+2", "output": "4"},
        {"input": "3+3", "output": "6"}
    ],
    example_prompt=example_prompt,
    input_variables=["input"]
)

# Build the final chat prompt.
final_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart assistant."),
    few_shot_chat,
    ("human", "{input}")
])
formatted_messages = final_chat_prompt.format_messages(input="4+4")
for msg in formatted_messages:
    print(msg)
```

**Output:**
```
[
    SystemMessage(content="You are a smart assistant.", additional_kwargs={}, response_metadata={}),
    HumanMessage(content="What is 2+2?", additional_kwargs={}, response_metadata={}),
    AIMessage(content="4", additional_kwargs={}, response_metadata={}),
    HumanMessage(content="What is 3+3?", additional_kwargs={}, response_metadata={}),
    AIMessage(content="6", additional_kwargs={}, response_metadata={}),
    HumanMessage(content="4+4", additional_kwargs={}, response_metadata={}),
]
```

**Explanation:**  
This example demonstrates how to integrate few-shot examples within a chat conversation. The `FewShotChatMessagePromptTemplate` is used to insert fixed arithmetic examples, and the final prompt is generated by combining system messages, few-shot examples, and a new human query.

---

#### Example 2: Async Formatting of Few-Shot Chat Prompts
 
Utilize the asynchronous `aformat_messages()` method to generate a chat prompt, suitable for asynchronous environments.

```python
import nest_asyncio
nest_asyncio.apply()

import asyncio
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# Define a chat template for formatting few-shot examples.
# This template formats each example as a human query and AI response.
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Convert {value} Celsius to Fahrenheit."),
    ("ai", "It is {result} Fahrenheit.")
])

# Create a few-shot chat prompt template with static examples.
# These examples serve as reference for the conversion task.
few_shot_chat = FewShotChatMessagePromptTemplate(
    examples=[
        {"value": "0", "result": "32"},
        {"value": "100", "result": "212"}
    ],
    example_prompt=example_prompt,
    input_variables=["value"]
)

# Build the final chat prompt by combining:
#  - A system message that sets the context,
#  - The few-shot chat examples,
#  - And a human message that poses a new conversion question.
final_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a conversion assistant."),
    few_shot_chat,
    ("human", "{value} Celsius in Fahrenheit?")
])

# Define an asynchronous function to format the messages.
async def generate_chat():
    # Use aformat_messages to asynchronously generate the list of chat messages.
    messages = await final_chat_prompt.aformat_messages(value="37")
    # Iterate over the generated messages and print each one.
    for msg in messages:
        print(f"{msg.__class__.__name__}: {msg.content}")

# Run the asynchronous function in the Colab notebook environment.
asyncio.run(generate_chat())
```

**Output:**
```
SystemMessage: You are a conversion assistant.
HumanMessage: Convert 0 Celsius to Fahrenheit.
AIMessage: It is 32 Fahrenheit.
HumanMessage: Convert 100 Celsius to Fahrenheit.
AIMessage: It is 212 Fahrenheit.
HumanMessage: 37 Celsius in Fahrenheit?
```

**Explanation:**  
This example highlights asynchronous usage with `aformat_messages()`. It builds a chat prompt that includes temperature conversion examples and then generates the final conversation asynchronously. This is particularly useful when integrating with async chat models.

---

### MessagesPlaceholder

The **MessagesPlaceholder** class is a utility for injecting pre-formatted message lists into a chat prompt. It is especially useful when you have an external source of conversation history or dynamic context that you want to include.

#### Major Member Functions

- **`format_messages(**kwargs)`**  
  Retrieves and formats a list of messages from the input variables. It supports options like returning an empty list if the variable is optional.

- **`pretty_repr(html: bool = False)`**  
  Returns a human-readable representation of the placeholder, optionally formatted as HTML.

---

#### Example 1: Basic Message Injection

Inject a list of pre-formatted messages into a chat prompt using `MessagesPlaceholder`.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a chat prompt that includes a placeholder for conversation history.
# The MessagesPlaceholder named "history" will be replaced with actual messages at runtime.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),     # Static system message setting the context.
    MessagesPlaceholder("history", optional=True),  # Placeholder for dynamic conversation history.
    ("human", "{question}")                         # Human message with a placeholder for the new question.
])

# Invoke the prompt by providing the conversation history and a new question.
# The 'history' key is used to fill in the MessagesPlaceholder.
chat_output = prompt.invoke({
    "history": [
        ("human", "How do I prepare coffee?"),                       # Previous human message.
        ("ai", "Start by boiling water and adding coffee grounds.")  # Previous AI response.
    ],
    "question": "What about adding milk?"                            # New human question.
})
print(chat_output)
```

**Output:**
```
messages = [
    SystemMessage(
        content="You are a helpful assistant.", additional_kwargs={}, response_metadata={}
    ),
    HumanMessage(
        content="How do I prepare coffee?", additional_kwargs={}, response_metadata={}
    ),
    AIMessage(
        content="Start by boiling water and adding coffee grounds.", additional_kwargs={}, response_metadata={}
    ),
    HumanMessage(
        content="What about adding milk?", additional_kwargs={}, response_metadata={}
    ),
]
```

**Explanation:**  
This example uses `MessagesPlaceholder` to inject an optional conversation history into the chat prompt. If the history is provided, it gets converted into proper message objects; otherwise, it defaults to an empty list.

---

#### Example 2: Limiting the Number of Messages

Use the `n_messages` parameter to restrict the number of injected messages, ensuring that only the most recent messages are included.

```python
from langchain_core.prompts import MessagesPlaceholder

# Create a placeholder that only accepts the last 1 message.
# Here, "conversation" is the key that will hold the list of messages.
history_placeholder = MessagesPlaceholder("conversation", n_messages=1)

# Format the messages using more than one message in the input.
# Even though two messages are provided, only the most recent one will be returned.
formatted_messages = history_placeholder.format_messages(conversation=[
    ("human", "Hi there!"),                # First message in the conversation.
    ("ai", "Hello! How can I assist?")     # Second (and most recent) message.
])
# Print the formatted message(s) which should only include the last message.
for message in formatted_messages:
    print(message)
```

**Output:**
```
content='Hello! How can I assist?' additional_kwargs={} response_metadata={}
```

**Explanation:**  
In this example, `MessagesPlaceholder` is configured with `n_messages=1`, which means that even if multiple messages are provided under the "conversation" key, only the last message will be returned. This is useful for limiting context to the most recent interaction.

---

## Comparison Table

Below is a summary of the main LangChain prompt template classes, along with a brief description of their purpose, key methods, and typical use cases. This table helps you quickly understand which class to use depending on your prompt requirements.

| **Class**                             | **Purpose**                                                                                      | **Key Methods**                                        | **Use Case**                                         |
|---------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------|
| **PromptTemplate**                    | Creates string-based prompt templates with placeholders for dynamic variable substitution.       | `format()`, `from_template()`, `__add__()`             | Basic text prompt formatting, quick and simple prompts.                        |
| **FewShotPromptTemplate**             | Integrates few-shot examples into a prompt, providing contextual examples along with the prompt.   | `format()`, `from_examples()`, `save()`                | Enhancing prompts with illustrative examples for improved responses.          |
| **ChatPromptTemplate**                | Constructs conversation flows for chat models by organizing multiple message types (system, human, AI). | `format_messages()`, `invoke()`, `__add__()`           | Developing chatbots and interactive conversational models with dynamic messages. |
| **FewShotChatMessagePromptTemplate**  | Combines few-shot learning with chat templates to include multi-message examples in the conversation. | `format()`, `format_messages()`, `aformat_messages()`  | Complex chat prompts where example-driven context is critical for performance.    |

---
## Practical Examples

Below are practical examples demonstrating how each class can be used.

### Example 1: Dynamic Survey Question Generator

A template to generate customized survey questions based on industry and target audience.

```python
# Import PromptTemplate from LangChain
from langchain.prompts import PromptTemplate

# Define a survey template with placeholders for industry, audience, and focus_area.
survey_template = PromptTemplate(
    input_variables=["industry", "audience", "focus_area"],
    template="""Create 3 survey questions for {audience} in the {industry} industry focusing on {focus_area}.
Questions should be concise and use appropriate terminology for the target audience."""
)

# Format the template by providing specific values for the placeholders.
result = survey_template.format(
    industry="healthcare",
    audience="medical professionals",
    focus_area="patient care technology adoption"
)

# Print the generated survey prompt.
print(result)
```

**Output:**
```
Create 3 survey questions for medical professionals in the healthcare industry focusing on patient care technology adoption.
Questions should be concise and use appropriate terminology for the target audience.
```

**Explanation:**  
- **Template Setup:**  
  - Uses a multi-line string with placeholders (`{industry}`, `{audience}`, `{focus_area}`) to define the survey prompt.
- **Dynamic Replacement:**  
  - The `format()` method dynamically replaces placeholders with provided values.
- **Use Case:**  
  - Useful for generating industry-specific survey questions.

---

### Example 2: Case Study Analysis with Examples

Using few-shot learning to analyze business cases based on previous examples.

```python
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

# Define examples with case details and corresponding analyses.
examples = [
    {
        "case": "Company X experienced 30% revenue drop after website redesign",
        "analysis": "Root cause: Poor user experience testing before launch\nImpact: Lost customer confidence\nSolution: A/B testing implementation and gradual rollout"
    },
    {
        "case": "Startup Y faced 50% employee turnover in 6 months",
        "analysis": "Root cause: Inadequate onboarding and unclear growth paths\nImpact: Knowledge loss and decreased productivity\nSolution: Structured mentorship program and career development plans"
    }
]

# Define a template to format each example.
example_formatter_template = """
Case: {case}
Analysis: {analysis}
"""

# Create an example prompt using the defined template.
example_prompt = PromptTemplate(
    input_variables=["case", "analysis"],
    template=example_formatter_template
)

# Create a FewShotPromptTemplate using the examples and example prompt.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Analyze the following business case following the format in these examples:",
    suffix="Case: {input}\nAnalysis:",
    input_variables=["input"]
)

# Format the prompt for a new business case.
formatted_prompt = few_shot_prompt.format(input="5+5")
print(formatted_prompt)
```

**Output:**
```
Analyze the following business case following the format in these examples:

Case: Company X experienced 30% revenue drop after website redesign
Analysis: Root cause: Poor user experience testing before launch
Impact: Lost customer confidence
Solution: A/B testing implementation and gradual rollout

Case: Startup Y faced 50% employee turnover in 6 months
Analysis: Root cause: Inadequate onboarding and unclear growth paths
Impact: Knowledge loss and decreased productivity
Solution: Structured mentorship program and career development plans

Case: 5+5
Analysis:
```

**Explanation:**  
- **Few-Shot Learning:**  
  - Integrates multiple examples to provide context.
- **Prefix and Suffix:**  
  - Uses a prefix to introduce the examples and a suffix to append the new input.
- **Structured Output:**  
  - Helps guide the model to generate an analysis following the given format.

---

### Example 3: Customer Support Chat Template

Creating structured chat templates for customer support interactions.

```python
# Import ChatPromptTemplate and message-specific templates for system and human messages.
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Create a chat prompt template using message-based templates.
chat_template = ChatPromptTemplate.from_messages([
    # System message to set context.
    SystemMessagePromptTemplate.from_template(
        "You are a customer support specialist for {company_name}. Focus on {product_line}."
    ),
    # Human message that describes the customer's issue.
    HumanMessagePromptTemplate.from_template(
        "Customer Issue: {issue_description}\nProduct: {product_name}\nPurchase Date: {purchase_date}"
    )
])

# Format the chat messages by providing specific values.
messages = chat_template.format_messages(
    company_name="TechGear",
    product_line="wireless headphones",
    issue_description="Device won't pair with my phone",
    product_name="SoundPro X1",
    purchase_date="2024-01-15"
)
print(messages)
```

**Output:**
```
[
    SystemMessage(
        content="You are a customer support specialist for TechGear. Focus on wireless headphones.",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="Customer Issue: Device won't pair with my phone\nProduct: SoundPro X1\nPurchase Date: 2024-01-15",
        additional_kwargs={},
        response_metadata={},
    ),
]
```

**Explanation:**  
- **Structured Conversation:**  
  - Separates system instructions and human queries.
- **Message Templates:**  
  - Uses `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate` for clarity.
- **Customization:**  
  - Dynamic replacement of placeholders to generate a complete conversation.

---

### Example 4: Contextual Conversation Handler

Using MessagesPlaceholder for maintaining conversation context in a support scenario.

```python
# Import ChatPromptTemplate and MessagesPlaceholder along with required message classes.
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.prompts.chat import HumanMessagePromptTemplate

# Create a chat prompt that includes a system message, a placeholder for previous chat history,
# and a human message that asks a new question.
template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content="You are a technical support specialist. Use previous conversation context to provide consistent help."
    ),
    # Placeholder for dynamically injected conversation history.
    MessagesPlaceholder(variable_name="chat_history"),
    # Human message template for the new question.
    HumanMessagePromptTemplate.from_template("{new_question}")
])

# Define an example chat history list.
chat_history = [
    # Prior human message.
    HumanMessage(content="My laptop won't turn on"),
    # Previous AI response.
    AIMessage(content="Let's try basic troubleshooting. Is the battery charged?"),
    # Additional human message.
    HumanMessage(content="Yes, the battery is at 100%"),
    # AI message with a suggestion.
    AIMessage(content="Try holding the power button for 30 seconds to force a reset.")
]

# Format the prompt with the chat history and a new question.
messages = template.format_messages(
    chat_history=chat_history,
    new_question="The reset didn't work, what should I try next?"
)
print(messages)
```

**Output:**
```
[
    SystemMessage(
        content="You are a technical support specialist. Use previous conversation context to provide consistent help.",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="My laptop won't turn on", additional_kwargs={}, response_metadata={}
    ),
    AIMessage(
        content="Let's try basic troubleshooting. Is the battery charged?",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="Yes, the battery is at 100%",
        additional_kwargs={},
        response_metadata={},
    ),
    AIMessage(
        content="Try holding the power button for 30 seconds to force a reset.",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="The reset didn't work, what should I try next?",
        additional_kwargs={},
        response_metadata={},
    ),
]
```

**Explanation:**  
- **Dynamic Context Injection:**  
  - Uses `MessagesPlaceholder` to inject past conversation history dynamically.
- **Maintaining Continuity:**  
  - The prompt includes both previous context and a new query to provide consistent assistance.
- **Modular Chat Construction:**  
  - Combines system messages, historical conversation, and new user input seamlessly.

---

### Example 5: Dynamic Survey Question Generator with Few-Shot

Using few-shot examples to generate creative survey questions based on given topics.

```python
# Import necessary modules for prompt templates.
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Define a few-shot example for survey questions.
examples = [
    {
        "industry": "technology",
        "audience": "software developers",
        "focus_area": "emerging programming trends",
        "questions": "1. What emerging languages are you exploring?\n2. How do you evaluate new frameworks?\n3. What challenges do you face in rapid prototyping?"
    },
    {
        "industry": "education",
        "audience": "high school teachers",
        "focus_area": "digital learning tools",
        "questions": "1. Which digital tools have transformed your classroom?\n2. How do you integrate technology into lesson plans?\n3. What are the challenges of remote learning?"
    }
]

# Define an example formatter template for survey questions.
example_formatter_template = """
Industry: {industry}
Audience: {audience}
Focus Area: {focus_area}
Survey Questions:
{questions}
"""

# Create an example prompt for the few-shot template.
example_prompt = PromptTemplate(
    input_variables=["industry", "audience", "focus_area", "questions"],
    template=example_formatter_template
)

# Create a FewShotPromptTemplate with the examples.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Generate a creative survey for the following context based on these examples:",
    suffix="\nIndustry: {industry}\nAudience: {audience}\nFocus Area: {focus_area}\nSurvey Questions:",
    input_variables=["industry", "audience", "focus_area"]
)

# Format the prompt with new survey context.
formatted_prompt = few_shot_prompt.format(
    industry="healthcare",
    audience="nurses",
    focus_area="workplace stress management"
)
print(formatted_prompt)
```

**Output:**
```
Generate a creative survey for the following context based on these examples:

Industry: technology
Audience: software developers
Focus Area: emerging programming trends
Survey Questions:
1. What emerging languages are you exploring?
2. How do you evaluate new frameworks?
3. What challenges do you face in rapid prototyping?

Industry: education
Audience: high school teachers
Focus Area: digital learning tools
Survey Questions:
1. Which digital tools have transformed your classroom?
2. How do you integrate technology into lesson plans?
3. What are the challenges of remote learning?

Industry: healthcare
Audience: nurses
Focus Area: workplace stress management
Survey Questions:
```

**Explanation:**  
- **Few-Shot Integration:**  
  - Combines multiple few-shot examples to provide context for generating survey questions.
- **Prefix and Suffix:**  
  - Uses a prefix to introduce the examples and a suffix to append the new context.
- **Dynamic Generation:**  
  - The `format()` method replaces placeholders with new input values, guiding the model to produce relevant survey questions.
  
---

## Best Practices

When working with LangChain's prompt templates, adhering to best practices can help ensure that your prompts are secure, modular, and effective. Below are some best practices along with code snippets to illustrate each point.

- **Prefer f-string Formatting**  
  *Use f-string formatting (`template_format="f-string"`) as it is secure and efficient.*  
  ```python
  from langchain_core.prompts import PromptTemplate

  # Using f-string formatting (default and secure)
  prompt = PromptTemplate.from_template("Hello, {name}!")
  formatted_prompt = prompt.format(name="Alice")
  print(formatted_prompt)  # Output: "Hello, Alice!"
  ```

- **Validate Templates When Necessary**  
  *Enable template validation to ensure that all placeholders match the expected input variables.*  
  ```python
  from langchain_core.prompts import PromptTemplate

  # Enable validation to ensure the template's placeholders are correct.
  prompt = PromptTemplate.from_template("Hello, {name}!", validate_template=True)
  # This will raise an error if the provided variables do not match the template.
  formatted_prompt = prompt.format(name="Bob")
  print(formatted_prompt)
  ```

- **Utilize Partial Variables**  
  *Leverage partial variables to pre-fill common placeholders, reducing redundancy and potential errors.*  
  ```python
  from langchain_core.prompts import PromptTemplate

  # Pre-fill the 'greeting' variable to avoid repetitive input.
  prompt = PromptTemplate.from_template("{greeting} {name}", partial_variables={"greeting": "Hello"})
  formatted_prompt = prompt.format(name="Carol")
  print(formatted_prompt)  # Output: "Hello Carol"
  ```

- **Design Modular Prompts**  
  *Build complex prompts by combining smaller, reusable prompt templates using the `+` operator or composition techniques.*  
  ```python
  from langchain_core.prompts import PromptTemplate

  # Create two small prompt templates and combine them.
  greeting = PromptTemplate.from_template("Hello, {name}!")
  followup = PromptTemplate.from_template(" How can I assist you today?")
  combined_prompt = greeting + followup
  print(combined_prompt.format(name="Dave"))
  # Output: "Hello, Dave! How can I assist you today?"
  ```

- **Use Dynamic Message Injection**  
  *In chat applications, use `MessagesPlaceholder` to inject dynamic conversation history, making your chat prompts adaptable to various contexts.*  
  ```python
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

  # Create a chat prompt with a placeholder for dynamic history.
  chat_template = ChatPromptTemplate.from_messages([
      ("system", "You are a helpful assistant."),
      MessagesPlaceholder("history", optional=True),
      ("human", "{question}")
  ])

  # Inject conversation history into the prompt.
  chat_output = chat_template.invoke({
      "history": [
          ("human", "How do I make tea?"),
          ("ai", "Boil water and steep the tea bag for 3-5 minutes.")
      ],
      "question": "What if I want to add lemon?"
  })
  print(chat_output)
  ```

---

## Conclusion

LangChain's prompt templating framework empowers developers to build flexible and effective interactions with language models. By using classes like **PromptTemplate** and **ChatPromptTemplate**, you can create simple yet powerful text prompts, while **FewShotPromptTemplate** and **FewShotChatMessagePromptTemplate** enable you to incorporate context through example-driven learning. Additionally, **MessagesPlaceholder** allows for the dynamic injection of conversation history, ensuring that your chat applications remain contextually relevant.

Through the practical examples provided, we demonstrated how to generate survey questions, analyze business cases, structure customer support dialogues, and maintain conversation context in dynamic scenarios. By following best practices and leveraging the modular design of these templates, you can build applications that not only meet but exceed the demands of modern AI-driven interactions.

LangChain’s prompt system is a vital asset for anyone looking to harness the power of language models, enabling efficient prompt design and facilitating more natural, human-like interactions. Whether you are a developer, data scientist, or researcher, integrating these prompt templates into your projects will help you unlock the full potential of your language models.

---