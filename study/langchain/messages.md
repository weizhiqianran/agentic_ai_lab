# Introduction to LangChain Messages

LangChain Messages are a key component in the LangChain framework. They serve as the input and output representations for chat models, encapsulating the conversation flow between humans and AI systems. In this article, we will explore the structure, attributes, and various types of LangChain Messages through detailed explanations and examples.

---

### Table of Contents

- [Introduction](#introduction)
- [Attributes of Messages](#attributes-of-messages)
  - [Comparison of Message Types](#comparison-of-message-types)
  - [Comparison of Message Attributes](#comparison-of-message-attributes)
- [Message Types](#message-types)
  - [SystemMessage](#systemmessage)
  - [HumanMessage](#humanmessage)
  - [AIMessage](#aimessage)
  - [AIMessageChunk](#aimessagechunk)
  - [ToolMessage](#toolmessage)
- [Utility Functions](#utility-functions)
  - [trim_messages: Trimming Chat History](#trim_messages-trimming-chat-history)
  - [filter_messages: Filtering Chat History](#filter_messages-filtering-chat-history)
- [Best Practices](#best-practices)
- [Conclusion](#conclusion)

---

## Introduction

LangChain Messages form the backbone of the LangChain framework, enabling seamless communication between users and AI models. They encapsulate every interaction—whether it’s a system prompt, a user query, or an AI response—ensuring that each message carries the necessary context, metadata, and structure required for robust dialogue management. In LangChain, messages are built on a common foundation, the `BaseMessage`, and are extended into various specialized types such as `SystemMessage`, `HumanMessage`, `AIMessage`, `AIMessageChunk`, and `ToolMessage`. These message types not only standardize interactions but also offer enhanced functionalities like streaming responses, tool integrations, and granular filtering and trimming of conversation history.

This comprehensive guide covers the attributes of messages, comparisons between different message types and their attributes, and details the utility functions—such as `trim_messages` and `filter_messages`—designed to manage chat history effectively. Additionally, we discuss best practices to ensure that your conversational AI maintains context, efficiency, and clarity. Whether you are building a simple chatbot or a complex AI-driven application, understanding LangChain Messages is essential for creating scalable and maintainable solutions.

---

## Attributes of Messages

Every message in LangChain is derived from the `BaseMessage` class and includes several important attributes:

- **content**:  
  This attribute contains the primary text or data of the message. It can be a simple string or a list of strings/dictionaries.

- **additional_kwargs**:  
  A dictionary reserved for any extra payload data. For example, an AI message might include tool call details within this field.

- **response_metadata**:  
  This stores metadata related to the message, such as response headers, token counts, or logging information.

- **type**:  
  A string that uniquely identifies the message type. This is crucial during the serialization and deserialization process.

- **name**:  
  An optional, human-readable name for the message. This can be used for display purposes or easier debugging.

- **id**:  
  An optional unique identifier, typically provided by the model or provider that creates the message.

These attributes ensure that each message carries all necessary information to support various operations such as serialization, pretty-printing, and even combining messages.

---

### Comparison of Message Types

The following table provides a side-by-side comparison of the various LangChain message types, outlining their roles and key characteristics.

| **Message Type**   | **Role/Purpose**                                          | **Key Characteristics**                                          | **Additional Information**                        |
|--------------------|-----------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------|
| **SystemMessage**  | Primes the AI by providing system-level instructions     | Sets the context for subsequent interactions                     | Typically used as the first message in a conversation |
| **HumanMessage**   | Captures user input and queries                           | Represents human communication                                   | Can include an optional flag for example conversations  |
| **AIMessage**      | Represents a complete response from the AI                | Contains both the AI's raw output and standardized metadata      | May include tool calls and usage metadata          |
| **AIMessageChunk** | Handles incremental or streaming AI responses             | Supports chunking of AI responses for real-time updates            | Chunks can be combined using the `+` operator      |
| **ToolMessage**    | Conveys results from tool execution to the model          | Links tool call IDs with their outputs and any associated artifacts | Provides error status and additional tool-specific data |

---

### Comparison of Message Attributes

Beyond the type-specific details, each message type builds upon a common set of attributes defined in `BaseMessage`. The table below compares these shared attributes as well as additional ones specific to each message type.

| **Attribute**           | **SystemMessage** | **HumanMessage** | **AIMessage**                     | **AIMessageChunk**                 | **ToolMessage**                              |
|-------------------------|-------------------|------------------|-----------------------------------|------------------------------------|----------------------------------------------|
| **content**             | ✓                 | ✓                | ✓                                 | ✓                                  | ✓                                            |
| **additional_kwargs**   | ✓                 | ✓                | ✓                                 | ✓                                  | ✓                                            |
| **response_metadata**   | ✓                 | ✓                | ✓                                 | ✓                                  | ✓                                            |
| **type**                | "system"          | "human"          | "ai"                              | "AIMessageChunk"                   | "tool"                                       |
| **name**                | Optional          | Optional         | Optional                          | Optional                           | Not typically used                           |
| **id**                  | Optional          | Optional         | Optional                          | Optional                           | Not typically used                           |
| **Tool-Specific Data**  | -                 | -                | `tool_calls`, `invalid_tool_calls`| `tool_call_chunks`                 | `tool_call_id`, `artifact`, `status`         |

---

## Message Types

LangChain defines several message types, each serving a distinct role in the communication flow. Let’s look at each one in detail.

### SystemMessage

Used to prime the AI’s behavior. The system message is typically the first message in a conversation, setting the context and guidelines for subsequent interactions.

#### **Example 1: Direct Invocation with SystemMessage**

In this example, a `SystemMessage` is used along with a `HumanMessage` to directly invoke a chat model. The system message provides the initial context, while the human message contains the query.

```python
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Retrieve the API key
gemini_api_key = userdata.get('api-key-gemini')

# Initialize ChatGoogleGenerativeAI with the API key and desired model
llm_gemini = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.0-flash",
    temperature=0
)

messages = [
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
    HumanMessage(content="What is your name?")
]

# Assuming a chat model instance `model`:
print(llm_gemini.invoke(messages))
```

**Output:**
```
content='My name is Bob.'
additional_kwargs={}
response_metadata = {
    "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
    "finish_reason": "STOP",
    "safety_ratings": [],
}
id='run-f05484b5-a849-48e5-b03f-4b81e2b918d1-0' 
usage_metadata = {
    "input_tokens": 16,
    "output_tokens": 6,
    "total_tokens": 22,
    "input_token_details": {"cache_read": 0},
}
```

**Explanation:**
- **Context Setup:** The `SystemMessage` ("You are a helpful assistant! Your name is Bob.") primes the AI with the necessary background.
- **User Query:** The `HumanMessage` ("What is your name?") follows the system message, triggering the conversation.
- **Direct Invocation:** The messages are passed directly to the chat model via `model.invoke(messages)`, resulting in a response that leverages the provided context.

---

#### **Example 2: Using ChatPromptTemplate with SystemMessage**

This example demonstrates how to create a unified prompt using the `ChatPromptTemplate`, which combines the `SystemMessage` and `HumanMessage` into a formatted prompt ready for the chat model.

```python
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Create a ChatPromptTemplate using a list of messages
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
    HumanMessage(content="What is your name?")
])

# Format the prompt for the chat model
formatted_prompt = chat_prompt.format_prompt()
print(formatted_prompt)
```

**Output:**
```
messages = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob.",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="What is your name?", additional_kwargs={}, response_metadata={}
    ),
]
```

**Explanation:**
- **Context Setup:** The `SystemMessage` provides the background context.
- **Combined Message Sequence:** The `ChatPromptTemplate` merges both the system and human messages into one cohesive prompt.
- **Formatted Output:** The `format_prompt()` method formats the messages as expected by the chat model, making it easier to manage and deploy conversation flows.

---

### HumanMessage

Represents messages from a human user. These messages include the questions or commands that the user inputs to interact with the chat model.

#### **Example 1: Direct Invocation with HumanMessage**

In this example, a `HumanMessage` is used to send a query directly to the chat model. The human message is paired with a system message to provide context.

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
    HumanMessage(content="What is your name?")
]

# Invoking the chat model with human input
print(llm_gemini.invoke(messages))
```

**Output:**
```
content='My name is Bob.'
additional_kwargs={}
response_metadata = {
    "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
    "finish_reason": "STOP",
    "safety_ratings": [],
}
id='run-c318dbcc-847e-4add-ade9-3171786599b6-0'
usage_metadata = {
    "input_tokens": 16,
    "output_tokens": 6,
    "total_tokens": 22,
    "input_token_details": {"cache_read": 0},
}
```

**Explanation:**
- **User Input:** The `HumanMessage` encapsulates the user's query ("What is your name?").
- **Sequential Interaction:** The human message follows the system message to ensure that the conversation starts with proper context.
- **Direct Communication:** The chat model processes the messages in order, using the context from the system message and the query from the human message to generate a response.

---

#### **Example 2: Using ChatPromptTemplate with HumanMessage**

This example illustrates how to leverage the `ChatPromptTemplate` to combine the system and human messages into a single prompt that is formatted for the chat model.

```python
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Create a ChatPromptTemplate using a list of messages
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
    HumanMessage(content="What is your name?")
])

# Format and print the prompt that will be sent to the chat model
formatted_prompt = chat_prompt.format_prompt()
print(formatted_prompt)
```

**Output:**
```
messages = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob.",
        additional_kwargs={},
        response_metadata={},
    ),
    HumanMessage(
        content="What is your name?", additional_kwargs={}, response_metadata={}
    ),
]
```

**Explanation:**
- **User Query Integration:** The `HumanMessage` ("What is your name?") is combined with the system context.
- **Unified Prompt:** The `ChatPromptTemplate` merges these messages into a single, formatted prompt.
- **Streamlined Communication:** This approach ensures consistent formatting and easier management of conversation flows, facilitating direct communication with the chat model.

---

### AIMessage

The `AIMessage` is a specialized message type used to represent responses generated by an AI. It not only contains the text or content produced by the AI but also includes additional metadata such as tool calls, usage data, and any errors that may have occurred. This makes it easier to handle complex interactions where the AI might delegate tasks to external tools before delivering a final response.

#### **Example 1: Invoking One Tool**

In this example, the AI invokes a single tool (e.g., a calculator) to perform a computation.

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# System message to prime the AI
system_message = SystemMessage(content="You are a helpful assistant. You can use tools to perform calculations.")

# Human asks a question
human_message = HumanMessage(content="What is 40 + 2?")

# AI responds with a tool call to a calculator
ai_response = AIMessage(
    content="I will calculate that for you.",
    tool_calls=[
        {
            "name": "calculator",              # Name of the tool
            "args": {"expression": "40 + 2"},  # Arguments for the tool
            "id": "call_12345"                 # Unique ID for the tool call
        }
    ]
)

# Print the AI's response
print(ai_response.pretty_repr())
```

**Output:**
```
================================== Ai Message ==================================

I will calculate that for you.
Tool Calls:
  calculator (call_12345)
 Call ID: call_12345
  Args:
    expression: 40 + 2
```

**Explanation:**
- **System Message**: Sets the context by instructing the AI about its role and capability to use tools.
- **Human Message**: Contains the user's question asking for a calculation.
- **AI Response**:
  - Provides a textual confirmation ("I will calculate that for you.").
  - Includes a `tool_calls` list that specifies:
    - The tool name (`calculator`).
    - The arguments (`{"expression": "40 + 2"}`) required to perform the calculation.
    - A unique identifier (`"call_12345"`) to track the tool invocation.
- **Output:** The printed output shows both the AI's message and the details of the tool call in a formatted manner.

---

#### **Example 2: Invoking Three Tools**

In this example, the AI invokes three tools to gather information or perform multiple tasks.

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# System message to prime the AI
system_message = SystemMessage(content="You are a helpful assistant. You can use tools to fetch data or perform tasks.")

# Human asks a complex question
human_message = HumanMessage(content="What is the weather in New York, the population of France, and the current time in London?")

# AI responds with tool calls to three different tools
ai_response = AIMessage(
    content="I will fetch that information for you.",
    tool_calls=[
        {
            "name": "weather_tool",     # Tool to fetch weather
            "args": {"location": "New York"},
            "id": "call_67890"
        },
        {
            "name": "population_tool",  # Tool to fetch population
            "args": {"country": "France"},
            "id": "call_54321"
        },
        {
            "name": "time_tool",        # Tool to fetch current time
            "args": {"city": "London"},
            "id": "call_98765"
        }
    ]
)

# Print the AI's response
print(ai_response.pretty_repr())
```

**Output:**
```
================================== Ai Message ==================================

I will fetch that information for you.
Tool Calls:
  weather_tool (call_67890)
 Call ID: call_67890
  Args:
    location: New York
  population_tool (call_54321)
 Call ID: call_54321
  Args:
    country: France
  time_tool (call_98765)
 Call ID: call_98765
  Args:
    city: London
```

**Explanation:**
- **System Message**: Provides instructions indicating that the AI can use tools to fetch various data.
- **Human Message**: Contains a multi-part query asking for weather, population, and time.
- **AI Response**:
  - Announces that it will fetch the required information.
  - Specifies three separate tool calls:
    - **Weather Tool**: Fetches weather data for New York.
    - **Population Tool**: Retrieves population information for France.
    - **Time Tool**: Obtains the current time in London.
- **Output:** The formatted output lists all three tool calls with their respective names, IDs, and arguments.

---

#### **Example 3: Handling Tool Responses**

After the tools are invoked, the results are passed back to the AI using `ToolMessage`. The AI then processes the results and responds to the user.

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# System message to prime the AI
system_message = SystemMessage(
    content="You are a helpful assistant. You can use tools to fetch data or perform tasks."
)

# Human asks a complex question
human_message = HumanMessage(
    content="What is the weather in New York, the population of France, and the current time in London?"
)

# AI responds with tool calls to three different tools
ai_response = AIMessage(
    content="I will fetch that information for you.",
    tool_calls=[
        {
            "name": "weather_tool",
            "args": {"location": "New York"},
            "id": "call_67890"
        },
        {
            "name": "population_tool",
            "args": {"country": "France"},
            "id": "call_54321"
        },
        {
            "name": "time_tool",
            "args": {"city": "London"},
            "id": "call_98765"
        }
    ]
)

# Tool responses
weather_tool_response = ToolMessage(
    content="Sunny, 75°F",
    tool_call_id="call_67890"
)

population_tool_response = ToolMessage(
    content="67 million",
    tool_call_id="call_54321"
)

time_tool_response = ToolMessage(
    content="10:30 AM",
    tool_call_id="call_98765"
)

# AI processes the tool responses and provides a final answer
final_ai_response = AIMessage(
    content=(
        "Here is the information you requested:\n"
        "- Weather in New York: Sunny, 75°F\n"
        "- Population of France: 67 million\n"
        "- Current time in London: 10:30 AM"
    )
)

# Print the conversation
messages = [
    system_message,
    human_message,
    ai_response,
    weather_tool_response,
    population_tool_response,
    time_tool_response,
    final_ai_response
]

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

**Output:**
```
system: You are a helpful assistant. You can use tools to fetch data or perform tasks.
human: What is the weather in New York, the population of France, and the current time in London?
ai: I will fetch that information for you.
tool: Sunny, 75°F
tool: 67 million
tool: 10:30 AM
ai: Here is the information you requested:
- Weather in New York: Sunny, 75°F
- Population of France: 67 million
- Current time in London: 10:30 AM
```

**Explanation:**
- **Initial Messages**:
  - The **System Message** sets the context for tool usage.
  - The **Human Message** poses a complex query.
- **AI's First Response**:
  - The AI message includes tool calls for fetching weather, population, and time information.
- **Tool Responses**:
  - Each `ToolMessage` corresponds to the tool call, identified by matching the `tool_call_id` with the one specified in the AI's message.
- **Final AI Response**:
  - The AI consolidates the information from the tool responses and presents a final, formatted answer.
- **Output Flow**:
  - The printed conversation shows the sequential flow of messages, from setting context and asking the question to invoking tools and providing the final answer.

---

#### **Example 4: Handling Invalid Tool Calls**

In this example, the AI attempts to invoke a tool, but one of the tool calls fails due to invalid arguments. The AI handles the error gracefully and provides a response.

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# System message to prime the AI
system_message = SystemMessage(
    content="You are a helpful assistant. You can use tools to fetch data or perform tasks."
)

# Human asks a question
human_message = HumanMessage(
    content="What is the weather in New York and the population of Mars?"
)

# AI responds with tool calls, including an invalid one
ai_response = AIMessage(
    content="I will fetch that information for you.",
    tool_calls=[
        {
            "name": "weather_tool",
            "args": {"location": "New York"},
            "id": "call_67890"
        },
        {
            "name": "population_tool",
            "args": {"planet": "Mars"},  # Invalid argument (tool expects a country, not a planet)
            "id": "call_54321"
        }
    ],
    invalid_tool_calls=[
        {
            "name": "population_tool",
            # Convert args to a string representation for invalid_tool_calls
            "args": str({"planet": "Mars"}),  
            "id": "call_54321",
            "error": "Invalid argument: 'planet' is not a valid parameter."
        }
    ]
)

# Tool response for the valid tool call
weather_tool_response = ToolMessage(
    content="Sunny, 75°F",
    tool_call_id="call_67890"
)

# AI processes the tool response and the invalid tool call
final_ai_response = AIMessage(
    content=(
        "Here is the information I could fetch:\n"
        "- Weather in New York: Sunny, 75°F\n"
        "- Population of Mars: Could not fetch data. Invalid argument: 'planet' is not a valid parameter."
    )
)

# Print the conversation
messages = [
    system_message,
    human_message,
    ai_response,
    weather_tool_response,
    final_ai_response
]

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

**Output:**
```
system: You are a helpful assistant. You can use tools to fetch data or perform tasks.
human: What is the weather in New York and the population of Mars?
ai: I will fetch that information for you.
tool: Sunny, 75°F
ai: Here is the information I could fetch:
- Weather in New York: Sunny, 75°F
- Population of Mars: Could not fetch data. Invalid argument: 'planet' is not a valid parameter.
```

**Explanation:**
- **Invalid Tool Call**:
  - The AI attempts to invoke the `population_tool` with an invalid argument (`planet` instead of `country`).
  - The `invalid_tool_calls` field captures the error details, including the tool name, arguments, and error message.
- **Graceful Handling**:
  - The AI processes the valid tool response for the weather and includes the error message for the invalid tool call in its final response.
- **Output Flow**:
  - The conversation shows how the AI handles both successful and failed tool executions, providing a clear and informative response to the user.

---

### AIMessageChunk

In scenarios where the AI response is generated incrementally (e.g., streaming responses), `AIMessageChunk` represents parts or "chunks" of the complete message. This mechanism allows real-time updates and efficient handling of large responses.

---

#### **Example 1: Combining Two AIMessageChunks**

This example demonstrates how two message chunks can be combined into a complete AI response using the overloaded `+` operator.

```python
from langchain_core.messages import AIMessageChunk

chunk1 = AIMessageChunk(
    content="My name is Bob, ",
    tool_call_chunks=[]
)

chunk2 = AIMessageChunk(
    content="and I'm here to help!",
    tool_call_chunks=[]
)

# Combining chunks (assuming the '+' operator is overloaded to combine chunks)
full_response = chunk1 + chunk2
print(full_response.pretty_repr())
```

**Output:**
```
============================ Aimessagechunk Message ============================

My name is Bob, and I'm here to help!
```

**Explanation:**
- **Streaming Responses:** The AI delivers its response in parts, allowing for incremental updates.
- **Chunk Combination:** The individual chunks are combined seamlessly to form a complete message.
- **Real-time Interaction:** This method enhances responsiveness by processing and displaying partial outputs as they become available.

---

#### **Example 2: Streaming AI Response with Tool Call Chunks**

This example shows an AI message chunk that includes a tool call. It demonstrates how tool call details can be included in a chunk and how multiple chunks, one with tool call information and one without, are combined into a single comprehensive response.

```python
from langchain_core.messages import AIMessageChunk

chunk1 = AIMessageChunk(
    content="Fetching data for your request... ",
    tool_call_chunks=[
        {
            "name": "data_fetcher",
            "args": "query=latest news",  # Changed to a string representation
            "id": "call_001"
        }
    ]
)

chunk2 = AIMessageChunk(
    content="Data fetched successfully.",
    tool_call_chunks=[]
)

# Combining the chunks
full_response = chunk1 + chunk2
print(full_response.pretty_repr())
```

**Output:**
```
============================ Aimessagechunk Message ============================

Fetching data for your request... Data fetched successfully.
Invalid Tool Calls:
  data_fetcher (call_001)
 Call ID: call_001
  Args:
    query=latest news
```

**Explanation:**
- **Partial Tool Call Inclusion:** The first chunk includes a tool call, indicating that an external data fetcher was used.
- **Incremental Data Delivery:** The AI message is streamed in parts, with tool-related metadata attached to the relevant chunk.
- **Combined Result:** The final response merges both text segments and displays the tool call information, providing a cohesive view of the complete interaction.

---

### ToolMessage

`ToolMessage` is used to pass the results of tool executions back to the chat model. It encapsulates the output from external tools (e.g., calculators, data fetchers) and links this output to the corresponding tool call via a unique identifier.

#### **Example 1: Basic Tool Response**

```python
from langchain_core.messages import ToolMessage

# A ToolMessage representing a tool call result
tool_message = ToolMessage(
    content='42',                                 # The result of the tool execution
    tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL'  # Unique ID linking to the tool call
)

print(tool_message.pretty_repr())
```

**Output:**
```
content='42' 
tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL'
```

**Explanation:**
- **Tool Integration:** The `ToolMessage` wraps the output from a tool (e.g., result of a calculation).
- **Identifier Linking:** It includes the `tool_call_id` to link the response back to the original tool call.
- **Concise Output:** Provides a simple and direct summary of the tool's result.

---

#### **Example 2: Complex Tool Response with Artifacts**

```python
from langchain_core.messages import ToolMessage

# Simulating a complex tool output
tool_output = {
    "stdout": "From the graph we can see that the correlation between x and y is ...",  # Summary of the result
    "stderr": None,  # No errors
    "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},  # Additional data (e.g., an image)
}

# Creating a ToolMessage with the tool output
tool_message = ToolMessage(
    content=tool_output["stdout"],                # The summary content
    artifact=tool_output,                         # The full tool output
    tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL'  # Unique ID linking to the tool call
)

print(tool_message)
```

**Output:**
```
content='From the graph we can see that the correlation between x and y is ...' 
tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL' 
artifact={
    "stdout": "From the graph we can see that the correlation between x and y is ...",
    "stderr": None,
    "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
}
```

**Explanation:**
- **Enhanced Output:** Besides a brief summary (`stdout`), the full tool output is preserved in the `artifact` attribute.
- **Error Handling:** Although not shown here, additional information like errors (if any) can also be included.
- **Versatility:** This approach enables the handling of both simple and complex tool outputs, making it adaptable for various use cases.

---

## Utility Functions

In addition to the core message types, LangChain offers two powerful utility functions to manage chat histories: **`trim_messages`** and **`filter_messages`**. These functions help you refine conversation histories for optimal performance and relevance.

### trim_messages: Trimming Chat History

The `trim_messages` function reduces a conversation's history based on a specified token or message count limit. It supports multiple strategies and customization options:

- **Strategies:**
  - `"first"`: Keeps messages from the beginning of the conversation.
  - `"last"`: Keeps messages from the end of the conversation.

- **Customization Options:**
  - **`max_tokens`**: The maximum number of tokens or messages allowed.
  - **`token_counter`**: A function to compute the token count. This can be a model-integrated counter (like one from ChatOpenAI) or a custom function.
  - **`start_on`**: Specifies the required starting message type (e.g., ensuring the history starts with a HumanMessage or a SystemMessage followed by a HumanMessage).
  - **`include_system`**: Determines whether the SystemMessage should be preserved.
  - **`allow_partial`**: Indicates if partial messages are allowed when the limit is reached mid-message.

#### Example 1. Trimming Based on Token Count (Strategy: "last")

```python
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.chat_models import ChatOpenAI

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage("Well, I guess they thought 'WordRope' and 'SentenceString' just didn't have the same ring to it!"),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage("Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"),
    HumanMessage("what do you call a speechless parrot"),
]

trimmed = trim_messages(
    messages,
    max_tokens=45,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o"),
    start_on="human",
    include_system=True,
    allow_partial=False,
)
print(trimmed)
```

**Output:**
```
[
    SystemMessage(content="you're a good assistant, you always respond with a joke."),
    HumanMessage(content='what do you call a speechless parrot'),
]
```

#### Example 2. Trimming Based on Message Count (Using `len` as the Token Counter)

```python
trimmed = trim_messages(
    messages,
    max_tokens=4,
    strategy="last",
    token_counter=len,
    start_on="human",
    include_system=True,
    allow_partial=False,
)
print(trimmed)
```

**Output:**
```
[
    SystemMessage(content="you're a good assistant, you always respond with a joke."),
    HumanMessage(content='and who is harrison chasing anyways'),
    AIMessage(content="Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"),
    HumanMessage(content='what do you call a speechless parrot'),
]
```

#### Example 3. Using a Custom Token Counter Function

```python
def dummy_token_counter(messages: list) -> int:
    # Each message adds a fixed number of tokens:
    # 3 (prefix) + 4 (default content) + 3 (suffix) = 10 tokens per message.
    default_content_len = 4
    default_msg_prefix_len = 3
    default_msg_suffix_len = 3

    count = 0
    for msg in messages:
        if isinstance(msg.content, str):
            count += default_msg_prefix_len + default_content_len + default_msg_suffix_len
        elif isinstance(msg.content, list):
            count += default_msg_prefix_len + len(msg.content) * default_content_len + default_msg_suffix_len
    return count

messages = [
    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="first"),
    AIMessage([
        {"type": "text", "text": "This is the FIRST 4 token block."},
        {"type": "text", "text": "This is the SECOND 4 token block."},
    ], id="second"),
    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="third"),
    AIMessage("This is a 4 token text. The full message is 10 tokens.", id="fourth"),
]

trimmed = trim_messages(
    messages,
    max_tokens=30,
    token_counter=dummy_token_counter,
    strategy="first",
    allow_partial=True,
)
print(trimmed)
```

**Output:**
```
[
    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="first"),
    AIMessage([{"type": "text", "text": "This is the FIRST 4 token block."}], id="second"),
]
```

#### trim_messages Options Comparison

| **Option**         | **Description**                                                                                                                                                                 | **Example**                                        |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `max_tokens`       | Maximum allowed token/message count (can represent tokens or message count based on the token counter function).                                                                 | `max_tokens=45` or `max_tokens=4`                   |
| `strategy`         | Determines whether to keep messages from the `"first"` (beginning) or `"last"` (end) of the conversation.                                                                      | `strategy="last"`                                 |
| `token_counter`    | A function to compute the total token count. Can be a model counter (e.g., ChatOpenAI) or a custom function like `dummy_token_counter` or Python’s `len`.                     | `token_counter=ChatOpenAI(model="gpt-4o")` or `len`   |
| `start_on`         | Specifies the required starting message type (e.g., ensuring the conversation starts with a HumanMessage).                                                                      | `start_on="human"`                                |
| `include_system`   | Whether to preserve the SystemMessage (if present) as it may contain important instructions.                                                                                    | `include_system=True`                             |
| `allow_partial`    | Indicates if partial messages are permitted when the limit is reached in the middle of a message.                                                                               | `allow_partial=False`                             |

---

### filter_messages: Filtering Chat History

The `filter_messages` function enables you to extract a subset of messages from your conversation history based on specific criteria. It supports filtering by:

- **`incl_names`**: Only include messages from senders with specified names.
- **`incl_types`**: Only include messages of specified types (e.g., `"system"`).
- **`excl_ids`**: Exclude messages with certain unique identifiers.

This functionality is particularly useful when you need to isolate key parts of a conversation for display, analysis, or further processing.

### Example 1: 

```python
from langchain_core.messages import filter_messages, AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage("you're a good assistant."),
    HumanMessage("what's your name", id="foo", name="example_user"),
    AIMessage("steve-o", id="bar", name="example_assistant"),
    HumanMessage("what's your favorite color", id="baz"),
    AIMessage("silicon blue", id="blah"),
]

filtered = filter_messages(
    messages,
    incl_names=("example_user", "example_assistant"),
    incl_types=("system",),
    excl_ids=("bar",),
)
print(filtered)
```

**Output:**
```
[
    SystemMessage("you're a good assistant."),
    HumanMessage("what's your name", id="foo", name="example_user"),
]
```

#### filter_messages Options Comparison

| **Parameter**   | **Purpose**                                                              | **Usage Example**                                     | **Notes**                                              |
|-----------------|--------------------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| `incl_names`    | Include only messages from senders with specified names.                 | `incl_names=("example_user", "example_assistant")`    | Filters based on the message's `name` attribute.       |
| `incl_types`    | Include only messages of certain types (e.g., `"system"`).               | `incl_types=("system",)`                              | Useful to retain specific message types, like system messages. |
| `excl_ids`      | Exclude messages that have specific unique identifiers.                  | `excl_ids=("bar",)`                                   | Removes messages identified by these IDs.              |

---

## Best Practices

When working with LangChain Messages, following best practices can help ensure that your chat interactions are efficient, maintain context, and remain manageable. Below are some recommended best practices along with code snippets that illustrate how to implement them.

### 1. Always Start with a SystemMessage

**Recommendation:**  
Begin your conversation with a `SystemMessage` to prime the AI. This message should set the context, provide guidelines, or deliver specific instructions for how the AI should behave throughout the conversation.

**Code Example:**

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a helpful assistant! Please provide concise answers."),
    HumanMessage(content="What is the weather like today?")
]

# The system message ensures that the assistant responds in a helpful and concise manner.
print(model.invoke(messages))
```

---

### 2. Maintain Consistent Message Attributes

**Recommendation:**  
Ensure that all messages adhere to the common structure defined by the `BaseMessage`. This includes setting attributes like `content`, `type`, and optionally `name` and `id` for traceability.

**Code Example:**

```python
from langchain_core.messages import HumanMessage, SystemMessage

# Consistent message creation ensures traceability and easier debugging.
system_msg = SystemMessage(content="You are a knowledgeable assistant.")
human_msg = HumanMessage(content="Can you explain quantum computing?")

messages = [system_msg, human_msg]
print(model.invoke(messages))
```

---

### 3. Use ChatPromptTemplate for Structured Prompts

**Recommendation:**  
Utilize the `ChatPromptTemplate` to combine multiple messages into a single, well-formatted prompt. This makes it easier to manage complex interactions and ensures that your prompt meets the expected format for chat models.

**Code Example:**

```python
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Create a unified prompt from a list of messages
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content="What are the benefits of renewable energy?")
])

formatted_prompt = chat_prompt.format_prompt()
print(formatted_prompt)
```

---

### 4. Trim Chat History to Manage Token Usage

**Recommendation:**  
When dealing with long conversations, use the `trim_messages` utility function to reduce the chat history based on token count or message count. This helps prevent exceeding token limits and improves model performance.

**Code Example:**

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages

messages = [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content="Tell me a joke."),
    AIMessage(content="Why did the chicken cross the road?"),
    HumanMessage(content="I don't get it. Explain."),
    AIMessage(content="To get to the other side!")
]

trimmed_messages = trim_messages(
    messages,
    max_tokens=50,
    strategy="last",
    token_counter=len,  # Using len for simplicity
    start_on="human",
    include_system=True,
    allow_partial=False,
)
print(trimmed_messages)
```

---

### 5. Filter Messages to Extract Relevant Content

**Recommendation:**  
Use the `filter_messages` utility function to isolate messages based on specific criteria such as sender name, message type, or message ID. This can be useful for logging, debugging, or displaying only relevant parts of a conversation.

**Code Example:**

```python
from langchain_core.messages import filter_messages, HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="System initialization message."),
    HumanMessage(content="What is the capital of France?", id="msg1", name="user_1"),
    AIMessage(content="The capital of France is Paris.", id="msg2", name="assistant_1"),
    HumanMessage(content="Thank you!", id="msg3", name="user_1"),
]

filtered_messages = filter_messages(
    messages,
    incl_names=("user_1",),
    incl_types=("system", "human"),
    excl_ids=("msg3",)
)
print(filtered_messages)
```

---

### 6. Leverage AIMessage and AIMessageChunk for Complex Responses

**Recommendation:**  
When the AI response involves tool calls or requires incremental streaming, use `AIMessage` to encapsulate full responses with metadata or `AIMessageChunk` to handle streaming responses. This approach ensures clarity and maintains context even with complex interactions.

**Code Example (Streaming Response with AIMessageChunk):**

```python
from langchain_core.messages import AIMessageChunk

chunk1 = AIMessageChunk(
    content="Analyzing the data... ",
    tool_call_chunks=[]
)
chunk2 = AIMessageChunk(
    content="Data analysis complete. The result is 42.",
    tool_call_chunks=[]
)

# Combine the chunks to form a complete response
full_response = chunk1 + chunk2
print(full_response.pretty_repr())
```

---

## Conclusion

In summary, LangChain Messages provide a structured, flexible, and powerful framework for managing conversations in AI applications. By leveraging the core message types—`SystemMessage`, `HumanMessage`, `AIMessage`, `AIMessageChunk`, and `ToolMessage`—developers can build systems that not only maintain context but also integrate complex functionalities like tool calls and streaming responses. The utility functions such as `trim_messages` and `filter_messages` further enhance this capability by ensuring that chat histories remain manageable and focused, while best practices help maintain the integrity and clarity of the conversation.

Together, these components and practices empower developers to create intelligent, responsive, and context-aware conversational agents. By understanding and utilizing the full spectrum of LangChain Messages, you can ensure that your AI-driven applications deliver consistent, accurate, and engaging user experiences.

---