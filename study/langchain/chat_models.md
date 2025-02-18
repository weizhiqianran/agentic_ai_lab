# A Comprehensive Guide to `BaseChatModel`

The `BaseChatModel` class is the backbone of LangChain’s modern chat models. It provides a unified interface for interacting with large language models (LLMs) in a chat-based format. By implementing the Runnable interface, `BaseChatModel` enables advanced features such as asynchronous processing, streaming, batching, tool integration, and structured output parsing. This comprehensive guide explores its core functionality, best practices, and real-world use cases.

---

### Table of Contents

- [Introduction](#introduction)
- [Major Member Variables](#major-member-variables)
- [Major Member Functions](#major-member-functions)
- [Examples of Chat Models](#examples-of-chat-models)
   - [Example 1: Basic Chat Model Invocation](#example-1-basic-chat-model-invocation)
   - [Example 2: Generating Structured Responses](#example-2-generating-structured-responses)
   - [Example 3: Using Chat Models with Tools](#example-3-using-chat-models-with-tools)
- [Examples of Tool Calling](#examples-of-tool-calling)
   - [Example 1: Weather Information Tool](#example-1-weather-information-tool)
   - [Example 2: Currency Conversion Tool](#example-2-currency-conversion-tool)
   - [Example 3: Restaurant Recommendation Tool](#example-3-restaurant-recommendation-tool)
- [Examples of Structured Outputs](#examples-of-structured-outputs)
   - [Example 1: Product Details Extraction](#example-1-product-details-extraction)
   - [Example 2: Answer with Justification and References](#example-2-answer-with-justification-and-references)
   - [Example 3: Product Details Extraction with Specifications](#example-3-product-details-extraction-with-specifications)
- [Examples of Multimodality](#examples-of-multimodality)
   - [Example 1: Image Captioning with ChatOpenAI](#example-1-image-captioning-with-chatopenai)
   - [Example 2: Multimodal Summarization with ChatGoogleGenerativeAI](#example-2-multimodal-summarization-with-chatgooglegenerativeai)
- [Best Practices](#best-practices)
- [Conclusion](#conclusion)

---

## Introduction

The BaseChatModel is a foundational interface in LangChain that underpins all modern chat models. It not only abstracts the interaction with large language models (LLMs) through a chat-based interface but also incorporates the Runnable Interface. This dual role enables chat models to support advanced features such as asynchronous programming, optimized batching, and streaming of results. Modern LLMs are typically accessed via chat models that accept structured messages—where each message is tagged with a role (e.g., system, human, or assistant) and content (text and, in some cases, multimodal data). In documentation and practice, you may see the terms "LLM" and "Chat Model" used interchangeably.

---

## Major Member Variables

BaseChatModel includes several critical member variables that help configure its behavior:

- **callback_manager / callbacks**:  
  Although `callback_manager` is now deprecated, callbacks remain central for tracing and logging during model execution. They allow you to track token usage, generation progress, and error handling.

- **rate_limiter**:  
  An optional variable that enforces rate limits to avoid overloading the model provider, ensuring that API requests are appropriately spaced out.

- **disable_streaming**:  
  This variable determines whether streaming is enabled. It supports boolean values and special flags (e.g., `"tool_calling"`) to selectively disable streaming when tools are involved.

- **Model Parameters**:  
  BaseChatModel holds various parameters (such as model name, temperature, max_tokens, and timeout) that control the behavior of the underlying model. These are used not only for configuration but also for tracing and caching of model outputs.

---

## Major Member Functions

The BaseChatModel interface defines several imperative and declarative methods. Key functions include:

- **invoke**:  
  The primary synchronous method for interacting with a chat model. It takes input messages (as strings, dicts, or a LangChain PromptValue) and returns a generated message as output.

- **ainvoke**:  
  An asynchronous counterpart to invoke. It wraps the synchronous method for use in async environments.

- **stream / astream / astream_events**:  
  These methods support streaming output. The `stream` method yields message chunks as the model generates text, while `astream` provides the same functionality asynchronously. The `astream_events` method additionally emits events (e.g., start, stream, and end events) during model execution.

- **batch / abatch / batch_as_completed / abatch_as_completed**:  
  Methods to batch multiple calls together, optimizing throughput and efficiency. They are useful when you have a collection of prompts that need to be processed concurrently.

- **bind_tools**:  
  This method allows you to integrate external tools into the chat model’s execution context. Tools can be called by the model during a conversation, enabling functionalities like API calls or database queries.

- **with_structured_output**:  
  A declarative wrapper that instructs the model to return structured data (e.g., JSON or a Pydantic model instance) according to a specified schema. This is particularly useful for extraction tasks where a precise format is required.

---

## Examples of Chat Models

This section provides examples of using various chat models, including OpenAI's GPT, Anthropic's Claude, Google's Gemini, and DeepSeek's ChatDeepSeek. These models can be used for natural language processing (NLP) tasks such as answering questions, generating structured responses, and interacting with external tools.

### Example 1: Basic Chat Model Invocation

This example demonstrates how to initialize and use different chat models from OpenAI, Anthropic, Google, and DeepSeek. Each model is initialized with an API key, and a prompt is sent to generate a response.

```python
from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

# Retrieve API keys for different models
openai_api_key = userdata.get('api-key-openai')
anthropic_api_key = userdata.get('api-key-anthropic')
gemini_api_key = userdata.get('api-key-gemini')
deepseek_api_key = userdata.get('api-key-deepseek')

# Initialize OpenAI Chat Model
llm_openai = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4o-mini",
    temperature=0
)

# Initialize Anthropic's Claude Chat Model
llm_anthropic = ChatAnthropic(
    api_key=anthropic_api_key,
    model="claude-3-5-sonnet-latest",
    temperature=0
)

# Initialize Google's Gemini Chat Model
llm_gemini = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.0-flash",
    temperature=0
)

# Initialize DeepSeek Chat Model
llm_deepseek = ChatDeepSeek(
    api_key=deepseek_api_key,
    model="deepseek-chat",
    temperature=0
)

# Generate responses from each model with sample prompts
response_openai = llm_openai.invoke("Tell me a joke about cats")
print("OpenAI Response:", response_openai)

response_anthropic = llm_anthropic.invoke("What is the capital of France?")
print("Anthropic Response:", response_anthropic)

response_gemini = llm_gemini.invoke("Summarize the latest news in technology.")
print("Google Generative AI Response:", response_gemini)

response_deepseek = llm_deepseek.invoke("Explain the significance of quantum computing in simple terms.")
print("DeepSeek Response:", response_deepseek)
```

**Output:** 
```
OpenAI Response: content='Why was the cat sitting on the computer?\n\nBecause it wanted to keep an eye on the mouse!' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 13, 'total_tokens': 34, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_00428b782a', 'finish_reason': 'stop', 'logprobs': None} id='run-4b04b24a-270b-4bd2-bdc9-8a1d2c822090-0' usage_metadata={'input_tokens': 13, 'output_tokens': 21, 'total_tokens': 34, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

Anthropic Response: content='The capital of France is Paris.' additional_kwargs={} response_metadata={'id': 'msg_014ijmVBVLDZt7s3dmnY4P6Z', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 14, 'output_tokens': 10}} id='run-e79d3614-c523-4c90-9ac1-4cfd037fa64b-0' usage_metadata={'input_tokens': 14, 'output_tokens': 10, 'total_tokens': 24, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}

Google Generative AI Response: content="Okay, here's a summary of some of the latest tech news, hitting the major trends and headlines:\n\n**Artificial Intelligence (AI):**\n\n*   **AI Model Development & Competition:** The race to develop more powerful and efficient AI models continues. Google's Gemini is still a major focus, with ongoing improvements and integrations across Google products. I hope this is helpful!" additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-6d461df2-b952-494c-a524-835e4daeda45-0' usage_metadata={'input_tokens': 8, 'output_tokens': 753, 'total_tokens': 761, 'input_token_details': {'cache_read': 0}}

DeepSeek Response: content='Quantum computing is a new way of processing information that uses the principles of quantum mechanics, which is the science of how very small particles like atoms and electrons behave. Unlike regular computers, which use bits that can be either 0 or 1, quantum computers use quantum bits, or qubits, which can be 0, 1, or both at the same time. This is called superposition.\n\nHere’s why quantum computing is ... But the potential is huge, and it could change the way we solve problems and process information in the future.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 313, 'prompt_tokens': 13, 'total_tokens': 326, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 13}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3a5770e1b4', 'finish_reason': 'stop', 'logprobs': None} id='run-5804676f-e655-48ac-9b7b-cbe89e4b4b7a-0' usage_metadata={'input_tokens': 13, 'output_tokens': 313, 'total_tokens': 326, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}
```

**Explanation:**  
- This example demonstrates initializing and invoking different chat models.
- The models process various prompts, including jokes, general knowledge, and news summaries.
- Each model provides its response based on its training data and capabilities.

---

### Example 2: Generating Structured Responses

This example shows how to generate structured responses using a Pydantic model. Structured outputs ensure that the responses follow a predefined format, such as a JSON structure.

```python
from pydantic import BaseModel

# Define a Pydantic model for structured joke responses
class JokeResponse(BaseModel):
    setup: str
    punchline: str

# Generate structured output for each model

# OpenAI structured response
structured_llm_openai = llm_openai.with_structured_output(JokeResponse)
response_openai = structured_llm_openai.invoke("Tell me a joke about cats")
print("OpenAI Structured Response:", response_openai)

# Anthropic structured response
structured_llm_anthropic = llm_anthropic.with_structured_output(JokeResponse)
response_anthropic = structured_llm_anthropic.invoke("Tell me a joke about dogs")
print("Anthropic Structured Response:", response_anthropic)

# Google Generative AI structured response
structured_llm_gemini = llm_gemini.with_structured_output(JokeResponse)
response_gemini = structured_llm_gemini.invoke("Tell me a joke about birds")
print("Google Generative AI Structured Response:", response_gemini)

# DeepSeek structured response
structured_llm_deepseek = llm_deepseek.with_structured_output(JokeResponse)
response_deepseek = structured_llm_deepseek.invoke("Tell me a joke about fish")
print("DeepSeek Structured Response:", response_deepseek)
```

**Output:** 
```
OpenAI Structured Response: setup='Why was the cat sitting on the computer?' punchline='Because it wanted to keep an eye on the mouse!'

Anthropic Structured Response: setup='What do you call a dog that does magic tricks?' punchline='A labracadabrador!'

Google Generative AI Structured Response: setup='Why are birds such good musicians?' punchline="Because they're always tweeting!"

DeepSeek Structured Response: setup='Why are fish so bad at basketball?' punchline='Because they’re afraid of the net!'
```

**Explanation:**  
- Instead of free-text output, responses are structured as JSON objects.
- This is useful for applications requiring predictable response formats.
- Pydantic ensures the responses conform to a predefined schema.

---

### Example 3: Using Chat Models with Tools

This example demonstrates how to integrate external tools (functions) into chat models, allowing them to retrieve real-world data such as weather information.

```python
from langchain_core.tools import StructuredTool

# Define a structured tool (function) that the model can call
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Simulate a weather API response
    return f"The weather in {location} is sunny and 75°F."

# Wrap the function in a StructuredTool
weather_tool = StructuredTool.from_function(
    func=get_current_weather,
    name="get_current_weather",
    description="Get the current weather for a given location."
)

# Bind the tool to each model
# OpenAI with tool binding
llm_openai_with_tools = llm_openai.bind_tools([weather_tool])
response_openai = llm_openai_with_tools.invoke("What's the weather like in San Francisco?")
print("OpenAI Response with Tools:", response_openai)

# Anthropic with tool binding
llm_anthropic_with_tools = llm_anthropic.bind_tools([weather_tool])
response_anthropic = llm_anthropic_with_tools.invoke("What's the weather like in New York?")
print("Anthropic Response with Tools:", response_anthropic)

# Google Generative AI with tool binding
llm_gemini_with_tools = llm_gemini.bind_tools([weather_tool])
response_gemini = llm_gemini_with_tools.invoke("What's the weather like in London?")
print("Google Generative AI Response with Tools:", response_gemini)

# DeepSeek with tool binding
llm_deepseek_with_tools = llm_deepseek.bind_tools([weather_tool])
response_deepseek = llm_deepseek_with_tools.invoke("What's the weather like in Tokyo?")
print("DeepSeek Response with Tools:", response_deepseek)
```

**Output:**  
```
OpenAI Response with Tools: content='' additional_kwargs={'tool_calls': [{'id': 'call_FSoa9Z2it87kvtRLLEE2eLW0', 'function': {'arguments': '{"location":"San Francisco"}', 'name': 'get_current_weather'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 55, 'total_tokens': 72, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_13eed4fce1', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-bbd49c5a-9550-44e8-9bc3-8d954171ad6c-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'San Francisco'}, 'id': 'call_FSoa9Z2it87kvtRLLEE2eLW0', 'type': 'tool_call'}] usage_metadata={'input_tokens': 55, 'output_tokens': 17, 'total_tokens': 72, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

Anthropic Response with Tools: content=[{'text': "I'll check the current weather in New York for you.", 'type': 'text'}, {'id': 'toolu_01Uipx4fxKWD7X7txt6w1eVk', 'input': {'location': 'New York'}, 'name': 'get_current_weather', 'type': 'tool_use'}] additional_kwargs={} response_metadata={'id': 'msg_018mek8caSagDmYdNJY7mPZJ', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 387, 'output_tokens': 69}} id='run-854b1e53-bc0b-4b82-9f13-4b8c65c20344-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'New York'}, 'id': 'toolu_01Uipx4fxKWD7X7txt6w1eVk', 'type': 'tool_call'}] usage_metadata={'input_tokens': 387, 'output_tokens': 69, 'total_tokens': 456, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}

Google Generative AI Response with Tools: content='' additional_kwargs={'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "London"}'}} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-f47c5cb2-f6dc-48ba-a8bc-ecf9f8c22258-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'London'}, 'id': '870f96ae-5ee6-4bd0-ad8e-177aa83370bb', 'type': 'tool_call'}] usage_metadata={'input_tokens': 25, 'output_tokens': 7, 'total_tokens': 32, 'input_token_details': {'cache_read': 0}}

DeepSeek Response with Tools: content='' additional_kwargs={'tool_calls': [{'id': 'call_0_cca6e8bb-55f6-4bd6-8195-ae74ad8e0f50', 'function': {'arguments': '{"location":"Tokyo"}', 'name': 'get_current_weather'}, 'type': 'function', 'index': 0}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 114, 'total_tokens': 134, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 64}, 'prompt_cache_hit_tokens': 64, 'prompt_cache_miss_tokens': 50}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3a5770e1b4', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-026483b3-49a3-4785-935c-080249b5b932-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_0_cca6e8bb-55f6-4bd6-8195-ae74ad8e0f50', 'type': 'tool_call'}] usage_metadata={'input_tokens': 114, 'output_tokens': 20, 'total_tokens': 134, 'input_token_details': {'cache_read': 64}, 'output_token_details': {}}
```

**Explanation:**  
- A structured tool (`get_current_weather`) is defined and wrapped using `StructuredTool`.
- Each model is bound to this tool, enabling it to fetch weather data dynamically.
- This feature enhances the model's functionality, allowing real-time or external interactions.

---

## Examples of Tool Calling

Tool calling is a powerful feature of BaseChatModel that enables a chat model to delegate tasks to external tools. For instance, you might bind a weather lookup tool to a chat model:

### Example 1: Weather Information Tool
 
This example demonstrates how to create a weather information tool using `StructuredTool` and bind it to the `ChatGoogleGenerativeAI` model. The tool takes a location as input and returns the current weather for that location.

```python
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Define the input schema for the weather tool using Pydantic
class WeatherInput(BaseModel):
    location: str = Field(description="The city or region to get the weather for.")

# Define the weather tool function
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Simulate a weather API response
    return f"The weather in {location} is sunny and 75°F."

# Wrap the function in a StructuredTool
weather_tool = StructuredTool.from_function(
    func=get_current_weather,
    name="get_current_weather",
    description="Get the current weather for a given location.",
    args_schema=WeatherInput
)

# Initialize the ChatGoogleGenerativeAI model
llm_gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash", temperature=0)

# Bind the tool to the model
llm_gemini_with_tools = llm_gemini.bind_tools([weather_tool])

# Invoke the model with a prompt
response = llm_gemini_with_tools.invoke("What's the weather like in San Francisco?")
print("Response:", response)
```

**Output:**
```
Response: content='' additional_kwargs={'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco"}'}} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-d65d8696-527e-4f31-8ab1-31dcfa1b7b30-0' tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'San Francisco'}, 'id': '58e812dc-3c0a-42a1-9f01-522e2540e558', 'type': 'tool_call'}] usage_metadata={'input_tokens': 36, 'output_tokens': 8, 'total_tokens': 44, 'input_token_details': {'cache_read': 0}}
```

**Explanation:**  
- A `WeatherInput` Pydantic model is used to define the input schema for the weather tool.
- The `get_current_weather` function is wrapped in a `StructuredTool` with a description and input schema.
- The tool is bound to the `ChatGoogleGenerativeAI` model using `bind_tools()`.
- When the model is invoked, it uses the tool to generate a structured response.

---

### Example 2: Currency Conversion Tool

This example creates a currency conversion tool that converts an amount from one currency to another. The tool is bound to the `ChatGoogleGenerativeAI` model.

```python
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Define the input schema for the currency conversion tool
class CurrencyConversionInput(BaseModel):
    amount: float = Field(description="The amount of money to convert.")
    from_currency: str = Field(description="The currency to convert from.")
    to_currency: str = Field(description="The currency to convert to.")

# Define the currency conversion tool function
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another."""
    # Simulate a currency conversion API response
    conversion_rate = 0.85  # Example: 1 USD = 0.85 EUR
    converted_amount = amount * conversion_rate
    return f"{amount} {from_currency} is equal to {converted_amount} {to_currency}."

# Wrap the function in a StructuredTool
currency_tool = StructuredTool.from_function(
    func=convert_currency,
    name="convert_currency",
    description="Convert an amount from one currency to another.",
    args_schema=CurrencyConversionInput
)

# Initialize the ChatGoogleGenerativeAI model
llm_gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash", temperature=0)

# Bind the tool to the model
llm_gemini_with_tools = llm_gemini.bind_tools([currency_tool])

# Invoke the model with a prompt
response = llm_gemini_with_tools.invoke("Convert 100 USD to EUR.")
print("Response:", response)
```

**Output:**
```
Response: content='' additional_kwargs={'function_call': {'name': 'convert_currency', 'arguments': '{"to_currency": "EUR", "amount": 100.0, "from_currency": "USD"}'}} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-442e0c12-33e5-4e05-88bf-2a9a453e992c-0' tool_calls=[{'name': 'convert_currency', 'args': {'to_currency': 'EUR', 'amount': 100.0, 'from_currency': 'USD'}, 'id': '4870ca03-5b77-4bdb-a28d-df81e4af91c9', 'type': 'tool_call'}] usage_metadata={'input_tokens': 54, 'output_tokens': 12, 'total_tokens': 66, 'input_token_details': {'cache_read': 0}}
```

**Explanation:**  
- A `CurrencyConversionInput` Pydantic model defines the input schema for the currency conversion tool.
- The `convert_currency` function is wrapped in a `StructuredTool` with a description and input schema.
- The tool is bound to the `ChatGoogleGenerativeAI` model, which uses it to perform currency conversion.

---

### Example 3: Restaurant Recommendation Tool

This example creates a restaurant recommendation tool that suggests restaurants based on a given cuisine and location. The tool is bound to the `ChatGoogleGenerativeAI` model.

```python
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Define the input schema for the restaurant recommendation tool
class RestaurantInput(BaseModel):
    cuisine: str = Field(description="The type of cuisine to recommend.")
    location: str = Field(description="The city or region to recommend restaurants in.")

# Define the restaurant recommendation tool function
def recommend_restaurant(cuisine: str, location: str) -> str:
    """Recommend a restaurant based on cuisine and location."""
    # Simulate a restaurant recommendation API response
    return f"For {cuisine} cuisine in {location}, I recommend 'The Golden Spoon'."

# Wrap the function in a StructuredTool
restaurant_tool = StructuredTool.from_function(
    func=recommend_restaurant,
    name="recommend_restaurant",
    description="Recommend a restaurant based on cuisine and location.",
    args_schema=RestaurantInput
)

# Initialize the ChatGoogleGenerativeAI model
llm_gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash", temperature=0)

# Bind the tool to the model
llm_gemini_with_tools = llm_gemini.bind_tools([restaurant_tool])

# Invoke the model with a prompt
response = llm_gemini_with_tools.invoke("Can you recommend an Italian restaurant in New York?")
print("Response:", response)
```

**Output:**
```
Response: content='' additional_kwargs={'function_call': {'name': 'recommend_restaurant', 'arguments': '{"location": "New York", "cuisine": "Italian"}'}} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-ef446b55-faaf-4b07-b0ac-66c101b7c0aa-0' tool_calls=[{'name': 'recommend_restaurant', 'args': {'location': 'New York', 'cuisine': 'Italian'}, 'id': '9f876b41-8268-4bbb-8d0e-f1ebf13b3775', 'type': 'tool_call'}] usage_metadata={'input_tokens': 42, 'output_tokens': 8, 'total_tokens': 50, 'input_token_details': {'cache_read': 0}}
```

**Explanation:**  
- A `RestaurantInput` Pydantic model defines the input schema for the restaurant recommendation tool.
- The `recommend_restaurant` function is wrapped in a `StructuredTool` with a description and input schema.
- The tool is bound to the `ChatGoogleGenerativeAI` model, which uses it to recommend restaurants based on user input.

---

## Examples of Structured Outputs

Structured outputs allow the model to return information in a well-defined format. Using `with_structured_output`, you can ensure the output adheres to a specific schema.

### Example 1: Product Details Extraction

Here, we design a Pydantic model to extract product details from a description. The model captures the product's name, price, and description. This is useful for tasks like extracting structured information from marketing content.

```python
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Define the Pydantic schema for product details.
class ProductDetails(BaseModel):
    product_name: str = Field(description="The name of the product")
    price: float = Field(description="The price of the product")
    description: str = Field(description="A short description of the product")

# Initialize the ChatGoogleGenerativeAI model.
llm_gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash", temperature=0)

# Wrap the model to return structured output as defined by the ProductDetails schema.
structured_llm = llm_gemini.with_structured_output(ProductDetails)

# Invoke the model with a sample prompt.
result = structured_llm.invoke(
    "Extract the product details: The new UltraPhone X costs $999.99 and features a 6.5-inch display with advanced AI capabilities."
)

print(result)
```

**Output:**
```
product_name='UltraPhone X' price=999.99 description='6.5-inch display with advanced AI capabilities'
```

**Explanation:**  
The prompt provides a product description. The structured output, based on the `ProductDetails` model, extracts the product name, price, and a brief description. This output is validated and returned as a Pydantic instance, making it straightforward to use in downstream applications.

---

### Example 2: Answer with Justification and References

In this example, the output schema includes not only the answer and justification but also a list of reference URLs. The `AnswerWithJustification` model nests a list field to capture multiple reference links.

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI

# Define a Pydantic schema for references.
class Reference(BaseModel):
    title: str = Field(description="Title of the reference")
    url: str = Field(description="URL of the reference")

# Define the main Pydantic schema with a nested list of Reference objects.
class AnswerWithJustification(BaseModel):
    answer: str = Field(description="The answer to the question")
    justification: str = Field(description="Explanation supporting the answer")
    references: List[Reference] = Field(default_factory=list, description="List of references for further reading")

# Initialize the ChatGoogleGenerativeAI model.
llm_gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash", temperature=0)

# Wrap the model to return structured output matching the Pydantic model.
structured_llm = llm_gemini.with_structured_output(AnswerWithJustification)

# Invoke the model with a sample prompt.
result = structured_llm.invoke("What is the capital of France? Provide the answer, justification, and at least one reference.")

print(result)
```

**Output:**
```
answer='Paris' 
justification='Paris is the capital and most populous city of France.' 
references=[Reference(title='Wikipedia', url='https://en.wikipedia.org/wiki/Paris')]
```

**Explanation:**  
This example uses a nested `Reference` model to capture additional reference links. The main schema returns an answer, a justification, and a list of references. The structured output ensures that the response adheres to the specified schema.

---

### Example 3: Product Details Extraction with Specifications

This example uses a Pydantic model to extract detailed product information. The output schema includes a nested model for product specifications and a dictionary to capture additional metadata. The prompt has been revised to ensure that the model returns non-empty values for both specifications and metadata.

```python
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI

# Define a nested model for product specifications.
class Specification(BaseModel):
    key: str = Field(description="Specification name")
    value: str = Field(description="Specification value")

# Define the main product details model, including a list of specifications and a metadata dict.
class ProductDetails(BaseModel):
    product_name: str = Field(description="The name of the product")
    price: float = Field(description="The price of the product")
    description: str = Field(description="A short description of the product")
    specifications: List[Specification] = Field(
        default_factory=list, 
        description="List of product specifications"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, 
        description="Additional product metadata"
    )

# Initialize the ChatGoogleGenerativeAI model.
llm_gemini = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash", temperature=0)

# Wrap the model to return structured output matching the ProductDetails schema.
structured_llm = llm_gemini.with_structured_output(ProductDetails)

# Revised prompt that instructs the model to extract product details along with specifications and metadata.
prompt = (
    "Extract product details with the following information:\n"
    "1. Product name\n"
    "2. Price (as a float)\n"
    "3. A short description\n"
    "4. Specifications: provide a list with keys 'Display', 'Camera', and 'Connectivity'\n"
    "5. Metadata: include a key 'shipping' with the value 'free'\n\n"
    "Product details: The new UltraPhone X is priced at $999.99. "
    "It features a 6.5-inch display, a 48MP camera, and supports 5G. "
    "Additionally, it comes with free shipping."
)

result = structured_llm.invoke(prompt)
print(result)
```

**Output:**
```
product_name='UltraPhone X' price=999.99 description='The new UltraPhone X features a 6.5-inch display, a 48MP camera, and supports 5G.' 
specifications=[Specification(key='Display', value='6.5-inch'), Specification(key='Camera', value='48MP'), Specification(key='Connectivity', value='5G')] 
metadata={}
```

**Explanation:**  
In this revised example, the prompt explicitly instructs the model to extract and include:
- The product name, price, and a short description.
- A list of specifications with keys for "Display," "Camera," and "Connectivity."
- A metadata dictionary containing a key `shipping` with the value `"free"`.

---

## Examples of Multimodality

While BaseChatModel primarily handles text, modern implementations can also support multimodal inputs. Some chat models are now capable of processing images, audio, or video alongside text. For example, a multimodal chat model might accept an image message along with text to perform tasks such as image captioning or visual question answering.

### Example 1: Image Captioning with ChatOpenAI

This example uses the ChatOpenAI model (e.g., GPT-4o) to generate a caption for an image. The model is provided with a system prompt to act as an image captioning assistant and a human message that includes both a textual instruction and an image URL via additional keyword arguments.

```python
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Initialize the ChatOpenAI model.
llm_openai = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4o",
    temperature=1.0
)

# Prepare messages with a multimodal prompt.
messages = [
    SystemMessage(content="You are an image captioning assistant."),
    HumanMessage(
        content="Please generate a creative caption for this image.",
        additional_kwargs={"image": "https://cbx-prod.b-cdn.net/COLOURBOX45651352.jpg?width=400&height=400&quality=70"}
    )
]

# Invoke the model with the multimodal prompt.
response = llm_openai.invoke(messages)
print("ChatOpenAI Response:", response)
```

**Output:**
```
ChatOpenAI Response: content='Certainly! Could you please describe the image you would like me to caption?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 28, 'total_tokens': 44, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_523b9b6e5f', 'finish_reason': 'stop', 'logprobs': None} id='run-5e8cf309-69dd-43a8-a80f-44071e23bf29-0' usage_metadata={'input_tokens': 28, 'output_tokens': 16, 'total_tokens': 44, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
```

**Explanation:**  
In this example, the ChatOpenAI model is instructed to act as an image captioning assistant. The human message contains both text and an image URL (passed via `additional_kwargs`). The model processes both inputs and returns a descriptive caption for the image.

---

### Example 2: Multimodal Summarization with ChatGoogleGenerativeAI

This example demonstrates how to use the ChatGoogleGenerativeAI model to summarize information from both textual and visual inputs. The prompt provides a description of an image (via URL) along with additional text details. The model returns a concise summary that combines cues from both modalities.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Initialize the ChatGoogleGenerativeAI model.
llm_gemini = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.0-flash",
    temperature=1.0
)

# Prepare messages with a multimodal prompt.
messages = [
    SystemMessage(content="You are a summarization assistant that processes both text and images."),
    HumanMessage(
        content=(
            "Summarize the key points: "
            "The image shows a bustling city street with lively crowds, "
            "and the text describes the city’s rich cultural heritage and dynamic urban life."
        ),
        additional_kwargs={"image": "https://img.freepik.com/free-photo/european-mother-african-son-family-summer-park-people-plays-with-ball_1157-41248.jpg"}
    )
]

# Invoke the model with the multimodal prompt.
response = llm_gemini.invoke(messages)
print("ChatGoogleGenerativeAI Response:", response)
```

**Output:**
```
ChatGoogleGenerativeAI Response: content='The image and text depict a vibrant city, characterized by a lively street scene with bustling crowds, and a rich cultural heritage alongside a dynamic urban environment.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-a56f51dd-ccb4-4a4b-8248-ba45eff83bdd-0' usage_metadata={'input_tokens': 46, 'output_tokens': 31, 'total_tokens': 77, 'input_token_details': {'cache_read': 0}}
```

**Explanation:**  
Here, the ChatGoogleGenerativeAI model receives a multimodal input: a human message that includes both textual content and an image URL. The system message sets the role for summarization. By combining visual cues from the image and details from the text, the model generates a concise summary that encapsulates the essence of the provided information.

---

## Best Practices

When working with `BaseChatModel`, follow these best practices to ensure efficient, robust, and scalable implementations:

### 1. **Leverage Structured Outputs**  
Use `with_structured_output` to enforce precise data extraction and validation. This is especially useful when integrating with downstream systems that require consistent data formats.

```python
from pydantic import BaseModel

# Define a Pydantic model for structured output
class UserInfo(BaseModel):
    name: str
    age: int
    email: str

# Bind structured output to the model
structured_model = chat_model.with_structured_output(UserInfo)
response = structured_model.invoke("Extract user info: John, 30, john@example.com")
print(response)
```

**Why:**  
- Ensures consistent and validated outputs.  
- Simplifies integration with APIs or databases.  

---

### 2. **Integrate Tools Wisely**  
Use `bind_tools` to extend the model's capabilities with external tools. Ensure tools have clear descriptions and descriptive names for better model understanding.

```python
from langchain_core.tools import StructuredTool

# Define a tool for calculating square roots
def square_root(number: float) -> float:
    """Calculate the square root of a number."""
    return number**0.5

# Wrap the tool in a StructuredTool
sqrt_tool = StructuredTool.from_function(
    func=square_root,
    name="square_root",
    description="Calculate the square root of a number."
)

# Bind the tool to the model
model_with_tools = chat_model.bind_tools([sqrt_tool])
response = model_with_tools.invoke("What is the square root of 64?")
print(response)
```

**Why:**  
- Extends model functionality with custom logic.  
- Clear descriptions help the model understand when and how to use tools.  

---

### 3. **Utilize Streaming**  
For real-time applications, use streaming methods (`stream`, `astream`, `astream_events`) to process outputs incrementally and reduce latency.

```python
# Stream responses in real-time
for chunk in chat_model.stream("Explain quantum computing in simple terms."):
    print(chunk.content, end="", flush=True)
```

**Why:**  
- Improves user experience by displaying outputs as they are generated.  
- Reduces perceived latency for long-running tasks.  

---

### 4. **Manage Context Window**  
Be mindful of the model's token limit. Use memory management techniques to trim or summarize conversation history when necessary.

```python
from langchain.memory import ConversationBufferWindowMemory

# Use a sliding window memory to limit context size
memory = ConversationBufferWindowMemory(k=3)  # Keeps only the last 3 exchanges
memory.save_context({"input": "Hi!"}, {"output": "Hello!"})
memory.save_context({"input": "How are you?"}, {"output": "I'm good, thanks!"})
memory.save_context({"input": "What's your name?"}, {"output": "I'm an AI."})

# Load memory into the model
response = chat_model.invoke("What did I say earlier?", memory=memory)
print(response)
```

**Why:**  
- Prevents errors caused by exceeding token limits.  
- Maintains relevant context without overwhelming the model.  

---

### 5. **Optimize Batching**  
Use batching methods (`batch`, `abatch`) to process multiple prompts efficiently, improving throughput and reducing API calls.

```python
# Batch process multiple prompts
prompts = [
    "Tell me a joke about cats.",
    "Explain the theory of relativity.",
    "What is the capital of France?"
]
responses = chat_model.batch(prompts)
for response in responses:
    print(response.content)
```

**Why:**  
- Reduces latency by processing multiple inputs in parallel.  
- Improves efficiency for bulk operations.  

---

### 6. **Handle Errors Gracefully**  
Implement error handling for robust interactions. Use callbacks or structured error handlers to manage tool errors or API failures.

```python
from langchain_core.tools import ToolException

# Define a tool with error handling
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ToolException("Division by zero is not allowed.")
    return a / b

# Wrap the tool in a StructuredTool
divide_tool = StructuredTool.from_function(
    func=divide_numbers,
    name="divide_numbers",
    description="Divide two numbers."
)

# Bind the tool to the model
model_with_tools = chat_model.bind_tools([divide_tool])

# Handle tool errors gracefully
try:
    response = model_with_tools.invoke("Divide 10 by 0.")
    print(response)
except ToolException as e:
    print(f"Error: {e}")
```

**Why:**  
- Ensures the application remains robust and user-friendly.  
- Prevents crashes due to unexpected errors.  

---

### 7. **Use Callbacks for Monitoring**  
Leverage callbacks to monitor and log interactions, enabling debugging and performance analysis.

```python
from langchain_core.callbacks import StdOutCallbackHandler

# Use a callback handler to log interactions
callback_handler = StdOutCallbackHandler()
response = chat_model.invoke("Tell me a joke.", callbacks=[callback_handler])
```

**Why:**  
- Provides visibility into model interactions.  
- Facilitates debugging and performance optimization.  

---

## Conclusion

BaseChatModel is the backbone of modern LangChain chat models, providing a robust, flexible interface that supports a wide range of functionalities—from synchronous and asynchronous execution to streaming, batching, tool calling, and multimodality. Its design facilitates rich, conversational interactions with LLMs while ensuring that outputs can be structured and validated for downstream applications.

By understanding and leveraging the major member variables, functions, and best practices outlined in this article, developers can build sophisticated applications that harness the full power of modern LLMs via the BaseChatModel interface.

---