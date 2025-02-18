# A Comprehensive Guide to `StructuredTool`

In this article, we will introduce the `langchain_core.tools.structured.StructuredTool` class, a key component in the LangChain ecosystem that enables the creation of structured tools capable of processing and managing inputs efficiently. We will discuss its major member functions and variables, illustrate its use with examples, and provide helpful tables for quick reference.

---

### Table of Contents

- [Introduction](#introduction)
- [Major Member Variables](#major-member-variables)
- [Major Member Functions](#major-member-functions)
- [Practical Examples](#practical-examples)
  - [Example 1: Simple Calculator Tool](#example-1-simple-calculator-tool)
  - [Example 2: Weather Forecast Tool](#example-2-weather-forecast-tool)
  - [Example 3: Text Summarization Tool](#example-3-text-summarization-tool)
  - [Example 4: Currency Conversion Tool](#example-4-currency-conversion-tool)
  - [Example 5: Process Numbers Tool](#example-5-process-numbers-tool)
- [Best Practices](#best-practices)
- [Conclusion](#conclusion)

---

## Introduction

The `StructuredTool` class is a core component of the LangChain framework, providing a robust and flexible interface for defining and invoking tools. Built on top of `BaseTool`, `StructuredTool` not only manages the tool's input schema using Pydantic models for data validation but also supports both synchronous and asynchronous execution modes. This dual capability allows developers to tailor tool behavior based on specific performance requirements or integration contexts.

Key features of `StructuredTool` include:

- **Schema Management:**  
  Utilizes Pydantic models to define and validate the tool's input parameters. This ensures that all inputs meet the expected format, improving the reliability of tool execution.

- **Docstring Parsing:**  
  Automatically extracts descriptive information from function docstrings. When using the `from_function` method, the tool can infer both the input schema and a user-friendly description directly from the function’s signature and docstring.

- **Execution Flexibility:**  
  Supports both synchronous (`func`) and asynchronous (`coroutine`) execution. This means that tools can be integrated seamlessly into a variety of workflows, whether they need to run in blocking or non-blocking contexts.

- **Ease of Integration:**  
  With the helper method `from_function`, wrapping a regular Python function or coroutine into a `StructuredTool` is straightforward. Additional metadata such as versioning or author information can also be supplied, making the tool self-documenting and easy to maintain.

By managing input validation, documentation, and execution strategy in a unified interface, `StructuredTool` simplifies the process of developing, deploying, and maintaining custom tools in LangChain-powered applications.

---

## Major Member Variables

The `StructuredTool` object has several important member variables that dictate its behavior:

| **Member Variable** | **Type** | **Description** |
|---------------------|----------|-----------------|
| `description`       | `str`    | A textual description of what the tool does. |
| `args_schema`       | `Annotated[TypeBaseModel, SkipValidation()]` | Defines the schema for the tool's input arguments using a Pydantic model. |
| `func`              | `Optional[Callable[..., Any]]` | The synchronous function that implements the tool’s logic. |
| `coroutine`         | `Optional[Callable[..., Awaitable[Any]]]` | The asynchronous function counterpart to `func`, if asynchronous execution is desired. |

These variables allow the `StructuredTool` to enforce input validation, provide clear documentation through descriptions, and support both sync and async operations.

---

## Major Member Functions

The `StructuredTool` class offers several key member functions to interact with the tool:

- **`ainvoke`**  
  This asynchronous method is responsible for invoking the tool. It checks if an asynchronous function (`coroutine`) is available. If not, it falls back to executing the synchronous version (`_run`) in an executor.

- **`_run`**  
  The synchronous execution method. It directly calls the provided function (`func`) with the required arguments. If no function is provided, it raises a `NotImplementedError`.

- **`_arun`**  
  Similar to `_run`, but designed for asynchronous execution. It either directly calls the asynchronous function (`coroutine`) or, if not present, falls back to a default asynchronous implementation that delegates to `_run` on a separate thread.

- **`from_function` (classmethod)**  
  A helper method for creating a `StructuredTool` instance from a plain Python function or coroutine. It automatically infers the input schema from the function's signature and docstring (if configured to do so), and wraps the function into a tool with an associated description and arguments schema.

These functions ensure that the tool is flexible and can be integrated seamlessly into various workflows, whether the execution is synchronous or asynchronous.

---

## Practical Examples

Below is an example of how to create and use a simple addition tool using `StructuredTool.from_function`.

### Example 1: Simple Calculator Tool

A tool that performs basic arithmetic operations like addition, subtraction, multiplication, and division.

```python
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

# Define input schema using Pydantic for input validation
class CalculatorInput(BaseModel):
    a: float = Field(description="The first number")  # First operand
    b: float = Field(description="The second number")  # Second operand
    operation: str = Field(description="The operation to perform: add, subtract, multiply, divide")  # Operation type

# Function performing basic arithmetic operations
def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")  # Handle division by zero
        return a / b
    else:
        raise ValueError("Invalid operation")  # Handle invalid operations

# Create a StructuredTool instance from the calculator function
calculator_tool = StructuredTool.from_function(
    func=calculator,  # Synchronous function to wrap
    name="calculator",  # Tool name
    description="Performs basic arithmetic operations: add, subtract, multiply, divide",  # Tool description
    args_schema=CalculatorInput,  # Pydantic schema for input validation
)

# Example usage: Running the tool with valid input
result = calculator_tool.run({"a": 10, "b": 5, "operation": "add"})
print(result)  # Expected output: 15.0
```

**Output:**
```
15.0
```

**Explanation:**  

- **Input Schema Definition:**  
  - A `CalculatorInput` Pydantic model is defined with three fields:  
    - `a`: the first number (float).  
    - `b`: the second number (float).  
    - `operation`: a string indicating the arithmetic operation (add, subtract, multiply, divide).

- **Function Implementation:**  
  - The `calculator` function accepts three parameters corresponding to the schema and performs the appropriate arithmetic operation.  
  - It handles division by zero and raises errors for invalid operations.

- **Tool Creation:**  
  - `StructuredTool.from_function()` is used to wrap the `calculator` function into a tool.  
  - The tool is provided with a name, description, and the custom input schema (`CalculatorInput`).

- **Usage:**  
  - The tool is executed by calling its `run()` method with a dictionary that matches the `CalculatorInput` schema.  
  - The result is the output of the arithmetic operation, in this case, addition.

---

### Example 2: Weather Forecast Tool

A tool that fetches the current weather forecast for a given city.

```python
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field
import requests  # Library to make HTTP requests

# Define the input schema for fetching weather data
class WeatherInput(BaseModel):
    city: str = Field(description="The city for which to fetch the weather forecast")  # City name input

# Function to fetch weather data from an external API
def get_weather(city: str) -> str:
    """Fetch the current weather forecast for a given city."""
    api_key = "your_api_key_here"  # Replace with a valid API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)  # Make an API request
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        return f"Weather in {city}: {data['weather'][0]['description']}"
    else:
        return "Failed to fetch weather data"  # Handle API request failure

# Create a StructuredTool instance from the get_weather function
weather_tool = StructuredTool.from_function(
    func=get_weather,  # Function to fetch weather data
    name="get_weather",  # Tool name
    description="Fetches the current weather forecast for a given city",  # Tool description
    args_schema=WeatherInput,  # Pydantic schema for input validation
)

# Example usage: Running the tool with a specific city
result = weather_tool.run({"city": "London"})
print(result)  # Expected output: Weather in London: clear sky (depending on API response)
```

**Output:**
```
Weather in London: clear sky
```

**Explanation:**  

- **Input Schema Definition:**  
  - A `WeatherInput` Pydantic model is defined with one field:  
    - `city`: a string representing the city for which the weather forecast is needed.

- **Function Implementation:**  
  - The `get_weather` function constructs an API URL using the provided city and an API key.  
  - It makes an HTTP GET request to fetch weather data and returns a description of the current weather.  
  - The function handles cases where the API call fails.

- **Tool Creation:**  
  - The `StructuredTool.from_function()` method wraps the `get_weather` function, associating it with a name, description, and the `WeatherInput` schema.

- **Usage:**  
  - The tool is executed by calling its `run()` method with a dictionary containing the city.  
  - The output is a formatted string describing the weather in the specified city.

---

### Example 3: Text Summarization Tool

A tool that summarizes a given text using a simple algorithm.

```python
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

# Define the input schema for text summarization
class SummarizeInput(BaseModel):
    text: str = Field(description="The text to summarize")  # Text input for summarization
    max_length: int = Field(description="The maximum length of the summary")  # Maximum allowed word count

# Function to summarize text by truncating it if it exceeds max_length words
def summarize_text(text: str, max_length: int) -> str:
    """Summarize the given text to a specified maximum length."""
    words = text.split()  # Split text into words
    if len(words) <= max_length:
        return text  # Return original text if within limit
    else:
        return " ".join(words[:max_length]) + "..."  # Truncate text and append ellipsis

# Create a StructuredTool instance from the summarize_text function
summarize_tool = StructuredTool.from_function(
    func=summarize_text,  # Function to perform text summarization
    name="summarize_text",  # Tool name
    description="Summarizes the given text to a specified maximum length",  # Tool description
    args_schema=SummarizeInput,  # Pydantic schema for input validation
)

# Example usage: Running the tool with sample text and max_length
result = summarize_tool.run({"text": "This is a long text that needs to be summarized.", "max_length": 15})
print(result)  # Expected output: A summarized version of the input text
```

**Output:**
```
This is a long text that needs to be summarized.
```

**Explanation:**  

- **Input Schema Definition:**  
  - A `SummarizeInput` Pydantic model is created with two fields:  
    - `text`: the text that needs summarizing.  
    - `max_length`: the maximum number of words allowed in the summary.

- **Function Implementation:**  
  - The `summarize_text` function splits the text into words and checks if it exceeds the specified maximum length.  
  - If the text is too long, it truncates the text and appends an ellipsis; otherwise, it returns the original text.

- **Tool Creation:**  
  - The function is wrapped into a tool using `StructuredTool.from_function()`, with the tool receiving a name, description, and the `SummarizeInput` schema.

- **Usage:**  
  - The tool is used by calling its `run()` method with a dictionary containing the text and the desired maximum length.  
  - The output is a summarized version of the input text.

---

### Example 4: Currency Conversion Tool

A tool that converts an amount from one currency to another using an external API.

```python
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field
import requests  # Library for HTTP requests

# Define the input schema for currency conversion
class CurrencyConversionInput(BaseModel):
    amount: float = Field(description="The amount to convert")  # Amount to be converted
    from_currency: str = Field(description="The currency to convert from")  # Source currency code
    to_currency: str = Field(description="The currency to convert to")  # Target currency code

# Function to convert currency using an external API
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert an amount from one currency to another."""
    api_key = "your_api_key_here"  # Replace with your actual API key if needed
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(url)  # Send API request to get exchange rates
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response
        rate = data['rates'][to_currency]  # Extract the conversion rate
        return amount * rate  # Calculate the converted amount
    else:
        raise ValueError("Failed to fetch currency data")  # Error handling for failed API request

# Create a StructuredTool instance from the convert_currency function
currency_tool = StructuredTool.from_function(
    func=convert_currency,  # Function to perform currency conversion
    name="convert_currency",  # Tool name
    description="Converts an amount from one currency to another",  # Tool description
    args_schema=CurrencyConversionInput,  # Pydantic schema for input validation
)

# Example usage: Running the tool with specified currency conversion parameters
result = currency_tool.run({"amount": 100, "from_currency": "USD", "to_currency": "EUR"})
print(result)  # Expected output: Converted amount (e.g., 85.0 based on current exchange rate)
```

**Output:**
```
95.39999999999999
```

**Explanation:**  

- **Input Schema Definition:**  
  - A `CurrencyConversionInput` Pydantic model is defined with three fields:  
    - `amount`: the numeric value to be converted.  
    - `from_currency`: the source currency code.  
    - `to_currency`: the target currency code.

- **Function Implementation:**  
  - The `convert_currency` function constructs an API URL using the `from_currency` value and sends a GET request to fetch the conversion rate.  
  - It calculates the converted amount by multiplying the input `amount` by the conversion rate.  
  - The function raises an error if the API call fails.

- **Tool Creation:**  
  - `StructuredTool.from_function()` is used to wrap the `convert_currency` function into a tool, along with a name, description, and the `CurrencyConversionInput` schema.

- **Usage:**  
  - The tool is executed by calling its `run()` method with a dictionary containing the amount and the currency codes.  
  - The output is the converted amount based on the fetched exchange rate.

---

### Example 5: Process Numbers Tool

This example demonstrates how to create a `StructuredTool` that computes the sum and product of two numbers. It uses both synchronous and asynchronous functions, while showcasing many of the parameters available in `StructuredTool.from_function()`. The example also includes a workaround for the `asyncio.run()` issue encountered in environments like Google Colab, which already have a running event loop.

```python
from pydantic import BaseModel
import asyncio
import nest_asyncio  # Needed for nested event loops in environments like Colab
from langchain_core.tools.structured import StructuredTool

# Apply nest_asyncio to allow nested event loops (for Colab compatibility)
nest_asyncio.apply()

# Define the synchronous function with docstring for automatic schema inference
def process_numbers(a: int, b: int) -> dict:
    """
    Process two numbers by computing their sum and product.

    Parameters:
        a (int): The first number.
        b (int): The second number.

    Returns:
        dict: A dictionary with keys 'sum' and 'product'.
    """
    return {"sum": a + b, "product": a * b}

# Define the asynchronous version of the function
async def process_numbers_async(a: int, b: int) -> dict:
    """
    Asynchronously process two numbers by computing their sum and product.

    Parameters:
        a (int): The first number.
        b (int): The second number.

    Returns:
        dict: A dictionary with keys 'sum' and 'product'.
    """
    await asyncio.sleep(0.1)  # Simulate asynchronous processing delay
    return {"sum": a + b, "product": a * b}

# Create a StructuredTool instance with multiple parameters using from_function
tool = StructuredTool.from_function(
    func=process_numbers,              # Synchronous implementation
    coroutine=process_numbers_async,   # Asynchronous implementation
    name="ProcessNumbersTool",         # Tool name
    description="A tool that computes the sum and product of two numbers.",  # Tool description
    return_direct=True,                # Directly return the result without extra wrapping
    args_schema=None,                  # Let infer_schema automatically create the schema from function signature and docstring
    infer_schema=True,                 # Enable schema inference
    response_format="content_and_artifact",  # Specify expected response format
    parse_docstring=True,              # Enable docstring parsing for schema and description
    error_on_invalid_docstring=False,  # Do not error out on invalid docstring formatting
    version="1.0",                     # Additional metadata: version
    author="Your Name"                 # Additional metadata: author
)

# Synchronous invocation of the tool using _run
result_sync = tool._run(3, 5, config=None)
print("Synchronous result:", result_sync)  # Expected output: {'sum': 8, 'product': 15}

# Asynchronous invocation: Define an async function to run the tool
async def invoke_tool():
    result_async = await tool._arun(7, 9, config=None)
    print("Asynchronous result:", result_async)  # Expected output: {'sum': 16, 'product': 63}

# Run the async invocation with asyncio.run (compatible with Colab due to nest_asyncio)
asyncio.run(invoke_tool())
```

**Output:**
```
Synchronous result: {'sum': 8, 'product': 15}
Asynchronous result: {'sum': 16, 'product': 63}
```

**Explanation:**  

In this example, we first apply `nest_asyncio.apply()` to allow the use of `asyncio.run()` in environments like Google Colab, where an event loop is already running. We define both a synchronous function (`process_numbers`) and an asynchronous function (`process_numbers_async`) that compute the sum and product of two numbers. The `StructuredTool.from_function()` method is used with several parameters:
- **`func` and `coroutine`:** Provide both synchronous and asynchronous implementations.
- **`name` and `description`:** Set the tool's name and description.
- **`return_direct`:** Indicates that the result should be returned directly.
- **`args_schema` and `infer_schema`:** Allow automatic schema inference from the function's signature and docstring.
- **`response_format`:** Specifies the output format.
- **`parse_docstring` and `error_on_invalid_docstring`:** Enable and configure docstring parsing for schema creation.
- **Extra kwargs (`version`, `author`):** Demonstrate additional metadata that can be passed to the tool.

The tool is invoked synchronously using `_run` and asynchronously using `_arun`, with the latter being managed using `asyncio.run()` (after applying `nest_asyncio`), ensuring compatibility with notebook environments.

---

## Best Practices

- **Define Clear Schemas:**  
  Always ensure that the input schema (`args_schema`) is well-defined. This not only validates inputs but also improves the clarity of your tool's API.

  ```python
  from pydantic import BaseModel, Field

  class MyToolInput(BaseModel):
      name: str = Field(..., description="Name of the entity")
      age: int = Field(..., description="Age of the entity")
  ```

- **Leverage Docstrings:**  
  Provide meaningful docstrings for your functions. When using `from_function`, these docstrings can be automatically parsed to generate the tool’s description, improving documentation and clarity.

  ```python
  def greet(name: str) -> str:
      """
      Generate a greeting message.

      Parameters:
          name (str): The name of the person.

      Returns:
          str: A personalized greeting message.
      """
      return f"Hello, {name}!"
  ```

- **Choose Execution Mode Wisely:**  
  Decide whether your tool should run synchronously or asynchronously based on the context. Use `func` for synchronous tasks and `coroutine` for tasks that benefit from asynchronous execution.

  ```python
  # Synchronous function
  def sync_task(data: int) -> int:
      return data * 2

  # Asynchronous function
  async def async_task(data: int) -> int:
      import asyncio
      await asyncio.sleep(0.1)
      return data * 2
  ```

- **Error Handling:**  
  Implement proper error handling within your functions to ensure that any unexpected issues are managed gracefully. This helps maintain stability in your tool and provides meaningful error messages.

  ```python
  def safe_divide(a: float, b: float) -> float:
      """
      Divide two numbers, with error handling for division by zero.

      Parameters:
          a (float): The numerator.
          b (float): The denominator.

      Returns:
          float: The result of division.
      """
      if b == 0:
          raise ValueError("Cannot divide by zero")
      return a / b
  ```

---

## Conclusion

The `StructuredTool` class is a powerful and flexible way to create tools within the LangChain ecosystem. By leveraging Pydantic for schema validation and offering both synchronous and asynchronous execution methods, it allows developers to build robust and scalable tools effortlessly. Whether you are wrapping a simple utility function or developing a complex processing component, `StructuredTool` provides the necessary structure and functionality to get the job done.

By following the best practices outlined in this article and utilizing the examples provided, you can effectively integrate and utilize `StructuredTool` in your LangChain projects. Happy coding!

---