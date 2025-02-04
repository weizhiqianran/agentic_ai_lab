# A Comprehensive Guide to Python Type Hints: From Basic Collections to Advanced Utilities

Python’s type hinting system has evolved into a robust framework that allows you to write self-documenting, maintainable, and error-resistant code. This guide covers both fundamental and advanced type hints—from built-in collection types to sophisticated utilities that enable generic programming, metadata annotations, and deprecation warnings.

Whether you're new to type annotations or looking to enhance your codebase with advanced features, this article provides practical examples and comparisons to help you leverage Python’s typing system effectively.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Type Hints](#basic-type-hints)
   - [List](#list)
   - [Dict](#dict)
   - [Set](#set)
   - [Tuple](#tuple)
   - [TypedDict](#typeddict)
   - [Optional](#optional)
   - [Union](#union)
   - [Literal](#literal)
3. [Advanced Type Hints and Utilities](#advanced-type-hints-and-utilities)
   - [Any](#any)
   - [Annotated](#annotated)
   - [Callable](#callable)
   - [ClassVar](#classvar)
   - [TypeVar](#typevar)
   - [deprecated](#deprecated)
4. [Comparison Table](#comparison-table)
5. [Conclusion](#conclusion)

---

## Introduction

Python’s dynamic nature is one of its greatest strengths, but it can also lead to runtime errors that static type checking might catch early. The `typing` module (and its extensions) provides a wealth of tools for annotating variables, function parameters, and return types. In this guide, we cover:

- **Basic Type Hints:** Essential for working with collections and enforcing structure on data.
- **Advanced Type Hints and Utilities:** Enhance your code with flexible type definitions, metadata, and generics.

Let’s dive into the specifics of each type and utility.

---

## Basic Type Hints

### List

**Example 1:**

A `List` is an ordered, mutable sequence of items. The type hint `List[T]` specifies that the list contains items of type `T`.

```python
from typing import List

def greet_names(names: List[str]) -> None:
    for name in names:
        print(f"Hello, {name}!")

# Usage:
greet_names(["Alice", "Bob", "Charlie"])
```

**Example 2:**

`List[Any]` allows a list to contain elements of any type, disabling type checking for the list's contents.


```python
from typing import List, Any

def process_items(items: List[Any]) -> None:
    for item in items:
        print(f"Processing: {item} (type: {type(item).__name__})")

# Usage:
mixed_list: List[Any] = ["Alice", 42, 3.14, {"key": "value"}, [1, 2, 3]]
process_items(mixed_list)
```

**Example 3:**

`List[Dict[str, int]]` specifies a list where each element is a dictionary with string keys and integer values.

```python
from typing import List, Dict

def summarize_data(records: List[Dict[str, int]]) -> None:
    for record in records:
        total = sum(record.values())
        print(f"Record: {record} -> Total: {total}")

# Usage:
data: List[Dict[str, int]] = [
    {"apples": 5, "bananas": 3},
    {"oranges": 7, "grapes": 2}
]
summarize_data(data)
```

---

### Dict

A `Dict` represents a key-value mapping. The type hint `Dict[K, V]` indicates that keys are of type `K` and values are of type `V`.

#### **Example 1: Storing User Information**

This example demonstrates how to use `Dict[str, int]` to store and display user-related information, where keys are strings (e.g., `"age"`, `"score"`) and values are integers.

```python
from typing import Dict

def print_user_info(user: Dict[str, int]) -> None:
    for key, value in user.items():
        print(f"{key}: {value}")

# Usage:
print_user_info({"age": 30, "score": 95})
```

#### **Example 2: Counting Word Frequency**

This example illustrates how a dictionary can be used to count occurrences of words in a given string. The function returns a `Dict[str, int]`, mapping each word to its frequency.

```python
from typing import Dict

def count_words(text: str) -> Dict[str, int]:
    word_count = {}
    for word in text.split():
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

# Usage:
result = count_words("hello world hello")
print(result)  # Output: {'hello': 2, 'world': 1}
```

#### **Example 3: Updating Inventory**

This function manages an inventory system where items are represented as dictionary keys (`str`), and their respective quantities as integer values (`int`). The function updates the quantity of an item, adding it if it does not exist.

```python
from typing import Dict

def update_inventory(inventory: Dict[str, int], item: str, quantity: int) -> Dict[str, int]:
    inventory[item] = inventory.get(item, 0) + quantity
    return inventory

# Usage:
inventory = {"apple": 10, "banana": 5}
updated_inventory = update_inventory(inventory, "apple", 5)
print(updated_inventory)  # Output: {'apple': 15, 'banana': 5}
```

#### **Example 4: Storing User Profiles with Mixed Data Types**

This example uses `Dict[str, Any]` to represent a user profile with diverse data types, including strings, integers, booleans, lists, and nested dictionaries.

```python
from typing import Dict, Any

def display_user_profile(profile: Dict[str, Any]) -> None:
    for key, value in profile.items():
        print(f"{key}: {value}")

# Usage:
user_profile = {
    "name": "Alice",
    "age": 30,
    "is_active": True,
    "preferences": ["reading", "traveling"],
    "metadata": {"last_login": "2023-10-01", "login_count": 42}
}

display_user_profile(user_profile)

# Output:
# name: Alice
# age: 30
# is_active: True
# preferences: ['reading', 'traveling']
# metadata: {'last_login': '2023-10-01', 'login_count': 42}
```

---

### Set

A `Set` is an unordered collection of unique items. The type hint `Set[T]` specifies that the set contains items of type `T`. It is useful for eliminating duplicate values and performing set operations such as unions, intersections, and differences.

#### **Example 1: Removing Duplicates from a Collection**

This example demonstrates how to use a `Set[int]` to filter out duplicate numbers from a collection.

```python
from typing import Set

def unique_numbers(numbers: Set[int]) -> None:
    for number in numbers:
        print(number)

# Usage:
unique_numbers({1, 2, 3, 2, 1})  # Output: 1, 2, 3 (order may vary)
```

#### **Example 2: Performing Set Operations (Union, Intersection, Difference)**

This example demonstrates various set operations using `Set[str]`. The function takes two sets of words and returns their union, intersection, and difference.

```python
from typing import Set, Tuple

def set_operations(set_a: Set[str], set_b: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    union_set = set_a | set_b         # Union: elements in either set
    intersection_set = set_a & set_b  # Intersection: elements in both sets
    difference_set = set_a - set_b    # Difference: elements in set_a but not in set_b
    return union_set, intersection_set, difference_set

# Usage:
words_set_1 = {"apple", "banana", "cherry"}
words_set_2 = {"banana", "cherry", "date", "fig"}

union, intersection, difference = set_operations(words_set_1, words_set_2)

print("Union:", union)                # Output: {'apple', 'banana', 'cherry', 'date', 'fig'}
print("Intersection:", intersection)  # Output: {'banana', 'cherry'}
print("Difference:", difference)      # Output: {'apple'}
```

#### **Example 3: Tracking Unique Users with a Set of Tuples**

This example uses `Set[Tuple[int, str]]` to store and track unique user IDs and names.

```python
from typing import Set, Tuple

def add_user(users: Set[Tuple[int, str]], user_id: int, user_name: str) -> Set[Tuple[int, str]]:
    users.add((user_id, user_name))
    return users

# Usage:
registered_users: Set[Tuple[int, str]] = {(1, "Alice"), (2, "Bob")}
updated_users = add_user(registered_users, 3, "Charlie")
updated_users = add_user(updated_users, 1, "Alice")  # Duplicate entry ignored

print(updated_users)  # Output: {(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')}
```

---

### Tuple

A `Tuple` is an immutable sequence. The type hint `Tuple[T1, T2, ...]` indicates the types of elements in the tuple, often of fixed length.

**Example 1:**

This example demonstrates how to calculate the Euclidean distance from the origin for a point represented by a `Tuple[float, float]`.

```python
from typing import Tuple

def process_point(point: Tuple[float, float]) -> float:
    x, y = point
    return (x ** 2 + y ** 2) ** 0.5  # Euclidean distance from origin

# Usage:
distance = process_point((3.0, 4.0))
print(f"Distance: {distance}")
```

**Example 2: Representing RGB Colors**

This example uses `Tuple[int, int, int]` to represent RGB color values, where each integer ranges from 0 to 255.

```python
from typing import Tuple

def format_color(rgb: Tuple[int, int, int]) -> str:
    return f"RGB Color: ({rgb[0]}, {rgb[1]}, {rgb[2]})"

# Usage:
color = (255, 165, 0)  # Orange
formatted_color = format_color(color)
print(formatted_color)  # Output: RGB Color: (255, 165, 0)
```

**Example 3: Returning Multiple Values from a Function**

This example demonstrates returning multiple values as a tuple. It calculates the perimeter and area of a rectangle given its width and height.

```python
from typing import Tuple

def calculate_rectangle_properties(width: float, height: float) -> Tuple[float, float]:
    perimeter = 2 * (width + height)
    area = width * height
    return perimeter, area

# Usage:
rect_properties = calculate_rectangle_properties(5.0, 10.0)
print(f"Perimeter: {rect_properties[0]}, Area: {rect_properties[1]}")
```

---

### TypedDict

`TypedDict` allows you to specify the types of values associated with specific keys in a dictionary, similar to defining a simple record or data class.

**Example 1:**

This example defines an `Employee` `TypedDict`, which ensures that a dictionary representing an employee has the required keys (`name`, `id`, and `department`) with specific data types.

```python
from typing import TypedDict

class Employee(TypedDict):
    name: str
    id: int
    department: str

def display_employee(employee: Employee) -> None:
    print(f"Employee {employee['id']}: {employee['name']} works in {employee['department']}")

# Usage:
emp: Employee = {"name": "Jane Doe", "id": 101, "department": "HR"}
display_employee(emp)
```

**Example 2: Using Optional Keys in a TypedDict**

This example demonstrates a `TypedDict` with optional fields using `total=False`, meaning that some keys may be omitted from the dictionary.

```python
from typing import TypedDict, Optional

class Product(TypedDict, total=False):
    name: str
    price: float
    stock: Optional[int]  # The stock field is optional

def display_product(product: Product) -> None:
    name = product.get("name", "Unknown Product")
    price = product.get("price", 0.0)
    stock = product.get("stock", "N/A")
    print(f"Product: {name}, Price: ${price}, Stock: {stock}")

# Usage:
product_1: Product = {"name": "Laptop", "price": 1299.99, "stock": 10}
product_2: Product = {"name": "Mouse", "price": 29.99}  # Stock is omitted
display_product(product_1)
display_product(product_2)
```

**Example 3: Nested TypedDict for Complex Data Structures**

This example demonstrates how to use `TypedDict` inside another `TypedDict` to represent more complex structures.

```python
from typing import TypedDict

class Address(TypedDict):
    street: str
    city: str
    zip_code: str

class UserProfile(TypedDict):
    username: str
    age: int
    address: Address  # Nested TypedDict

def display_profile(profile: UserProfile) -> None:
    print(f"User: {profile['username']}, Age: {profile['age']}")
    print(f"Address: {profile['address']['street']}, {profile['address']['city']}, {profile['address']['zip_code']}")

# Usage:
user: UserProfile = {
    "username": "john_doe",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "zip_code": "10001"
    }
}
display_profile(user)
```

---

### Optional

`Optional[T]` is shorthand for `Union[T, None]`. It indicates that a variable can be of type `T` or `None`.

**Example 1:**

This example demonstrates using `Optional[int]` to indicate that the function may return either an integer or `None`. The function searches for a target string in a list and returns its index if found.

```python
from typing import Optional, List

def find_item(items: List[str], target: str) -> Optional[int]:
    try:
        return items.index(target)
    except ValueError:
        return None

# Usage:
result = find_item(["apple", "banana", "cherry"], "banana")
if result is not None:
    print(f"Found at index {result}")
else:
    print("Not found.")
```

**Example 2: Handling Optional Dictionary Keys**

This example demonstrates using `Optional[str]` for a dictionary key, allowing the function to handle missing values gracefully.

```python
from typing import Dict, Optional

def get_user_email(user_data: Dict[str, Optional[str]]) -> str:
    email = user_data.get("email")
    if email is None:
        return "No email provided"
    return f"User email: {email}"

# Usage:
user_with_email = {"name": "Alice", "email": "alice@example.com"}
user_without_email = {"name": "Bob"}

print(get_user_email(user_with_email))     # Output: User email: alice@example.com
print(get_user_email(user_without_email))  # Output: No email provided
```

**Example 3: Using Optional in Function Parameters**

This example shows how `Optional[int]` can be used as a default function parameter, allowing the function to behave differently based on whether the parameter is provided.

```python
from typing import Optional

def greet(name: str, age: Optional[int] = None) -> str:
    if age is None:
        return f"Hello, {name}!"
    return f"Hello, {name}! You are {age} years old."

# Usage:
print(greet("Alice"))    # Output: Hello, Alice!
print(greet("Bob", 30))  # Output: Hello, Bob! You are 30 years old.
```

---

### Union

`Union` represents a type that could be any one of several specified types. For example, `Union[int, str]` means the value can be either an integer or a string.

**Example 1:**

This example shows how `Union[int, str]` allows a function to process either an integer or a string and return a formatted string based on the input type.

```python
from typing import Union

def process_value(value: Union[int, str]) -> str:
    if isinstance(value, int):
        return f"Integer: {value}"
    elif isinstance(value, str):
        return f"String: {value}"
    else:
        return "Unsupported type"

# Usage:
print(process_value(42))
print(process_value("hello"))
```

**Example 2: Returning Different Data Types Based on Input**

This example demonstrates a function that uses `Union[str, list[str]]` as a return type. If a single word is passed, it returns the word in uppercase. If a list of words is passed, it returns a list of words in uppercase.

```python
from typing import Union, List

def transform_text(text: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(text, str):
        return text.upper()
    elif isinstance(text, list):
        return [word.upper() for word in text]

# Usage:
print(transform_text("hello"))  # Output: HELLO
print(transform_text(["hello", "world"]))  # Output: ['HELLO', 'WORLD']
```

**Example 3: Handling Multiple Input Types in a Dictionary**

This example shows how a dictionary can store different types of values using `Union[str, int, float]` to allow a mix of strings, integers, and floating-point numbers.

```python
from typing import Dict, Union

UserProfile = Dict[str, Union[str, int, float]]

def display_profile(profile: UserProfile) -> None:
    for key, value in profile.items():
        print(f"{key}: {value}")

# Usage:
user = {
    "name": "Alice",
    "age": 30,
    "height": 5.6,
    "city": "New York"
}

display_profile(user)

# Output:
# name: Alice
# age: 30
# height: 5.6
# city: New York
```

---

### Literal

`Literal` allows you to constrain a value to a specific literal value (or set of literal values). This is especially useful for function parameters that are expected to have a fixed set of possible values.

**Example 1:**

This example demonstrates how `Literal` can be used to restrict a function parameter to specific string values representing log levels.

```python
from typing import Literal

def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
    print(f"Log level set to {level}")

# Usage:
set_log_level("INFO")
# set_log_level("VERBOSE")  # This would be flagged by a type checker
```

**Example 2: Using Different Data Types in Literal**

This example demonstrates how `Literal` can contain different types, including integers, strings, and floating-point numbers.

```python
from typing import Literal

def get_discount(category: Literal["student", "veteran", 10, 15, 20]) -> str:
    return f"Applying discount: {category}"

# Usage:
print(get_discount("student"))  # Output: Applying discount: student
print(get_discount(15))         # Output: Applying discount: 15
# print(get_discount(25))       # Type checker will flag this as invalid
```

**Example 3: Using Mixed Types in Function Return Values**

This example shows how `Literal` can be used as a return type with mixed types, ensuring that a function only returns specific values.

```python
from typing import Literal

def check_status(code: int) -> Literal["success", 200, None]:
    if code == 1:
        return "success"
    elif code == 2:
        return 200
    else:
        return None

# Usage:
print(check_status(1))  # Output: success
print(check_status(2))  # Output: 200
print(check_status(3))  # Output: None
```

---

## Advanced Type Hints and Utilities

---

### Any

The `Any` type indicates that a variable can hold any type of value, effectively disabling type checking for that variable.

**Example 1:**

This example demonstrates how `Any` can be used as a function parameter to accept any type of data without enforcing type checking.

```python
from typing import Any

def process_data(data: Any) -> None:
    # Type checkers will ignore type issues in this function.
    print("Processing:", data)

# Usage:
process_data("A string")
process_data(42)
process_data([1, 2, 3])
```

**Example 2: Using Any in a Dictionary to Store Mixed Data Types**

This example demonstrates how `Any` can be used to define a dictionary that holds different types of values.

```python
from typing import Dict, Any

def display_user_profile(profile: Dict[str, Any]) -> None:
    for key, value in profile.items():
        print(f"{key}: {value}")

# Usage:
user_profile = {
    "name": "Alice",
    "age": 30,
    "is_active": True,
    "preferences": ["reading", "traveling"],
    "metadata": {"last_login": "2023-10-01", "login_count": 42}
}

display_user_profile(user_profile)

# Output:
# name: Alice
# age: 30
# is_active: True
# preferences: ['reading', 'traveling']
# metadata: {'last_login': '2023-10-01', 'login_count': 42}
```

**Example 3: Using Any in a List to Hold Various Data Types**

This example demonstrates how `Any` can be used in a list to store elements of different types.

```python
from typing import List, Any

def print_items(items: List[Any]) -> None:
    for item in items:
        print(f"Item: {item} (Type: {type(item).__name__})")

# Usage:
mixed_list: List[Any] = ["Alice", 42, 3.14, {"key": "value"}, [1, 2, 3]]
print_items(mixed_list)

# Output:
# Item: Alice (Type: str)
# Item: 42 (Type: int)
# Item: 3.14 (Type: float)
# Item: {'key': 'value'} (Type: dict)
# Item: [1, 2, 3] (Type: list)
```

---

### Annotated

`Annotated` (available in `typing_extensions` and now in the standard library from Python 3.9 with [PEP 593](https://www.python.org/dev/peps/pep-0593/)) allows you to add metadata to type hints. This metadata can be used by external tools or frameworks to influence behavior (e.g., for validation or serialization).

**Example 1:**

This example demonstrates how `Annotated` can be used to add metadata to indicate that an integer should be a positive number.

```python
from typing_extensions import Annotated

# Suppose we want to annotate a parameter with some validation metadata.
def set_age(age: Annotated[int, "must be a positive integer"]) -> None:
    if age <= 0:
        raise ValueError("Age must be positive")
    print(f"Age set to {age}")

# Usage:
set_age(25)  # Valid
# set_age(-5)  # Would raise a ValueError
```

**Example 2: Using `Annotated` in Class Member Variables**

This example demonstrates how `Annotated` can be used to define metadata for class attributes. This metadata does not enforce constraints at runtime but serves as documentation and can be used by static analysis tools.

```python
from typing_extensions import Annotated

class Product:
    name: Annotated[str, "Product name, must be non-empty"]
    price: Annotated[float, "Price in USD, must be positive"]
    stock: Annotated[int, "Number of items in stock, must be >= 0"]

    def __init__(self, name: str, price: float, stock: int):
        if not name:
            raise ValueError("Product name cannot be empty")
        if price <= 0:
            raise ValueError("Price must be positive")
        if stock < 0:
            raise ValueError("Stock cannot be negative")

        self.name = name
        self.price = price
        self.stock = stock

# Usage:
p = Product("Laptop", 999.99, 10)
print(p.name, p.price, p.stock)
```

**Example 3: Using `Annotated` in Return Type for Additional Metadata**

This example demonstrates how `Annotated` can be used in function return types to describe additional constraints or context about the returned value. The `Annotated` type hints that the return value represents a temperature in Celsius.

```python
from typing_extensions import Annotated

def get_temperature() -> Annotated[float, "Temperature in Celsius"]:
    return 36.6

# Usage:
temp = get_temperature()
print(f"Body temperature: {temp}°C")  # Output: Body temperature: 36.6°C
```

**Example 4: Using Multiple Metadata Annotations in Return Type**

This example demonstrates how `Annotated` can support multiple metadata annotations. It combines metadata to specify **a numeric range, a unit of measurement, and a validation rule**.

```python
from typing_extensions import Annotated

def set_temperature(temp: Annotated[float, "Temperature in Celsius", "Must be between -50 and 100"]) -> None:
    if not (-50 <= temp <= 100):
        raise ValueError("Temperature must be between -50 and 100°C")
    print(f"Temperature set to {temp}°C")

# Usage:
set_temperature(36.5)   # Valid
# set_temperature(150)  # Raises ValueError
```

---

### Callable

`Callable` is used to indicate that a variable or parameter is a function or any object that implements the `__call__` method. You can also specify the function’s parameter types and return type.

**Example 1: Passing a Function as an Argument**

This example demonstrates how `Callable` can be used to define a function that takes another function as a parameter and applies it to two integers.

```python
from typing import Callable

def apply_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Define a couple of simple functions
def add(x: int, y: int) -> int:
    return x + y

def multiply(x: int, y: int) -> int:
    return x * y

# Usage:
print(apply_function(add, 2, 3))       # Outputs: 5
print(apply_function(multiply, 2, 3))  # Outputs: 6
```

**Example 2: Returning a Function from Another Function**

This example demonstrates how `Callable` can be used as a return type, specifying that a function returns another function. This approach allows returning different functions dynamically based on user input.

```python
from typing import Callable

def get_operator(operation: str) -> Callable[[int, int], int]:
    if operation == "add":
        return lambda x, y: x + y
    elif operation == "multiply":
        return lambda x, y: x * y
    else:
        raise ValueError("Unsupported operation")

# Usage:
add_func = get_operator("add")
multiply_func = get_operator("multiply")

print(add_func(5, 10))        # Output: 15
print(multiply_func(5, 10))   # Output: 50
```

**Example 3: Creating a Callable Class with `__call__` Method**

This example demonstrates how `Callable` can be used to type hint objects of a class that defines a `__call__` method. Here, `Repeater` is a class that behaves like a function, thanks to the `__call__` method. The function `execute_callable` can accept any `Callable[[int], str]`, including instances of `Repeater`. 

```python
from typing import Callable

class Repeater:
    def __init__(self, word: str):
        self.word = word

    def __call__(self, times: int) -> str:
        return " ".join([self.word] * times)

def execute_callable(func: Callable[[int], str], n: int) -> str:
    return func(n)

# Usage:
repeater = Repeater("Hello")
print(execute_callable(repeater, 3))  # Output: Hello Hello Hello
```

**Example 4: Using `Callable` with Variable Arguments (`*args`, `**kwargs`)**

This example demonstrates how `Callable` can be used to accept functions with flexible keyword arguments. This makes `execute_task` a flexible function capable of executing any callable with variable arguments.

```python
from typing import Callable, Any

def execute_task(task: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return task(*args, **kwargs)

# Define a function with keyword arguments
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# Usage:
print(execute_task(greet, "Alice"))               # Output: Hello, Alice!
print(execute_task(greet, "Bob", greeting="Hi"))  # Output: Hi, Bob!
```

---

### ClassVar  
`ClassVar` is used to indicate that a variable is intended to be a class variable, not an instance variable. This is especially useful with data classes and when defining constants or shared properties.

**Example 1: Defining a Class-Level Constant in a Data Class**  
This example demonstrates how `ClassVar` can be used in a data class to define a class-level constant that applies to all instances of the class.

```python
from typing import ClassVar
from dataclasses import dataclass

@dataclass
class Configuration:
    # This is a class-level constant, not an instance variable.
    version: ClassVar[str] = "1.0.0"
    name: str

# Usage:
print(Configuration.version)  # Outputs: 1.0.0

config = Configuration(name="MyApp")
print(config.name)  # Outputs: MyApp
```

**Example 2: Tracking Total Instances of a Class**  
This example demonstrates how `ClassVar` can be used to track the number of instances of a class. The `total_count` variable is a shared class attribute that updates whenever a new instance is created.

```python
from typing import ClassVar

class Counter:
    total_count: ClassVar[int] = 0  # Shared class variable

    def __init__(self) -> None:
        Counter.total_count += 1

# Usage:
c1 = Counter()
c2 = Counter()
c3 = Counter()
print(Counter.total_count)  # Outputs: 3
```

**Example 3: Defining Role-Based Access in a Class**  
This example demonstrates how `ClassVar` can be used to define a set of shared class-level role permissions, ensuring that these values remain constant across all instances.

```python
from typing import ClassVar

class User:
    # Class-level variable to define role-based permissions
    roles: ClassVar[dict[str, list[str]]] = {
        "admin": ["read", "write", "delete"],
        "editor": ["read", "write"],
        "viewer": ["read"]
    }

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role

    def get_permissions(self) -> list[str]:
        return User.roles.get(self.role, [])

# Usage:
admin = User("Alice", "admin")
viewer = User("Bob", "viewer")

print(admin.get_permissions())  # Outputs: ['read', 'write', 'delete']
print(viewer.get_permissions())  # Outputs: ['read']
```

---

### TypeVar  

`TypeVar` is used to define generic types, allowing you to create functions and classes that work with any data type while still maintaining type safety.

**Example 1: Using `TypeVar` in a Generic Function**  
This example demonstrates how `TypeVar` can be used to create a generic function that returns the first item from a list, preserving the type of the input list.

```python
from typing_extensions import TypeVar

# Define a type variable
T = TypeVar('T')

def first_item(items: list[T]) -> T:
    """Returns the first item from a list."""
    if not items:
        raise ValueError("List is empty")
    return items[0]

# Usage:
print(first_item([1, 2, 3]))        # Works with int
print(first_item(["a", "b", "c"]))  # Works with str
```

**Example 2: Creating a Generic Stack Class**  
This example demonstrates how `TypeVar` can be used to create a type-safe stack class that works with any data type.

```python
from typing_extensions import TypeVar

# Define a type variable
T = TypeVar('T')

class Stack:
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

# Usage:
int_stack = Stack()
int_stack.push(10)
int_stack.push(20)
print(int_stack.pop())  # Outputs: 20
```

**Example 3: Using `TypeVar` with Multiple Type Parameters**  
This example demonstrates how `TypeVar` can be used with multiple type parameters to create a generic function that swaps two values while preserving their respective types.

```python
from typing_extensions import TypeVar

# Define two type variables
T1 = TypeVar('T1')
T2 = TypeVar('T2')

def swap(a: T1, b: T2) -> tuple[T2, T1]:
    """Swaps two values and returns them in reversed order."""
    return b, a

# Usage:
x, y = swap(10, "hello")
print(x, y)  # Output: hello 10
```

**Example 4: Using `TypeVar` to Enforce Subtype Constraints**  
This example demonstrates how `TypeVar` can enforce constraints by ensuring that a generic function only works with subclasses of a given base class.

```python
from typing_extensions import TypeVar

class Animal:
    def speak(self) -> str:
        return "Some generic animal sound"

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

# Define a type variable restricted to subclasses of Animal
A = TypeVar('A', bound=Animal)

def make_animal_speak(animal: A) -> str:
    """Calls the speak method on an Animal or its subclass."""
    return animal.speak()

# Usage:
dog = Dog()
print(make_animal_speak(dog))  # Output: Woof!
```

---

### deprecated  

The `deprecated` decorator from `typing_extensions` is used to mark functions or classes as deprecated. This can help signal to developers that a piece of code is outdated and may be removed in future versions.

**Example 1: Deprecating a Function with a Replacement Suggestion**  
This example demonstrates how `deprecated` can be used to mark an old function as deprecated and suggest a new function as its replacement.

```python
from typing_extensions import deprecated

@deprecated("Use new_function instead")
def old_function(x: int) -> int:
    return x * 2

def new_function(x: int) -> int:
    return x * 2

# Usage:
print(old_function(10))  # Will print a deprecation warning if your tooling supports it.
print(new_function(10))
```

**Example 2: Deprecating a Class**  
This example demonstrates how `deprecated` can be applied to an entire class, warning developers to use a new version instead.

```python
from typing_extensions import deprecated

@deprecated("Use NewClass instead")
class OldClass:
    def greet(self) -> str:
        return "Hello from OldClass"

class NewClass:
    def greet(self) -> str:
        return "Hello from NewClass"

# Usage:
old_instance = OldClass()
print(old_instance.greet())  # May trigger a deprecation warning

new_instance = NewClass()
print(new_instance.greet())  # Recommended approach
```

**Example 3: Deprecating a Method in a Class**  
This example demonstrates how `deprecated` can be used to mark a specific method inside a class as deprecated while keeping the rest of the class intact.

```python
from typing_extensions import deprecated

class Example:
    @deprecated("Use new_method instead")
    def old_method(self) -> str:
        return "This is the old method"

    def new_method(self) -> str:
        return "This is the new method"

# Usage:
obj = Example()
print(obj.old_method())  # May trigger a deprecation warning
print(obj.new_method())  # Recommended approach
```

---

## Comparison Table

| **Type/Decorator** | **Definition**                                                                                     | **Syntax Example**                              | **Typical Use Case**                                               |
|--------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------|--------------------------------------------------------------------|
| `Any`              | Represents any type; disables type checking for the annotated variable.                             | `def f(x: Any) -> None: ...`                     | When type is unknown or should be flexible.                        |
| `List[T]`          | Ordered mutable sequence of elements of type `T`.                                                 | `List[int]`                                     | Storing a sequence of numbers.                                     |
| `Dict[K, V]`       | Key-value mapping with keys of type `K` and values of type `V`.                                    | `Dict[str, int]`                                | Representing JSON objects or configurations.                       |
| `Set[T]`           | Unordered collection of unique items of type `T`.                                                | `Set[str]`                                      | Removing duplicates or membership testing.                         |
| `Tuple[T1, T2, ...]`| Immutable sequence of fixed number of elements with specified types.                           | `Tuple[str, int]`                               | Returning multiple values from a function.                         |
| `TypedDict`        | Dictionary with a fixed set of keys and associated value types.                                  | `class Employee(TypedDict): ...`                | Enforcing structure on dictionaries.                               |
| `Optional[T]`      | A type that can either be `T` or `None` (i.e., `Union[T, None]`).                                  | `Optional[str]`                                 | Representing a value that might be absent.                           |
| `Union[A, B, ...]` | A type that can be any one of the specified types.                                               | `Union[int, str]`                               | Allowing multiple valid types for a variable.                      |
| `Literal[...]`     | Restricts a variable to one or more literal values.                                              | `Literal["DEBUG", "INFO"]`                      | Function parameters with a fixed set of valid options.             |
| `Annotated`        | Adds metadata to type hints for additional context or processing by external tools.              | `Annotated[int, "positive integer"]`             | Passing extra information for validation, documentation, or serialization. |
| `Callable`         | Annotates callable objects (functions, methods) with specified argument and return types.        | `Callable[[int, int], int]`                      | Typing function arguments, callbacks, or any callable signature.     |
| `ClassVar`         | Marks a variable as a class variable (not part of an instance).                                   | `version: ClassVar[str] = "1.0"`                 | Defining constants or shared state on a class.                       |
| `TypeVar`          | Defines a generic type variable for creating generic functions or classes.                       | `T = TypeVar('T')`                              | Creating functions or classes that work with multiple types safely.  |
| `deprecated`       | Marks a function or class as deprecated to warn developers of its planned removal or obsolescence. | `@deprecated("Use new_function instead")`       | Signaling that a function or class should no longer be used.         |

---

## Conclusion

Python’s type hinting system—from basic constructs like `List`, `Dict`, and `Tuple` to advanced utilities such as `Annotated`, `deprecated`, and `TypeVar`—provides a powerful framework for writing clean, maintainable, and robust code. By integrating these hints into your development process, you enable static analysis tools to catch bugs early and create self-documenting code that is easier to understand and maintain.

Whether you’re managing collections or building generic, reusable components, the Python typing system offers tools for every need. Embrace these type hints to make your code safer and more expressive. Happy coding!

