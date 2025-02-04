# An Introduction to Pydantic's Field

Pydantic is a powerful library for data validation and settings management in Python. Its heart lies in the `BaseModel` class, which enables you to define models as simple Python classes with type annotations. Pydantic then takes care of validation, type conversion, error handling, and more. In this article, we’ll dive into the capabilities of `BaseModel` with plenty of examples and explanations. We’ll also include helpful tables and a detailed table of contents.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What Is the Field Class?](#what-is-the-field-class)
3. [Field Signature and Parameters](#field-signature-and-parameters)
    - [Main Parameters Table](#main-parameters-table)
4. [Basic Examples](#basic-examples)
    - [Basic Default Value](#basic-default-value)
    - [Using `default_factory` with a Function](#using-default_factory-with-a-function)
    - [Using `default_factory` with a Lambda Function](#using-default_factory-with-a-lambda-function)
    - [Using `default_factory` for Mutable Defaults](#using-default_factory-for-mutable-defaults)
    - [Aliases and Title Customization](#aliases-and-title-customization)
    - [Validation Constraints](#validation-constraints)
    - [Excluding and Deprecated Fields](#excluding-and-deprecated-fields)
5. [Advanced Examples](#advanced-examples)
    - [Custom JSON Schema Extra](#custom-json-schema-extra)
    - [Computed Fields](#computed-fields)
6. [Constraint Examples](#constraint-examples)
    - [Numeric Constraints](#numeric-constraints)
    - [String Constraints](#string-constraints)
    - [Strict Mode](#strict-mode)
7. [Conclusion](#conclusion)

---

## Introduction

The **Field** class is a fundamental component in Pydantic that allows you to configure and annotate the behavior of model attributes. It provides a flexible way to specify default values, validation rules, metadata for JSON schema generation, aliasing, and many other options. This article covers the various aspects of the **Field** class, offering many code examples and detailed explanations to help you get started.

---

## What Is the Field Class?

The **Field** function is used to create a field configuration object (a **FieldInfo** instance) that holds metadata and settings for a model’s attribute. You can use it in conjunction with type annotations to set defaults, validators, and other behaviors on fields. In addition, **Field** supports a rich set of parameters that help control both the validation and serialization of your models.

In our examples, we import the necessary classes as follows:

```python
from pydantic import BaseModel, Field
```

---

## Field Signature and Parameters

Below is a summary of the Field signature with its key parameters:

```python
Field(
    default: Any = PydanticUndefined,
    *,
    default_factory: Callable[[], Any] | Callable[[dict[str, Any]], Any] | None = _Unset,
    alias: str | None = _Unset,
    alias_priority: int | None = _Unset,
    validation_alias: str | AliasPath | AliasChoices | None = _Unset,
    serialization_alias: str | None = _Unset,
    title: str | None = _Unset,
    field_title_generator: Callable[[str, FieldInfo], str] | None = _Unset,
    description: str | None = _Unset,
    examples: list[Any] | None = _Unset,
    exclude: bool | None = _Unset,
    discriminator: str | Discriminator | None = _Unset,
    deprecated: Deprecated | str | bool | None = _Unset,
    json_schema_extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
    frozen: bool | None = _Unset,
    validate_default: bool | None = _Unset,
    repr: bool = _Unset,
    init: bool | None = _Unset,
    init_var: bool | None = _Unset,
    kw_only: bool | None = _Unset,
    pattern: str | Pattern[str] | None = _Unset,
    strict: bool | None = _Unset,
    coerce_numbers_to_str: bool | None = _Unset,
    gt: SupportsGt | None = _Unset,
    ge: SupportsGe | None = _Unset,
    lt: SupportsLt | None = _Unset,
    le: SupportsLe | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    union_mode: Literal["smart", "left_to_right"] = _Unset,
    fail_fast: bool | None = _Unset,
    **extra: Unpack[_EmptyKwargs]
) -> Any
```

> **Note:** Any parameters left as `_Unset` will be replaced by their default values as specified in the internal defaults dictionary.

### Main Parameters Table

| Name                | Type                                                                                      | Description                                                                                                                                      | Default               |
|---------------------|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| `default`           | `Any`                                                                                     | Default value if the field is not set.                                                                                                           | `PydanticUndefined`   |
| `default_factory`   | `Callable[[], Any] \| Callable[[dict[str, Any]], Any] \| None`                               | Callable to generate the default value. It may take no arguments or a single argument with validated data.                                        | `_Unset`              |
| `alias`             | `str \| None`                                                                             | Alternate name used during validation/serialization (e.g., converting between snake_case and camelCase).                                         | `_Unset`              |
| `title`             | `str \| None`                                                                             | Human-readable title for the field.                                                                                                              | `_Unset`              |
| `description`       | `str \| None`                                                                             | A short description of the field.                                                                                                                | `_Unset`              |
| `examples`          | `list[Any] \| None`                                                                       | Example values for the field.                                                                                                                    | `_Unset`              |
| `exclude`           | `bool \| None`                                                                            | Whether to exclude the field from the model serialization.                                                                                     | `_Unset`              |
| `gt`, `ge`, `lt`, `le` | `SupportsGt \| None`, etc.                                                              | Numeric validation constraints (greater than, greater or equal, less than, less or equal).                                                         | `_Unset`              |
| `pattern`           | `str \| Pattern[str] \| None`                                                               | Regular expression pattern constraint for strings.                                                                                             | `_Unset`              |
| `json_schema_extra` | `JsonDict \| Callable[[JsonDict], None] \| None`                                            | Extra information to add to the JSON schema generated for this field.                                                                           | `_Unset`              |

There are many other parameters available to fine-tune the field's behavior. For a complete list, refer to the full Field definition provided above.

---

## Basic Examples

### Basic Default Value

One of the simplest uses of **Field** is to set a default value for a field in a model.

```python
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    foo: int = Field(default=4)

m = MyModel()
print(m.foo)  # Output: 4
```

In this example, the field `foo` has a default value of `4`.

### Using `default_factory` with a Function

Sometimes you need a dynamic default value generated by a function. You can achieve this by using the `default_factory` parameter.

```python
from pydantic import BaseModel, Field
import datetime

def current_time():
    return datetime.datetime.now()

class Event(BaseModel):
    timestamp: datetime.datetime = Field(default_factory=current_time)

event = Event()
print(event.timestamp)  # Outputs the current datetime
```

Here, every time an `Event` is created without an explicit `timestamp`, the `current_time` function is called to generate the default.

---

### Using `default_factory` with a Lambda Function

You can also use a lambda function with `default_factory` to generate dynamic default values. For example, you might want to assign a unique identifier to each instance using the `uuid` module.

```python
from pydantic import BaseModel, Field
import uuid

class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

user = User()
print(user.user_id)  # Outputs a unique UUID string
```

---

### Using `default_factory` for Mutable Defaults

Using mutable types such as lists or dictionaries directly as default values can lead to unexpected behavior. Instead, you can use `default_factory` to ensure that a new object is created for each instance.

```python
from pydantic import BaseModel, Field

class ShoppingCart(BaseModel):
    items: list[str] = Field(default_factory=list)

cart1 = ShoppingCart()
cart2 = ShoppingCart()

# Append an item to the first cart
cart1.items.append("Apple")

print(cart1.items)  # Output: ['Apple']
print(cart2.items)  # Output: [] (each instance gets its own list)
```

In this example, each `ShoppingCart` instance receives its own independent list for the `items` field.

---

### Aliases and Title Customization

The `alias` parameter is useful for mapping between internal attribute names and external data keys (e.g., when working with APIs). Additionally, you can add a human-readable `title` for the field to improve documentation and readability.

**Example:**

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(..., alias='userName', title='User Name')
    age: int = Field(..., description='The age of the user')

# Usage:
user_data = {'userName': 'john_doe', 'age': 30}
user = User(**user_data)

# Accessing the model:
print(user)
# Output: username='john_doe' age=30

# Accessing the field with its alias:
print(user.username)  # Output: john_doe
```

---

### Key Points:
- **`alias`**: Maps an external key (e.g., `userName`) to an internal attribute name (e.g., `username`).
- **`title`**: Provides a human-readable title for the field, useful for documentation.
- **`description`**: Adds a description to the field for clarity.

This approach is particularly helpful when working with external data sources (e.g., APIs) where the field names may differ from your internal naming conventions.

---

### Validation Constraints

You can use **Field** to impose validation constraints on your model’s numeric attributes. Pydantic provides several keyword arguments for constraining numeric values:

- **gt**: Greater than
- **lt**: Less than
- **ge**: Greater than or equal to
- **le**: Less than or equal to
- **multiple_of**: A multiple of the given number
- **allow_inf_nan**: Allow `inf`, `-inf`, or `nan` values

Below is an example that applies several of these constraints to a `Product` model:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(
        ..., 
        gt=0,               # price must be greater than 0
        lt=10000,           # price must be less than 10000
        multiple_of=0.01,   # price must be a multiple of 0.01 (e.g., valid cent values)
        allow_inf_nan=False # do not allow infinity or NaN values
    )
    quantity: int = Field(
        ..., 
        ge=1,    # quantity must be at least 1
        le=1000  # quantity must be less than or equal to 1000
    )

# Valid instance
product = Product(name="Laptop", price=999.99, quantity=10)
print(product)
# Output: name='Laptop' price=999.99 quantity=10

# This will raise validation errors because:
# - price is 0 (which is not greater than 0)
# - quantity is 0 (which is less than the minimum of 1)
try:
    invalid_product = Product(name="Faulty", price=0, quantity=0)
except Exception as e:
    print(e)
```

In this example, the constraints ensure that:
- **`price`** is greater than 0, less than 10000, and adheres to valid monetary values by being a multiple of 0.01. Also, it disallows special float values like `inf` or `NaN`.
- **`quantity`** is between 1 and 1000 inclusive.

Attempting to create a product with invalid numeric values results in a validation error, as demonstrated in the `try` block.

---

### Excluding and Deprecated Fields

You might want to exclude certain fields from serialization or mark a field as deprecated. The `exclude` and `deprecated` parameters can help with that.

**Example:**

```python
from pydantic import BaseModel, Field

class LegacyModel(BaseModel):
    active: bool = Field(default=True, description="Indicates if the model is active.")
    old_field: str = Field(
        default="deprecated",
        deprecated=True,
        exclude=True,
        description="This field is deprecated and excluded from serialization."
    )

# Usage:
legacy = LegacyModel()

# Serialize the model (excluded fields will not appear in the output):
print(legacy.model_dump())
# Output: {'active': True}

# Accessing the deprecated field (still accessible in the model):
print(legacy.old_field)  # Output: deprecated
```

---

### Key Points:
- **`exclude=True`**: Excludes the field from serialization (e.g., `model_dump()`).
- **`deprecated=True`**: Marks the field as deprecated, signaling that it should no longer be used.
- **`description`**: Provides a description of the field for documentation purposes.

This approach is useful for maintaining backward compatibility while phasing out old fields.

---

## Advanced Examples

### Custom JSON Schema Extra

If you need to provide extra metadata for JSON schema generation (for instance, to customize how your API documentation appears), use the `json_schema_extra` parameter:

```python
from pydantic import BaseModel, Field

class Book(BaseModel):
    title: str = Field(..., json_schema_extra={"example": "The Great Gatsby"})
    pages: int = Field(..., json_schema_extra={"minimum": 1})

print(Book.schema_json(indent=2))
```

When you run this code, the output printed to the console will be the JSON schema for the `Book` model with the extra metadata included. The output will look similar to the following:

```json
{
  "title": "Book",
  "type": "object",
  "properties": {
    "title": {
      "title": "Title",
      "type": "string",
      "example": "The Great Gatsby"
    },
    "pages": {
      "title": "Pages",
      "type": "integer",
      "minimum": 1
    }
  },
  "required": [
    "title",
    "pages"
  ]
}
```

This JSON schema includes the extra properties:
- For the `title` field, an example value `"The Great Gatsby"`.
- For the `pages` field, a minimum value of `1`.

API documentation generators can use these details to better document your API endpoints.

---

### Computed Fields

Pydantic also supports computed fields through the `@computed_field` decorator. Computed fields are read-only fields computed from other fields. They are automatically included in model serialization when needed.

```python
from pydantic import BaseModel, Field, computed_field

class Rectangle(BaseModel):
    width: int
    length: int

    @computed_field
    @property
    def area(self) -> int:
        """Calculate the area of the rectangle."""
        return self.width * self.length

rect = Rectangle(width=3, length=4)
print(rect.model_dump())
# Output: {'width': 3, 'length': 4, 'area': 12}
```

Computed fields can also be given aliases, customized representation, and other metadata:

```python
import random
from pydantic import BaseModel, Field, computed_field

class Square(BaseModel):
    width: float

    @computed_field(alias='the magic number', repr=False)
    @property
    def random_number(self) -> int:
        """Generate a random number for demonstration."""
        return random.randint(0, 1000)

sq = Square(width=1.3)
print(repr(sq))  # 'random_number' will not appear in repr due to repr=False
print(sq.random_number)
```

---

## Constraint Examples

### Numeric Constraints

There are several keyword arguments that can be used to constrain numeric values:

- **gt**: Greater than
- **ge**: Greater than or equal to
- **lt**: Less than
- **le**: Less than or equal to
- **multiple_of**: A multiple of the given number
- **allow_inf_nan**: Allow `inf`, `-inf`, or `nan` values

Here's an example demonstrating these constraints:

```python
from pydantic import BaseModel, Field

class Foo(BaseModel):
    positive: int = Field(gt=0)
    non_negative: int = Field(ge=0)
    negative: int = Field(lt=0)
    non_positive: int = Field(le=0)
    even: int = Field(multiple_of=2)
    love_for_pydantic: float = Field(allow_inf_nan=True)

foo = Foo(
    positive=1,
    non_negative=0,
    negative=-1,
    non_positive=0,
    even=2,
    love_for_pydantic=float('inf'),
)
print(foo)
```

**Expected Output:**

```
positive=1 non_negative=0 negative=-1 non_positive=0 even=2 love_for_pydantic=inf
```

In this example:
- `positive` must be greater than 0.
- `non_negative` must be 0 or greater.
- `negative` must be less than 0.
- `non_positive` must be 0 or less.
- `even` must be a multiple of 2.
- `love_for_pydantic` allows special float values like infinity.

---

### String Constraints

You can also constrain string fields using various keyword arguments:

- **min_length**: Minimum length of the string.
- **max_length**: Maximum length of the string.
- **pattern**: A regular expression that the string must match.

Here's an example:

```python
from pydantic import BaseModel, Field

class Foo(BaseModel):
    short: str = Field(min_length=3)
    long: str = Field(max_length=10)
    regex: str = Field(pattern=r'^\d*$')

foo = Foo(short='foo', long='foobarbaz', regex='123')
print(foo)
```

**Expected Output:**

```
short='foo' long='foobarbaz' regex='123'
```

In this example:
- `short` must have at least 3 characters.
- `long` must have no more than 10 characters.
- `regex` must consist entirely of digits, as enforced by the pattern `r'^\d*$'`.

---

### Strict Mode

The `strict` parameter on a Field specifies whether the field should be validated in "strict mode." In strict mode, Pydantic will throw a validation error if the input type does not match exactly, instead of attempting to coerce the input to the specified type.

Here's an example:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(strict=True)     # No coercion: input must be a str
    age: int = Field(strict=False)     # Coercion allowed: input can be converted to int

# This will pass because the name is a string and age is coerced from a string to an integer.
user = User(name='John', age='42')
print(user)
```

**Expected Output:**

```
name='John' age=42
```

In this example:
- The `name` field is in strict mode, so only a string value is acceptable.
- The `age` field is not in strict mode, so a string that represents an integer (e.g., `'42'`) is automatically coerced to an integer.

---

## Conclusion

The **Field** class is a versatile and powerful tool that lets you configure how your model fields behave in terms of default values, validation, aliasing, metadata, and more. Whether you need simple default values, dynamic defaults with a `default_factory`, or complex constraints and JSON schema customizations, **Field** provides the necessary features to meet your needs.

The examples above—ranging from basic defaults to numeric and string constraints, as well as strict mode—illustrate many common use cases. The detailed parameters table should serve as a handy reference. With **Field**, you can write cleaner, more robust models that integrate seamlessly with Pydantic’s data validation and serialization system.

