# Introduction to Request Parameters in FastAPI

FastAPI offers a robust way to declare and validate request parameters in your API endpoints. By using special functions such as `Query()`, `Path()`, `Body()`, `Cookie()`, `Header()`, `Form()`, and `File()`, you can easily extract data from different parts of the request. This article introduces these request parameters, explains how to use them with practical examples (including error handling), and provides a handy reference table for common parameters.

## Table of Contents

1. [Overview](#overview)
2. [Using Query Parameters](#using-query-parameters)
3. [Using Path Parameters](#using-path-parameters)
4. [Using Body Parameters](#using-body-parameters)
5. [Using Cookie, Header, Form, and File](#using-cookie-header-form-and-file)
6. [Error Handling with Request Parameters](#error-handling-with-request-parameters)
7. [Reference Table of Request Parameters](#reference-table-of-request-parameters)
8. [Summary](#summary)

## Overview

FastAPI uses type hints and dependency injection to declare the source of the data. For example, if you want to retrieve data from the query string, you can use `Query()`. Similarly, you can use `Path()`, `Body()`, `Cookie()`, `Header()`, `Form()`, and `File()` to extract data from various parts of the request. Each of these functions accepts parameters for validation, metadata, and aliasing, making them very powerful for API development.

## Using Query Parameters

Query parameters are used to pass key-value pairs in the URL after the question mark (`?`).

**Example:**

```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/items/")
async def read_items(
    # The query parameter "q" is optional.
    # Query() is used to add validation and metadata.
    q: Optional[str] = Query(
        None,               # Default value is None
        min_length=3,       # Minimum length of the query string is 3
        max_length=50,      # Maximum length of the query string is 50
        title="Query string",
        description="Search query for the items"
    )
):
    # Return the query parameter value as a JSON response.
    return {"query": q}
```

In this example, the `Query()` function ensures that if the query parameter `q` is provided, it must be between 3 and 50 characters long.

---

## Using Path Parameters

Path parameters are embedded in the URL path. They are declared using `Path()`.

**Example:**

```python
from fastapi import FastAPI, Path
from typing import Annotated

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(
    # The "item_id" parameter is extracted from the URL path.
    # Annotated is used to provide extra metadata via Path().
    item_id: Annotated[int, Path(
        title="The ID of the item to get",   # Description for documentation
        ge=1                                 # Ensure the item_id is greater than or equal to 1
    )]
):
    # Return the item_id as part of the JSON response.
    return {"item_id": item_id}
```

Here, the `item_id` parameter is extracted from the URL and validated as an integer that must be at least 1.

---

## Using Body Parameters

When you need to pass complex data (such as JSON) in the request body, you use `Body()`. FastAPI leverages Pydantic models to perform validation and data parsing.

**Example:**

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

# Define a Pydantic model for the item
class Item(BaseModel):
    name: str
    description: str | None = None  # Optional description
    price: float
    tax: float | None = None         # Optional tax field

@app.post("/items/")
async def create_item(
    # The request body must match the Item model.
    # Body() is used to add metadata and documentation details.
    item: Item = Body(
        ...,  # Ellipsis indicates that the item field is required
        title="Item data",
        description="The item to create"
    )
):
    # Return the created item data as a JSON response.
    return {"item": item}
```

In this example, FastAPI uses Pydantic to validate the JSON body against the `Item` model.

---

## Using Cookie, Header, Form, and File

FastAPI also provides functions to extract data from cookies, headers, form fields, and files.

### Cookie Example

```python
from fastapi import FastAPI, Cookie

app = FastAPI()

@app.get("/cookies/")
async def read_cookie(
    # Extract the "ga" cookie value. It is optional.
    ga: str | None = Cookie(
        None, 
        title="Google Analytics cookie"  # Description for documentation
    )
):
    # Return the cookie value in a JSON response.
    return {"ga_cookie": ga}
```

### Header Example

```python
from fastapi import FastAPI, Header

app = FastAPI()

@app.get("/headers/")
async def read_header(
    # Extract the User-Agent header.
    # convert_underscores=True automatically converts underscores to hyphens.
    user_agent: str = Header(
        ..., 
        convert_underscores=True
    )
):
    # Return the User-Agent header in a JSON response.
    return {"User-Agent": user_agent}
```

### Form Example

```python
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/login/")
async def login(
    # Retrieve the username and password from form data.
    username: str = Form(...),  # Required field
    password: str = Form(...)   # Required field
):
    # Return the username in a JSON response (do not return the password in production!)
    return {"username": username}
```

### File Example

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/upload/")
async def upload_file(
    # UploadFile is used to handle file uploads.
    # File(...) marks the file as required.
    file: UploadFile = File(...)
):
    # Read the content of the uploaded file asynchronously.
    content = await file.read()
    # Return the filename and file size in a JSON response.
    return {"filename": file.filename, "size": len(content)}
```

---

## Error Handling with Request Parameters

FastAPI automatically handles errors when request parameters fail validation. For example, if a query parameter does not meet the length requirements or if the path parameter cannot be converted to the specified type, FastAPI will return an appropriate HTTP error response (typically a **422 Unprocessable Entity**).

**Example of Error Handling:**

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/validate/")
async def validate_query(
    # The query parameter "q" is required and must have a length between 3 and 10.
    q: str = Query(
        ...,            # Ellipsis indicates that this parameter is required
        min_length=3,
        max_length=10
    )
):
    # Return the query value if validation passes.
    return {"query": q}
```

If you try to access `/validate/?q=ab` (with a string too short), FastAPI returns a response similar to:

```json
{
  "detail": [
    {
      "loc": ["query", "q"],
      "msg": "ensure this value has at least 3 characters",
      "type": "value_error.any_str.min_length",
      "ctx": {"limit_value": 3}
    }
  ]
}
```

This built-in error handling ensures that your API clients receive clear feedback on why their request failed.

---

## Reference Table of Request Parameters

The following table summarizes some of the common parameters available in request parameter functions:

| Parameter            | Description                                                                                                    | Default Value                | Applicable To                 |
|----------------------|----------------------------------------------------------------------------------------------------------------|------------------------------|-------------------------------|
| `default`            | Default value if the parameter field is not set.                                                     | `Undefined` (or `...`)       | All request parameter types   |
| `default_factory`    | A callable to generate a default value (used for compatibility).                                      | `_Unset`                     | All request parameter types   |
| `alias`              | Alternative name for the parameter field.                                                            | `None`                       | All request parameter types   |
| `title`              | Human-readable title for the parameter.                                                              | `None`                       | All request parameter types   |
| `description`        | Human-readable description of the parameter.                                                         | `None`                       | All request parameter types   |
| `min_length` / `max_length` | Minimum/maximum length constraints for strings.                                               | `None`                       | `Query()`, `Path()`, `Body()`, etc. |
| `gt`, `ge`, `lt`, `le`       | Numeric constraints: greater than, greater than or equal, <br>less than, less than or equal. | `None`                       | `Query()`, `Path()`, `Body()`, etc. |
| `pattern`            | Regular expression pattern for string matching.                                                      | `None`                       | `Query()`, `Path()`, `Body()`, etc. |
| `media_type`         | The media type for the parameter field.                                                              | Depends on context | `Body()`, `Form()`, `File()`   |
| `embed`              | When `True`, the parameter is expected as a key in a JSON body <br>instead of the entire JSON body (only for Body parameters). | `None`                       | `Body()`                     |

> **Note:** The table above is a simplified reference. Each request parameter function supports additional parameters for advanced use cases. Refer to the [FastAPI documentation](https://fastapi.tiangolo.com/) for a complete list of options.

---

## Summary

FastAPI’s request parameter functions such as `Query()`, `Path()`, `Body()`, `Cookie()`, `Header()`, `Form()`, and `File()` provide a flexible and powerful way to extract, validate, and document the data coming into your API endpoints. By leveraging these tools, you can ensure that your API adheres to strict data validation rules while providing clear feedback to clients when errors occur.

Whether you are building simple endpoints or complex APIs with multiple request parameter types, FastAPI’s integrated features make it easier to maintain clear, consistent, and robust API contracts.

