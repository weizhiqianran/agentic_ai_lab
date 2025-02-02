# Introduction to `fastapi.FastAPI`: A Comprehensive Guide

This article introduces the [`FastAPI`](https://fastapi.tiangolo.com/) class—the heart of any FastAPI application. We will cover its public methods and features with examples for each one, along with a table summarizing the key functions. In addition, we provide **5 different examples** of using `app.middleware` to add global functionality to your FastAPI application.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Creating a FastAPI Application](#creating-a-fastapi-application)
3. [Public Methods of FastAPI](#public-methods-of-fastapi)
   - [openapi()](#openapi)
   - [websocket()](#websocket)
   - [include_router()](#include_router)
   - [HTTP Operation Decorators](#http-operation-decorators)
     - [get()](#get)
     - [put()](#put)
     - [post()](#post)
     - [delete()](#delete)
     - [options()](#options)
     - [head()](#head)
     - [patch()](#patch)
     - [trace()](#trace)
   - [Application Lifecycle Methods](#application-lifecycle-methods)
     - [on_event()](#on_event)
     - [middleware()](#middleware)
     - [exception_handler()](#exception_handler)
4. [Additional Middleware Examples](#additional-middleware-examples)
   - [Example 1: Logging Middleware](#example-1-logging-middleware)
   - [Example 2: Adding Custom Headers to Responses](#example-2-adding-custom-headers-to-responses)
   - [Example 3: Error Handling Middleware](#example-3-error-handling-middleware)
   - [Example 4: Rate Limiting Middleware](#example-4-rate-limiting-middleware)
   - [Example 5: Request Timing Middleware](#example-5-request-timing-middleware)
5. [Summary Table](#summary-table)
6. [Conclusion](#conclusion)

---

## Introduction

The `fastapi.FastAPI` class is the main entry point for building an API with FastAPI. In addition to standard HTTP routes, it supports WebSocket endpoints, middleware, exception handling, and lifecycle event handling. This guide shows you how to use each public method with concise examples and inline comments.

---

## Creating a FastAPI Application

Before you start adding routes or middleware, you need to create a FastAPI instance:

```python
from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI(title="My Awesome API", version="1.0.0")

# The app object is your main application entry point
```

---

## Public Methods of FastAPI

Below, we cover many of the public methods available on the FastAPI class.

### openapi()

The `openapi()` method generates the OpenAPI schema for your application. The first call caches the schema for faster subsequent calls.

```python
@app.get("/openapi-schema")
def get_openapi_schema():
    # Generate and return the OpenAPI schema
    return app.openapi()
```

---

### websocket()

The `websocket()` decorator allows you to define WebSocket endpoints.

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection
    await websocket.accept()
    # Echo received messages back to the client
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")
```

---

### include_router()

The `include_router()` method allows you to split your application into multiple modules. This is useful for organizing large applications.

```python
from fastapi import APIRouter

# Define a separate router in another file or section
router = APIRouter()

@router.get("/users")
def read_users():
    # Sample endpoint in the router
    return [{"username": "alice"}, {"username": "bob"}]

# Include the router in the main application with an optional prefix
app.include_router(router, prefix="/api", tags=["users"])
```

---

### HTTP Operation Decorators

FastAPI provides decorators for common HTTP methods. Each decorator automatically handles request parsing, response validation, and OpenAPI documentation.

#### get()

The `get()` decorator registers a GET endpoint.

```python
@app.get("/items", tags=["items"])
def read_items():
    # Return a list of items
    return [{"item_id": 1, "name": "Item One"}, {"item_id": 2, "name": "Item Two"}]
```

#### put()

The `put()` decorator registers a PUT endpoint. This example uses Pydantic to validate the request body.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None

@app.put("/items/{item_id}", tags=["items"])
def update_item(item_id: int, item: Item):
    # Replace the item with new data
    return {"item_id": item_id, "item": item.dict(), "message": "Item updated"}
```

#### post()

The `post()` decorator registers a POST endpoint for creating resources.

```python
@app.post("/items", tags=["items"], status_code=201)
def create_item(item: Item):
    # Create a new item (in memory for this example)
    return {"item": item.dict(), "message": "Item created"}
```

#### delete()

The `delete()` decorator registers a DELETE endpoint.

```python
@app.delete("/items/{item_id}", tags=["items"])
def delete_item(item_id: int):
    # Delete an item by its ID
    return {"item_id": item_id, "message": "Item deleted"}
```

#### options()

The `options()` decorator registers an OPTIONS endpoint, which can be used for CORS preflight or to return allowed operations.

```python
@app.options("/items", tags=["items"])
def options_items():
    # Return allowed methods for the /items endpoint
    return {"methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]}
```

#### head()

The `head()` decorator registers a HEAD endpoint. HEAD requests are similar to GET but do not return a response body.

```python
from fastapi import Response

@app.head("/items", status_code=204, tags=["items"])
def head_items(response: Response):
    # Set a custom header
    response.headers["X-Custom-Header"] = "Value"
    # No body is returned for HEAD requests
```

#### patch()

The `patch()` decorator registers a PATCH endpoint for partial updates.

```python
@app.patch("/items/{item_id}", tags=["items"])
def patch_item(item_id: int, item: Item):
    # Update part of the item resource
    return {"item_id": item_id, "updated_fields": item.dict(exclude_unset=True), "message": "Item partially updated"}
```

#### trace()

The `trace()` decorator registers a TRACE endpoint. TRACE is rarely used in production but can be useful for diagnostic purposes.

```python
@app.trace("/items/trace", tags=["items"])
def trace_item():
    # This endpoint simply echoes back the received request information
    return {"message": "TRACE endpoint reached"}
```

---

### Application Lifecycle Methods

FastAPI supports hooks for handling events, middleware for processing requests/responses, and exception handlers.

#### on_event()

The `on_event()` decorator registers startup and shutdown event handlers. (Note: FastAPI now recommends using lifespan handlers.)

```python
@app.on_event("startup")
async def startup_event():
    # Code to run on application startup
    print("Application is starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    # Code to run on application shutdown
    print("Application is shutting down...")
```

#### middleware()

The `middleware()` decorator lets you add custom middleware to the application. Middleware can process requests before they reach the endpoints and responses before they are sent back to clients.

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()  # Start timer
    response = await call_next(request)
    process_time = time.time() - start_time  # Calculate process time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

#### exception_handler()

The `exception_handler()` decorator registers a custom handler for exceptions. This is useful to handle errors gracefully and provide custom error messages.

```python
from fastapi import Request
from fastapi.responses import JSONResponse

# Define a custom exception
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

# Register the exception handler for UnicornException
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    # Return a custom JSON error response
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )

# Example endpoint that raises the custom exception
@app.get("/unicorns")
def read_unicorns():
    raise UnicornException(name="Charlie")
```

---

## Additional Middleware Examples

Below are **5 different examples** of using `app.middleware` in FastAPI. Middleware allows you to intercept and modify requests and responses globally before they reach your route handlers or after they leave them.

---

### Example 1: Logging Middleware

This middleware logs incoming requests and outgoing responses, including the request method, URL, and response status code.

```python
from fastapi import FastAPI, Request
import time

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log the incoming request
    start_time = time.time()
    print(f"Incoming request: {request.method} {request.url}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    # Log the outgoing response
    process_time = time.time() - start_time
    print(f"Outgoing response: {response.status_code} (took {process_time:.2f} seconds)")
    
    return response

@app.get("/")
async def root():
    return {"message": "Hello, World!"}
```

---

### Example 2: Adding Custom Headers to Responses

This middleware adds custom headers to every response.

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    # Process the request and get the response
    response = await call_next(request)
    
    # Add custom headers to the response
    response.headers["X-Custom-Header"] = "MyCustomValue"
    response.headers["X-Another-Header"] = "AnotherValue"
    
    return response

@app.get("/")
async def root():
    return {"message": "Hello, World!"}
```

---

### Example 3: Error Handling Middleware

This middleware catches exceptions and returns a custom error response.

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.middleware("http")
async def error_handler(request: Request, call_next):
    try:
        # Process the request and get the response
        response = await call_next(request)
        return response
    except HTTPException as http_exc:
        # Handle FastAPI's HTTPException
        return JSONResponse(
            status_code=http_exc.status_code,
            content={"message": http_exc.detail},
        )
    except Exception as e:
        # Handle unexpected exceptions
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error", "error": str(e)},
        )

@app.get("/error")
async def raise_error():
    raise HTTPException(status_code=400, detail="Something went wrong!")
```

---

### Example 4: Rate Limiting Middleware

This middleware limits the number of requests a client can make within a certain time period.

```python
from fastapi import FastAPI, Request, HTTPException
import time

app = FastAPI()

# Store request timestamps for each client
request_timestamps = {}

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Allow only 5 requests per 10 seconds per client
    if client_ip in request_timestamps:
        timestamps = request_timestamps[client_ip]
        timestamps = [t for t in timestamps if current_time - t < 10]  # Keep only recent requests
        if len(timestamps) >= 5:
            raise HTTPException(status_code=429, detail="Too many requests")
        timestamps.append(current_time)
    else:
        request_timestamps[client_ip] = [current_time]
    
    # Process the request and get the response
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {"message": "Hello, World!"}
```

---

### Example 5: Request Timing Middleware

This middleware measures the time taken to process a request and adds it as a response header.

```python
from fastapi import FastAPI, Request
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Record the start time
    start_time = time.time()
    
    # Process the request and get the response
    response = await call_next(request)
    
    # Calculate the processing time
    process_time = time.time() - start_time
    
    # Add the processing time as a custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.get("/")
async def root():
    return {"message": "Hello, World!"}
```

---

### Summary of Middleware Examples

1. **Logging Middleware**: Logs incoming requests and outgoing responses.
2. **Custom Headers Middleware**: Adds custom headers to every response.
3. **Error Handling Middleware**: Catches exceptions and returns custom error responses.
4. **Rate Limiting Middleware**: Limits the number of requests per client.
5. **Request Timing Middleware**: Measures and adds the request processing time as a header.

---

## Summary Table

Below is a table summarizing the public methods covered in this article:

| **Method**             | **Description**                                                                 | **Example Use-case**                        |
|------------------------|---------------------------------------------------------------------------------|---------------------------------------------|
| `openapi()`            | Generates (and caches) the OpenAPI schema                                       | Serve `/openapi.json` schema                |
| `websocket()`          | Registers a WebSocket endpoint                                                  | Real-time chat or notifications             |
| `include_router()`     | Includes an `APIRouter` into the main app                                       | Organize routes into modules                |
| `get()`                | Registers an HTTP GET endpoint                                                  | Retrieve data                               |
| `put()`                | Registers an HTTP PUT endpoint for full updates                                | Replace an entire resource                  |
| `post()`               | Registers an HTTP POST endpoint for creation                                   | Create new resources                        |
| `delete()`             | Registers an HTTP DELETE endpoint                                               | Delete a resource                           |
| `options()`            | Registers an HTTP OPTIONS endpoint                                              | Provide allowed methods info                |
| `head()`               | Registers an HTTP HEAD endpoint                                                 | Return headers without body                 |
| `patch()`              | Registers an HTTP PATCH endpoint for partial updates                           | Update parts of a resource                  |
| `trace()`              | Registers an HTTP TRACE endpoint (rarely used)                                  | Diagnostic purposes                         |
| `on_event()`           | Registers startup/shutdown event handlers                                      | Run initialization or cleanup tasks         |
| `middleware()`         | Adds middleware to process requests/responses                                  | Logging, timing, authentication, etc.       |
| `exception_handler()`  | Adds custom exception handlers for graceful error responses                    | Customize error messaging                   |

---

## Conclusion

In this guide we have introduced the `fastapi.FastAPI` class and provided examples for each of its public methods—from HTTP operation decorators and WebSocket endpoints to lifecycle event handlers and middleware. We also added five detailed middleware examples to demonstrate how you can extend your application globally. Whether you are building standard HTTP endpoints, WebSocket routes, or adding custom middleware and exception handlers, FastAPI makes it easy to build a robust and well-documented API. For more details, visit the [official FastAPI documentation](https://fastapi.tiangolo.com/).

