# Middleware in FastAPI

FastAPI’s middleware is a powerful tool for intercepting and processing HTTP requests and responses. It enables you to add cross-cutting concerns such as logging, authentication, performance monitoring, and more without cluttering your route handlers. In this article, we introduce Middleware in FastAPI with detailed explanations, five practical examples, and a discussion of common use-cases.

## Table of Contents

1. [Introduction](#introduction)
2. [What is Middleware?](#what-is-middleware)
3. [Use-cases of Middleware](#use-cases-of-middleware)
4. [How Middleware Works in FastAPI](#how-middleware-works-in-fastapi)
5. [Practical Examples](#practical-examples)
    - [Example 1: Logging Middleware](#example-1-logging-middleware)
    - [Example 2: CORS Middleware](#example-2-cors-middleware)
    - [Example 3: Custom Authentication Middleware](#example-3-custom-authentication-middleware)
    - [Example 4: Performance Monitoring Middleware](#example-4-performance-monitoring-middleware)
    - [Example 5: Response Header Injection Middleware](#example-5-response-header-injection-middleware)
6. [Comparison Table](#comparison-table)
7. [Conclusion](#conclusion)

---

## Introduction

Middleware in FastAPI is used to execute code before and/or after each request, allowing you to introduce functionality such as logging, security checks, data modification, and performance monitoring. This separation of concerns makes your application more modular and maintainable.

---

## What is Middleware?

Middleware is a layer that sits between the client's request and the application's route handlers. In FastAPI, middleware functions can:

- Preprocess requests (e.g., validate headers, log incoming data)
- Modify requests before they reach the endpoint
- Process responses before they are sent back to the client
- Handle cross-cutting concerns such as security, error handling, and performance logging

---

## Use-cases of Middleware

Middleware can be applied to a variety of use-cases, including but not limited to:

- **Logging:** Capture details of incoming requests and outgoing responses for debugging and monitoring.
- **Authentication & Authorization:** Validate API keys, tokens, or other credentials before processing the request.
- **Performance Monitoring:** Measure and log request processing time to help identify performance bottlenecks.
- **Request/Response Modification:** Inject or modify headers, compress responses, or transform request data.
- **CORS Handling:** Manage Cross-Origin Resource Sharing (CORS) to control access to your API from different domains.

These use-cases illustrate how middleware helps separate concerns and maintain clean, modular code.

---

## How Middleware Works in FastAPI

In FastAPI, middleware functions are executed in the order they are added to the application. When a request is received, it passes through each middleware layer sequentially before reaching the designated route handler. After the route handler processes the request, the response passes back through the middleware layers in reverse order, allowing you to modify the response on its way back to the client.

A typical middleware function looks like this:

```python
from fastapi import FastAPI, Request
import time

app = FastAPI()

@app.middleware("http")
async def sample_middleware(request: Request, call_next):
    # Code executed before passing the request to the route handler
    start_time = time.time()
    
    # Process the request and get the response from the next handler/middleware
    response = await call_next(request)
    
    # Code executed after the request is processed (e.g., adding a header)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

This function shows how you can capture request details and modify the response.

---

## Practical Examples

### Example 1: Logging Middleware

**Description:**  
This middleware logs the HTTP method and URL of every incoming request. It is helpful for debugging and tracing the request flow in your application.

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Log the HTTP method and URL of the incoming request
    print(f"Received request: {request.method} {request.url}")
    # Process the request and obtain the response
    response = await call_next(request)
    return response
```

---

### Example 2: CORS Middleware

**Description:**  
CORS (Cross-Origin Resource Sharing) is essential when your API is accessed from different domains. FastAPI offers built-in CORS middleware to handle such scenarios easily.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)
```

---

### Example 3: Custom Authentication Middleware

**Description:**  
This middleware checks for the presence of a custom authentication token (`X-Auth-Token`) in the request headers. If the token is missing or invalid, it raises an HTTP 401 error. This is a basic example to illustrate how middleware can enforce security policies.

```python
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Retrieve the custom auth token from the request headers
    auth_token = request.headers.get("X-Auth-Token")
    # Validate the token; here, we expect a token with the value "secret-token"
    if not auth_token or auth_token != "secret-token":
        # If token is missing or invalid, raise an Unauthorized error
        raise HTTPException(status_code=401, detail="Unauthorized access - invalid token")
    # If valid, continue processing the request
    response = await call_next(request)
    return response
```

---

### Example 4: Performance Monitoring Middleware

**Description:**  
Performance monitoring middleware measures the time taken to process each request. It logs the duration, which can be useful for detecting slow endpoints and performance bottlenecks.

```python
from fastapi import FastAPI, Request
import time
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    # Record the start time before processing the request
    start_time = time.time()
    
    # Process the request and get the response
    response = await call_next(request)
    
    # Calculate the time taken to process the request
    duration = time.time() - start_time
    
    # Log the method, URL, and duration of the request
    logging.info(f"{request.method} {request.url} completed in {duration:.4f} seconds")
    return response
```

---

### Example 5: Response Header Injection Middleware

**Description:**  
This middleware injects additional headers into the response. It can be used to add metadata or versioning information to every response from your API.

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def response_header_injection_middleware(request: Request, call_next):
    # Process the request and obtain the response from the next handler/middleware
    response = await call_next(request)
    
    # Inject custom headers into the response
    response.headers["X-App-Version"] = "1.0.0"  # Example version header
    response.headers["X-Powered-By"] = "FastAPI"
    return response
```

---

## Comparison Table

| Middleware Type                        | Use Case                                        | Key Benefit                          |
|----------------------------------------|-------------------------------------------------|--------------------------------------|
| **Logging Middleware**                 | Capturing request details for debugging         | Simplifies debugging and monitoring  |
| **CORS Middleware**                    | Handling Cross-Origin Resource Sharing          | Enables secure cross-domain access   |
| **Authentication Middleware**          | Validating authentication tokens                | Improves API security                |
| **Performance Monitoring Middleware**  | Measuring request processing time               | Identifies performance bottlenecks   |
| **Response Header Injection Middleware** | Adding custom metadata to responses            | Enhances client-side context         |

---

## Conclusion

Middleware in FastAPI offers a modular way to extend your application's functionality without cluttering your route handlers. Whether you're logging requests, enforcing security, monitoring performance, or injecting headers, middleware makes it easy to manage these cross-cutting concerns. The examples provided here cover common scenarios and serve as a starting point for more advanced implementations. With middleware, you can ensure that your application remains both efficient and maintainable.
