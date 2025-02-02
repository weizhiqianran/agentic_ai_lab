# Introducing FastAPI APIRouter: A Comprehensive Guide

FastAPI makes building modern APIs fast and simple. One of the powerful tools it provides is the `APIRouter` class, which lets you organize your API’s endpoints into multiple files or logical groups. This article will cover what `APIRouter` is, how it works, and how you can use it effectively.

---

## Table of Contents

1. [What is APIRouter?](#what-is-apirouter)
2. [How Does APIRouter Work?](#how-does-apirouter-work)
3. [Key Parameters of APIRouter](#key-parameters-of-apirouter)
    - [Router-Level Parameters](#router-level-parameters)
    - [Path Operation Decorator Parameters](#path-operation-decorator-parameters)
4. [Examples and Explanations](#examples-and-explanations)
    - [Basic Usage](#basic-usage)
    - [Including Routers](#including-routers)
    - [Using WebSockets with APIRouter](#using-websockets-with-apirouter)
    - [Versioned API with APIRouter](#versioned-api-with-apirouter)
    - [API with Multiple Tags and Descriptions](#api-with-multiple-tags-and-descriptions)
    - [API with Custom Response Models and Status Codes](#api-with-custom-response-models-and-status-codes)
5. [Advanced Usage and Considerations](#advanced-usage-and-considerations)
6. [Conclusion](#conclusion)

---

## What is APIRouter?

The `APIRouter` class in FastAPI is a tool to organize and group path operations (endpoints) into a single router. You can define endpoints in separate files or modules and then include these routers in your main FastAPI application. This modular approach helps in maintaining larger codebases by separating concerns and grouping related functionality.

### Key Points:
- **Modularity:** Break down your API into multiple routers.
- **Reusability:** Create common groups of endpoints that can be reused or even mounted on different apps.
- **Organization:** Keep your code clean by separating endpoint logic from application startup code.

---

## How Does APIRouter Work?

When you create an instance of `APIRouter`, you’re essentially creating a container for a set of related path operations. Later, you “mount” or include this router in your main FastAPI app (or even nest routers within routers).

Each router can have its own:
- **Prefix:** A path prefix that will be prepended to all endpoints in the router.
- **Tags:** Group tags for API documentation (visible in the OpenAPI docs).
- **Dependencies:** Global dependencies that will apply to every endpoint within the router.
- **Default response class and other response configurations.**

When you call methods like `.get()`, `.post()`, etc. on an `APIRouter` instance, you register path operations along with any special configuration for that operation. Finally, you include the router into your FastAPI app using `app.include_router(router)`, which integrates the endpoints into the application’s overall routing table.

---

## Key Parameters of APIRouter

FastAPI’s `APIRouter` constructor accepts many parameters. Below are some of the most common ones and what they do.

### Router-Level Parameters

| **Parameter**                    | **Type**          | **Default**  | **Description**                                                                                          |
|----------------------------------|-------------------|--------------|----------------------------------------------------------------------------------------------------------|
| `prefix`                       | `str`             | `""`         | A path prefix for all endpoints in the router.                                                         |
| `tags`                         | `List[str]`       | `None`       | Tags applied to all endpoints (used in OpenAPI documentation).                                           |
| `dependencies`                 | `List[Depends]`   | `None`       | Global dependencies applied to all endpoints in the router.                                            |
| `default_response_class`       | `Type[Response]`  | `JSONResponse` | The default response class to be used if no response is explicitly returned.                           |
| `responses`                    | `Dict`            | `None`       | Additional responses for documentation.                                                                |
| `redirect_slashes`             | `bool`            | `True`       | Whether to redirect URL requests with/without a trailing slash.                                         |
| `include_in_schema`            | `bool`            | `True`       | Whether to include the endpoints in the OpenAPI schema.                                                |
| `route_class`                  | `Type[APIRoute]`  | `APIRoute`   | Custom route class to be used for the endpoints in this router.                                         |

These parameters allow you to configure the router as a whole. For example, you might add a prefix of `/users` to a router that contains all user-related endpoints.

### Path Operation Decorator Parameters

Each path operation decorator (such as `.get()`, `.post()`, etc.) accepts additional parameters to further configure the endpoint. Here’s a simplified table of some common ones:

| **Parameter**                | **Type**            | **Default**      | **Description**                                                                                           |
|------------------------------|---------------------|------------------|-----------------------------------------------------------------------------------------------------------|
| `path`                       | `str`               | (required)       | The URL path for the endpoint.                                                                            |
| `response_model`             | `Any`               | `None`           | A Pydantic model (or type) used to validate and document the response.                                    |
| `status_code`                | `int`               | `None`           | The default status code for the response.                                                                 |
| `tags`                       | `List[str]`         | `None`           | Additional tags for this endpoint (merged with router-level tags).                                        |
| `dependencies`               | `List[Depends]`     | `None`           | Endpoint-specific dependencies.                                                                           |
| `summary`                    | `str`               | `None`           | A short summary for the endpoint (shown in docs).                                                         |
| `description`                | `str`               | `None`           | A longer description for the endpoint (supports Markdown).                                                |
| `include_in_schema`          | `bool`              | `True`           | Whether to include this endpoint in the OpenAPI documentation.                                            |

By combining router-level parameters with path-specific ones, you gain full control over how your endpoints are organized and documented.

---

## Examples and Explanations

This section contains several examples that illustrate how to use `APIRouter` to build robust, well-organized APIs.

### Basic Usage

The following example shows how to create a router, define a simple GET endpoint, and include the router in your main FastAPI app.

```python
from fastapi import FastAPI, APIRouter

# Create the main FastAPI application
app = FastAPI()

# Create a router instance with a prefix and tags
router = APIRouter(prefix="/users", tags=["users"])

# Define an endpoint within the router
@router.get("/")
async def read_users():
    """
    Returns a list of users.
    """
    return [{"username": "Rick"}, {"username": "Morty"}]

# Include the router in the main app
app.include_router(router)
```

**Explanation:**

- **Router Creation:**  
  An `APIRouter` instance is created with the prefix `/users`. This means that the endpoint defined as `/` will actually be accessible at `/users/`.

- **Endpoint Definition:**  
  The `@router.get("/")` decorator registers a GET operation. The function `read_users` returns a list of user dictionaries. This means a client can make a `GET` request to `http://localhost:8000/users/` to retrieve the list of users.

    | **Component** | **Value** |
    |--------------|----------|
    | Base URL | `http://localhost:8000` |
    | Router Prefix | `/users` |
    | Defined Path | `/` |
    | Final Endpoint URL | `http://localhost:8000/users/` |

- **Including the Router:**  
  Finally, `app.include_router(router)` mounts the router’s endpoints onto the FastAPI app.




### Including Routers

Routers can also be nested. This is useful for grouping endpoints by functionality.

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Internal router (could be used for versioning or grouping)
internal_router = APIRouter(prefix="/v1", tags=["v1"])

# Create a users router
users_router = APIRouter(prefix="/users", tags=["users"])

@users_router.get("/")
def get_users():
    return [{"name": "Rick"}, {"name": "Morty"}]

# Include the users_router inside the internal_router
internal_router.include_router(users_router)

# Mount the internal_router into the main app
app.include_router(internal_router)
```

**Explanation:**

- **Nesting Routers:**  
  The `users_router` is included in `internal_router` by calling `internal_router.include_router(users_router)`. Then the internal router is added to the main app. This structure allows you to build complex APIs with many layers of grouping.

    | **Component**       | **Value**                           |
    |---------------------|-----------------------------------|
    | **Base URL**        | `http://localhost:8000`           |
    | **Internal Router Prefix** | `/v1`                    |
    | **Users Router Prefix**    | `/users`                 |
    | **Defined Path in Users Router** | `/`               |
    | **Final Endpoint URL** | `http://localhost:8000/v1/users/` |

### Using WebSockets with APIRouter

APIRouter also supports WebSocket endpoints.

```python
from fastapi import FastAPI, APIRouter, WebSocket

app = FastAPI()
router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")

app.include_router(router)
```

**Explanation:**

- **WebSocket Route:**  
  The `@router.websocket("/ws")` decorator registers a WebSocket endpoint. When a client connects, the server accepts the connection and enters a loop to echo messages back to the client.

- **Router Inclusion:**  
  As with HTTP endpoints, the router is added to the main app using `app.include_router(router)`.

---

### Versioned API with APIRouter

#### Scenario:
You want to create a versioned API where different versions of the same endpoint are accessible under different paths (e.g., `/v1/users` and `/v2/users`).

#### Code:

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Create a router for version 1
v1_router = APIRouter(prefix="/v1", tags=["v1"])

@v1_router.get("/users")
def get_users_v1():
    """
    Returns a list of users for version 1.
    """
    return [{"username": "Rick"}, {"username": "Morty"}]

# Create a router for version 2
v2_router = APIRouter(prefix="/v2", tags=["v2"])

@v2_router.get("/users")
def get_users_v2():
    """
    Returns a list of users for version 2.
    """
    return [{"username": "Rick"}, {"username": "Morty"}, {"username": "Summer"}]

# Include both routers in the main app
app.include_router(v1_router)
app.include_router(v2_router)
```

**Explanation:**

- **Versioning:**  
  Two routers are created, `v1_router` and `v2_router`, each with its own prefix (`/v1` and `/v2`). This allows you to maintain different versions of the same endpoint under different paths.

- **Endpoint Definition:**  
  Each router has a `/users` endpoint, but the implementation differs slightly between versions. This is useful for maintaining backward compatibility while introducing new features.

- **Router Inclusion:**  
  Both routers are included in the main FastAPI app, making the endpoints accessible under `/v1/users` and `/v2/users`.

    | **Component**       | **Version** | **Prefix** | **Defined Path** | **Final Endpoint URL** |
    |---------------------|------------|------------|------------------|------------------------|
    | **Base URL**        | -          | -          | -                | `http://localhost:8000` |
    | **Version 1 Router** | v1         | `/v1`      | `/users`         | `http://localhost:8000/v1/users` |
    | **Version 2 Router** | v2         | `/v2`      | `/users`         | `http://localhost:8000/v2/users` |

---

### API with Multiple Tags and Descriptions

#### Scenario:
You want to create an API where endpoints are grouped by functionality and have detailed descriptions for better documentation.

#### Code:

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Create a router for user-related endpoints
user_router = APIRouter(prefix="/users", tags=["Users"])

@user_router.get("/", summary="Get all users", description="Returns a list of all users in the system.")
def get_all_users():
    return [{"username": "Rick"}, {"username": "Morty"}]

@user_router.post("/", summary="Create a new user", description="Creates a new user with the provided details.")
def create_user():
    return {"message": "User created"}

# Create a router for product-related endpoints
product_router = APIRouter(prefix="/products", tags=["Products"])

@product_router.get("/", summary="Get all products", description="Returns a list of all products in the system.")
def get_all_products():
    return [{"product": "Portal Gun"}, {"product": "Plumbus"}]

@product_router.post("/", summary="Create a new product", description="Creates a new product with the provided details.")
def create_product():
    return {"message": "Product created"}

# Include both routers in the main app
app.include_router(user_router)
app.include_router(product_router)
```

**Explanation:**

- **Tagging and Descriptions:**  
  Each router is tagged with a specific group (`Users` and `Products`), and each endpoint has a `summary` and `description`. This makes the API documentation more organized and informative.

- **Endpoint Definition:**  
  The `user_router` handles user-related endpoints, while the `product_router` handles product-related endpoints. This separation makes the API easier to navigate and maintain.

- **Router Inclusion:**  
  Both routers are included in the main FastAPI app, making the endpoints accessible under `/users` and `/products`.

    | **Component**           | **HTTP Method** | **Prefix**    | **Defined Path** | **Final Endpoint URL**                       | **Description**                         |
    |-------------------------|---------------|--------------|----------------|-----------------------------------------------|-----------------------------------------|
    | **Base URL**            | -             | -            | -              | `http://localhost:8000`                      | Root of the FastAPI application.       |
    | **User Router**         | `GET`         | `/users`     | `/`            | `http://localhost:8000/users/`               | Get all users.                         |
    | **User Router**         | `POST`        | `/users`     | `/`            | `http://localhost:8000/users/`               | Create a new user.                     |
    | **Product Router**      | `GET`         | `/products`  | `/`            | `http://localhost:8000/products/`            | Get all products.                      |
    | **Product Router**      | `POST`        | `/products`  | `/`            | `http://localhost:8000/products/`            | Create a new product.                  |

---

### API with Custom Response Models and Status Codes

#### Scenario:
You want to create an API where endpoints return custom response models and use specific HTTP status codes.

#### Code:

```python
from fastapi import FastAPI, APIRouter, status
from pydantic import BaseModel

app = FastAPI()

# Define a Pydantic model for the response
class UserResponse(BaseModel):
    username: str
    email: str

# Create a router for user-related endpoints
user_router = APIRouter(prefix="/users", tags=["Users"])

@user_router.get("/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
def get_user(user_id: int):
    """
    Returns the details of a specific user.
    """
    # Simulate fetching a user from a database
    return {"username": "Rick", "email": "rick@example.com"}

@user_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user():
    """
    Creates a new user and returns the user details.
    """
    # Simulate creating a user in a database
    return {"username": "Morty", "email": "morty@example.com"}

# Include the router in the main app
app.include_router(user_router)
```

**Explanation:**

- **Custom Response Models:**  
  The `UserResponse` Pydantic model is used to define the structure of the response for both the `GET` and `POST` endpoints. This ensures that the response is validated and documented correctly.

- **Custom Status Codes:**  
  The `status_code` parameter is used to specify the HTTP status code for each endpoint. The `GET` endpoint returns a `200 OK` status, while the `POST` endpoint returns a `201 CREATED` status.

- **Endpoint Definition:**  
  The `get_user` endpoint fetches a user by their ID, while the `create_user` endpoint simulates creating a new user. Both endpoints return a `UserResponse` object.

- **Router Inclusion:**  
  The `user_router` is included in the main FastAPI app, making the endpoints accessible under `/users`.

    | **Component**       | **HTTP Method** | **Prefix**   | **Defined Path**  | **Final Endpoint URL**                      | **Status Code** | **Description**                         |
    |---------------------|---------------|--------------|------------------|----------------------------------------------|---------------|-----------------------------------------|
    | **Base URL**        | -             | -            | -                | `http://localhost:8000`                     | -             | Root of the FastAPI application.       |
    | **User Router**     | `GET`         | `/users`     | `/{user_id}`     | `http://localhost:8000/users/{user_id}`     | `200 OK`      | Get details of a specific user.        |
    | **User Router**     | `POST`        | `/users`     | `/`              | `http://localhost:8000/users/`              | `201 Created` | Create a new user and return details.  |

---

## Advanced Usage and Considerations

### Global Dependencies and Middleware

You can add global dependencies or event handlers to a router. For example, you might add an authentication dependency that applies to all endpoints in a given router:

```python
from fastapi import Depends, HTTPException, status

def verify_token(token: str = "default-token"):
    if token != "expected-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

auth_router = APIRouter(
    prefix="/secure",
    dependencies=[Depends(verify_token)]
)

@auth_router.get("/data")
def get_secure_data():
    return {"data": "This is secure data"}

app.include_router(auth_router)
```

**Explanation:**

- **Global Dependency:**  
  The dependency `verify_token` is applied to every endpoint within the `auth_router`. Clients must provide the expected token, or they will receive a 401 error.

### Customizing OpenAPI Operation IDs

You can customize how operation IDs are generated by providing a custom function to the `generate_unique_id_function` parameter. This is especially useful when you are automatically generating client SDKs.

---

## Conclusion

The FastAPI `APIRouter` class is a versatile tool that allows you to build, structure, and maintain large APIs efficiently. By grouping endpoints, setting global configurations, and even nesting routers, you can keep your code organized and maintainable. Whether you’re creating a simple microservice or a complex, multi-module application, understanding how to use `APIRouter` is key to harnessing the full power of FastAPI.

We hope this guide has provided a clear explanation, practical examples, and useful tables to help you get started with FastAPI’s APIRouter. Happy coding!

---