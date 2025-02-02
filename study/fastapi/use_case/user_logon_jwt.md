# Implementing User Login and Logout with FastAPI and JWT: A Step-by-Step Guide

This article introduces a complete example of building a user authentication system using FastAPI. We demonstrate how to design **login** and **logout** endpoints using JSON Web Tokens (JWT) for authentication. The example includes a fake user database, token creation, security dependencies, and protected endpoints. We’ll explain the code step by step with detailed explanations and tables to help you understand each component.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Configuration & Application Setup](#configuration--application-setup)
3. [Fake User Database & Utility Functions](#fake-user-database--utility-functions)
4. [JWT Token Creation & Data Models](#jwt-token-creation--data-models)
5. [Security Dependencies](#security-dependencies)
6. [Login Endpoint](#login-endpoint)
7. [Logout Endpoint](#logout-endpoint)
8. [Protected Endpoint Example](#protected-endpoint-example)
9. [Conclusion](#conclusion)

---

## Introduction

In this guide, we build a simple user authentication system with FastAPI that supports:
- **Login:** Verifying credentials and issuing a JWT token.
- **Logout:** Revoking tokens via an in-memory blacklist.
- **Protected Endpoints:** Accessing resources only with a valid, non-revoked token.

By the end of this guide, you will understand how to create secure endpoints with FastAPI using JWT and manage user sessions through login and logout functionality.

---

## Configuration & Application Setup

At the beginning of our code, we import necessary modules and define configuration variables.

```python
from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from pydantic import BaseModel
```

### Key Configuration Variables

| Variable                           | Description                                                                                              |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `SECRET_KEY`                       | A secret key used to sign JWT tokens. In production, generate a strong key (e.g., using `openssl rand -hex 32`). |
| `ALGORITHM`                        | The algorithm used for encoding the JWT token (here, HS256).                                             |
| `ACCESS_TOKEN_EXPIRE_MINUTES`      | Token expiration time in minutes (here, set to 30 minutes).                                              |

Next, we instantiate the FastAPI application:

```python
app = FastAPI()
```

---

## Fake User Database & Utility Functions

For demonstration, we simulate a user database with a Python dictionary. In a real application, you would use a proper database.

```python
fake_users_db: Dict[str, Dict[str, Any]] = {
    "john": {
        "username": "john",
        "full_name": "John Doe",
        "email": "john@example.com",
        "hashed_password": "fakehashedsecret",  # For demo purposes only!
        "disabled": False,
    }
}
```

### Utility Functions and User Models

We define functions to hash passwords, retrieve users, and authenticate credentials. We also define Pydantic models for user data.

```python
def fake_hash_password(password: str) -> str:
    """
    In a real application, you would hash passwords using a secure method.
    Here, we just simulate it.
    """
    return "fakehashed" + password

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

def get_user(db: Dict[str, Dict[str, Any]], username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db: Dict[str, Dict[str, Any]], username: str, password: str) -> Optional[UserInDB]:
    """
    Verifies a user's credentials. Returns the user if authentication succeeds.
    """
    user = get_user(db, username)
    if not user:
        return None
    if user.hashed_password != fake_hash_password(password):
        return None
    return user
```

#### Explanation

- **`fake_hash_password`**: Simulates password hashing. (In production, use a secure hashing algorithm.)
- **`User` and `UserInDB`**: Pydantic models to represent user data. The `UserInDB` model includes the hashed password.
- **`get_user` and `authenticate_user`**: Retrieve a user from the fake database and check credentials.

---

## JWT Token Creation & Data Models

To secure our endpoints, we use JWT tokens. The following code defines the models for tokens and a function to create them.

```python
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT token with an expiration time.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

#### Explanation

- **`Token`**: Pydantic model that defines the structure of the response returned upon successful login.
- **`TokenData`**: Model used for decoding token data.
- **`create_access_token`**: Generates a JWT token with an expiration claim (`exp`).

---

## Security Dependencies

Security dependencies ensure that only authenticated users can access protected endpoints.

### OAuth2PasswordBearer

This dependency tells FastAPI where to expect the JWT token in the request.

```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
```

A note on requests and responses with tokens:

- **Request Header:**

  ```
  GET /users/me HTTP/1.1
  Authorization: Bearer <your-jwt-token>
  ```

- **Response (example):**

  ```json
  {
    "token": "<your-jwt-token>"
  }
  ```

### Token Blacklisting

For this demo, we use an in-memory set to blacklist (revoke) tokens upon logout.

```python
blacklisted_tokens = set()
```

### User Retrieval Dependencies

We create two dependencies: one to get the current user from the token, and another to ensure the user is active.

```python
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Dependency that decodes the JWT token and retrieves the current user.
    Also checks whether the token has been revoked (blacklisted).
    """
    if token in blacklisted_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency that ensures the current user is active.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

#### Explanation

- **`get_current_user`**:  
  - Checks if the token is blacklisted.
  - Decodes the token and extracts the username.
  - Retrieves the user from the database.
- **`get_current_active_user`**:  
  - Further checks that the user is not disabled.

---

## Login Endpoint

The login endpoint verifies user credentials and returns a JWT token if authentication is successful.

```python
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint:
      - Receives the username and password via form data.
      - Authenticates the user.
      - Returns a JWT token if credentials are valid.
    """
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}
```

### How It Works

- **Input:**  
  The endpoint expects form data containing `username` and `password`.
  
- **Authentication:**  
  It uses `authenticate_user` to verify credentials.

- **Token Creation:**  
  If authentication succeeds, a JWT token is created with an expiration time.

- **Response:**  
  Returns the token and its type in JSON format.

#### Example `curl` Request

```bash
curl -X POST "http://127.0.0.1:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=john&password=secret"
```

#### Example Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

## Logout Endpoint

The logout endpoint "revokes" the JWT token by adding it to an in-memory blacklist. After logging out, the token cannot be used to access protected endpoints.

```python
@app.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """
    Logout endpoint:
      - “Revokes” the token by adding it to an in-memory blacklist.
      - Instructs the client to delete the token.
    """
    blacklisted_tokens.add(token)
    return {"message": "Successfully logged out"}
```

### How It Works

- **Token Dependency:**  
  The endpoint retrieves the token using the same `oauth2_scheme`.

- **Token Revocation:**  
  The token is added to the `blacklisted_tokens` set.

- **Response:**  
  Returns a message confirming logout.

#### Example `curl` Request

```bash
curl -X POST "http://127.0.0.1:8000/logout" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### Example Response

```json
{
  "message": "Successfully logged out"
}
```

---

## Protected Endpoint Example

This endpoint demonstrates a protected resource that only an authenticated and active user can access.

```python
@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    A protected endpoint that returns the current logged-in user's data.
      - Uses the 'get_current_active_user' dependency which also checks the blacklist.
    """
    return current_user
```

### How It Works

- **Dependency:**  
  The endpoint uses `get_current_active_user` to ensure that the request includes a valid, non-revoked token.
  
- **Response:**  
  Returns the current user's data as JSON.

#### Example `curl` Request

```bash
curl -X GET "http://127.0.0.1:8000/users/me" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### Example Response

```json
{
  "username": "john",
  "email": "john@example.com",
  "full_name": "John Doe",
  "disabled": false
}
```

---

## Conclusion

This guide provided a comprehensive walkthrough of implementing user login and logout functionality using FastAPI and JWT. We covered:

- **Configuration:** Setting up secret keys, algorithms, and token expiration.
- **User Management:** Using a fake database and utility functions to authenticate users.
- **JWT Creation:** Building tokens with expiration using the `jose` library.
- **Security Dependencies:** Ensuring endpoints are protected and tokens are validated.
- **Endpoints:** Implementing login, logout, and protected endpoints with clear examples and `curl` commands.

This example serves as a solid foundation for building a secure authentication system in your FastAPI applications. With further enhancements—such as secure password hashing, persistent token blacklisting (using a database or Redis), and more refined user management—you can tailor this solution to meet production-grade requirements.
