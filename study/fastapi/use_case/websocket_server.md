# Mastering `fastapi.WebSocket`: A Comprehensive Guide with Standalone Examples

WebSockets enable real-time, bidirectional communication between a client and a server. FastAPI, a modern, high-performance web framework for Python 3.6+ based on standard type hints, provides robust support for WebSocket endpoints. In this guide, every example is self-contained, so you can experiment with each one independently.

## Table of Contents

1. [Introduction](#introduction)
2. [How to Run the Server](#how-to-run-the-server)
3. [Example 1: Basic Echo Server](#example-1-basic-echo-server)
4. [Example 2: Chat Room with Broadcasting](#example-2-chat-room-with-broadcasting)
5. [Example 3: Securing WebSocket Connections](#example-3-securing-websocket-connections)
6. [Example 4: Streaming Response with WebSocket](#example-4-streaming-response-with-websocket)
7. [Example 5: Detailed Method Usage Examples](#example-5-detailed-method-usage-examples)
    - [5.1 Text Data: `receive_text()` and `send_text()`](#51-text-data-receive_text-and-send_text)
    - [5.2 Bytes Data: `receive_bytes()` and `send_bytes()`](#52-bytes-data-receive_bytes-and-send_bytes)
    - [5.3 JSON Data: `receive_json()` and `send_json()`](#53-json-data-receive_json-and-send_json)
    - [5.4 Low-Level Methods: `receive()` and `send()`](#54-low-level-methods-receive-and-send)
8. [HTTP vs. WebSocket Endpoints](#http-vs-websocket-endpoints)
9. [Best Practices and Tips](#best-practices-and-tips)
10. [Conclusion](#conclusion)
11. [References](#references)

## Introduction

WebSockets allow persistent, bidirectional communication between clients and servers. Unlike traditional HTTP—where each request gets a new connection—WebSockets keep the connection open until it is explicitly closed. This makes them ideal for real-time applications such as chats, live notifications, and streaming data.

FastAPI’s built-in support for WebSockets makes it simple to create these applications using Python’s async features.

## How to Run the Server

For each example below:

1. **Save the code into a separate file.**  
   For instance, save the echo server example as `echo_server.py`.
2. **Run the server using Uvicorn.**  
   Open a terminal and run:
   ```bash
   uvicorn echo_server:app --reload
   ```
   Replace `echo_server` with the name of your file (without the `.py` extension).

---

## Example 1: Basic Echo Server

This simple example receives text messages and sends them back prefixed with `"Echo:"`.

**File:** `echo_server.py`

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/echo")
async def websocket_echo(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive a text message from the client
        data = await websocket.receive_text()
        # Send the text message back with an "Echo:" prefix
        await websocket.send_text(f"Echo: {data}")

# To run: uvicorn echo_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets

async def echo_client():
    uri = "ws://localhost:8000/ws/echo"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, FastAPI!")
        response = await websocket.recv()
        print(f"Response received: {response}")

asyncio.run(echo_client())
```

---

## Example 2: Chat Room with Broadcasting

This standalone example demonstrates a chat server where every connected client receives messages sent by any client.

**File:** `chat_server.py`

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive a message and broadcast it to all clients
            data = await websocket.receive_text()
            await manager.broadcast(f"Client says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("A client left the chat")

# To run: uvicorn chat_server:app --reload
```

---

## Example 3: Securing WebSocket Connections

This example accepts a connection only if a valid token is provided as a query parameter.

**File:** `secure_server.py`

```python
from fastapi import FastAPI, WebSocket, Query, HTTPException, WebSocketDisconnect

app = FastAPI()

async def get_current_user(token: str = Query(...)):
    if token != "secret-token":
        raise HTTPException(status_code=400, detail="Invalid token")
    return {"user": "authenticated_user"}

@app.websocket("/ws/secure")
async def secure_endpoint(websocket: WebSocket, token: str = Query(...)):
    try:
        user = await get_current_user(token)
    except HTTPException:
        await websocket.close(code=1008)  # Policy Violation
        return

    await websocket.accept()
    await websocket.send_text(f"Hello, {user['user']}! Welcome to the secure WebSocket.")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"You said: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")

# To run: uvicorn secure_server:app --reload
```

---

## Example 4: Streaming Response with WebSocket

This server sends a series of messages (one per second) to the client, simulating a streaming response.

**File:** `stream_server.py`

```python
import asyncio
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        for i in range(10):
            await websocket.send_text(f"Streaming message {i+1}")
            await asyncio.sleep(1)
    except Exception as e:
        print("Error during streaming:", e)
    finally:
        await websocket.close()

# To run: uvicorn stream_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets

async def stream_client():
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as websocket:
        try:
            while True:
                message = await websocket.recv()
                print(f"Streamed response: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("Stream ended.")

asyncio.run(stream_client())
```

---

## Example 5: Detailed Method Usage Examples

Each of the following examples is a standalone script demonstrating specific WebSocket methods.

### 5.1 Text Data: `receive_text()` and `send_text()`

**File:** `methods_text_server.py`

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/methods_text")
async def methods_text(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        print("Received text:", message)
        await websocket.send_text(f"Server received: {message}")

# To run: uvicorn methods_text_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets

async def text_client():
    uri = "ws://localhost:8000/ws/methods_text"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, text endpoint!")
        response = await websocket.recv()
        print("Text endpoint response:", response)

asyncio.run(text_client())
```

---

### 5.2 Bytes Data: `receive_bytes()` and `send_bytes()`

**File:** `methods_bytes_server.py`

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/methods_bytes")
async def methods_bytes(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        print("Received bytes:", data)
        await websocket.send_bytes(b"Server received: " + data)

# To run: uvicorn methods_bytes_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets

async def bytes_client():
    uri = "ws://localhost:8000/ws/methods_bytes"
    async with websockets.connect(uri) as websocket:
        await websocket.send(b"Hello, bytes endpoint!")
        response = await websocket.recv()
        print("Bytes endpoint response:", response)

asyncio.run(bytes_client())
```

---

### 5.3 JSON Data: `receive_json()` and `send_json()`

**File:** `methods_json_server.py`

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/methods_json")
async def methods_json(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        print("Received JSON:", data)
        response = {"message": "Server received your JSON", "data": data}
        await websocket.send_json(response)

# To run: uvicorn methods_json_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets
import json

async def json_client():
    uri = "ws://localhost:8000/ws/methods_json"
    async with websockets.connect(uri) as websocket:
        message = {"greeting": "Hello, JSON endpoint!"}
        await websocket.send(json.dumps(message))
        response = await websocket.recv()
        print("JSON endpoint response:", response)

asyncio.run(json_client())
```

---

### 5.4 Low-Level Methods: `receive()` and `send()`

This example shows how to use the lower-level methods to work with raw WebSocket messages.

**File:** `methods_raw_server.py`

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/methods_raw")
async def methods_raw(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive raw data (a dict containing either "text" or "bytes")
        message = await websocket.receive()
        print("Received raw data:", message)
        
        if "text" in message:
            response = {"type": "websocket.send", "text": f"Raw echo: {message['text']}"}
        elif "bytes" in message:
            response = {"type": "websocket.send", "bytes": b"Raw echo: " + message["bytes"]}
        else:
            response = {"type": "websocket.send", "text": "Unknown message type."}
        
        await websocket.send(response)

# To run: uvicorn methods_raw_server:app --reload
```

> **Note:** Although the low-level `receive()` and `send()` methods provide extra flexibility, the convenience methods (such as `receive_text()` and `send_text()`) are generally recommended.

---

## HTTP vs. WebSocket Endpoints

| **Feature**             | **HTTP Endpoints**                          | **WebSocket Endpoints**                             |
|-------------------------|---------------------------------------------|-----------------------------------------------------|
| **Protocol**            | Request/Response over HTTP                  | Full-duplex communication over TCP via WebSocket    |
| **Connection Lifespan** | Short-lived (one request, one response)     | Long-lived (persistent connection)                |
| **Communication Model** | Client sends a request and awaits a response| Both client and server can send messages at any time|
| **State Management**    | Typically stateless                         | Can maintain state for the duration of the connection|
| **Use Cases**           | CRUD operations, REST APIs                  | Chat apps, live updates, real-time notifications    |

---

## Best Practices and Tips

- **Validate Incoming Data:** Always check and sanitize the data you receive.
- **Graceful Disconnects:** Use exception handling (e.g., catching `WebSocketDisconnect`) to handle disconnections.
- **Connection Management:** In multi-client applications, manage connections (e.g., using a dedicated connection manager).
- **Authentication:** Secure endpoints by verifying tokens or credentials.
- **Async Programming:** Use Python’s async/await features to avoid blocking operations.
- **Error Handling:** Provide robust error handling for unexpected client behavior.
- **Resource Cleanup:** Ensure disconnected clients are removed to prevent memory leaks.

## Conclusion

FastAPI’s WebSocket support opens the door to building real-time applications—from simple echo servers to sophisticated chat rooms and streaming endpoints. Each example in this guide is a standalone script that you can run and test independently. Save the code in individual files and run them using Uvicorn (e.g., `uvicorn filename:app --reload`). Use the provided client examples or your favorite WebSocket client to interact with these endpoints.

Happy coding with FastAPI and WebSockets!

## References

- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [MDN WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Python websockets Library](https://pypi.org/project/websockets/)
- [Introduction to WebSockets](https://www.websocket.org/aboutwebsocket.html)
