# Building Real-Time Applications with FastAPI: A WebSocket Tutorial

WebSockets enable real-time, bidirectional communication between a client and a server. FastAPI, a modern, high-performance web framework for Python 3.6+ based on standard type hints, provides robust support for WebSocket endpoints. In this guide, every example is self-contained, so you can experiment with each one independently.

## Table of Contents

1. [Introduction](#introduction)
2. [How to Run the Server](#how-to-run-the-server)
3. [Example 1: Basic Echo Server](#example-1-basic-echo-server)
4. [Example 2: Chat Room with Broadcasting](#example-2-chat-room-with-broadcasting)
5. [Example 3: Securing WebSocket Connections](#example-3-securing-websocket-connections)
6. [Example 4: Streaming Response with WebSocket](#example-4-streaming-response-with-websocket)
7. [Example 5: Real-Time Chat Room](#example-5-real-time-chat-room)
8. [Example 6: Secure Notifications System](#example-6-secure-notifications-system)
9. [Example 7: Live Data Streaming](#example-7-live-data-streaming)
10. [HTTP vs. WebSocket Endpoints](#http-vs-websocket-endpoints)
11. [Best Practices and Tips](#best-practices-and-tips)
12. [Conclusion](#conclusion)
13. [References](#references)

---

## Introduction

WebSockets allow persistent, bidirectional communication between clients and servers. Unlike traditional HTTP—where each request gets a new connection—WebSockets keep the connection open until it is explicitly closed. This makes them ideal for real-time applications such as chats, live notifications, and streaming data.

FastAPI’s built-in support for WebSockets makes it simple to create these applications using Python’s async features.

---

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

# Create a FastAPI application instance.
app = FastAPI()

@app.websocket("/ws/echo")
async def websocket_echo(websocket: WebSocket):
    # Accept the WebSocket connection from the client.
    await websocket.accept()
    while True:
        # Receive a text message from the client.
        data = await websocket.receive_text()
        # Send the received message back to the client with an "Echo:" prefix.
        await websocket.send_text(f"Echo: {data}")

# To run: uvicorn echo_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets

async def echo_client():
    # Define the WebSocket server URI.
    uri = "ws://localhost:8000/ws/echo"
    async with websockets.connect(uri) as websocket:
        # Send a message to the server.
        await websocket.send("Hello, FastAPI!")
        # Receive the echoed response from the server.
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

# Create a FastAPI application instance.
app = FastAPI()

# Define a connection manager to handle active WebSocket connections.
class ConnectionManager:
    def __init__(self):
        # List to store active WebSocket connections.
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        # Accept a new WebSocket connection and add it to the active connections list.
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        # Remove a WebSocket connection from the active connections list.
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        # Send a message to all active WebSocket connections.
        for connection in self.active_connections:
            await connection.send_text(message)

# Create an instance of the ConnectionManager.
manager = ConnectionManager()

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    # Connect the client to the chat room.
    await manager.connect(websocket)
    try:
        while True:
            # Receive a message from the client.
            data = await websocket.receive_text()
            # Broadcast the received message to all connected clients.
            await manager.broadcast(f"Client says: {data}")
    except WebSocketDisconnect:
        # Handle client disconnection and notify other clients.
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

# Create a FastAPI application instance.
app = FastAPI()

# Function to validate the token provided by the client.
async def get_current_user(token: str = Query(...)):
    if token != "secret-token":
        # Raise an HTTPException if the token is invalid.
        raise HTTPException(status_code=400, detail="Invalid token")
    return {"user": "authenticated_user"}

@app.websocket("/ws/secure")
async def secure_endpoint(websocket: WebSocket, token: str = Query(...)):
    try:
        # Validate the token before accepting the connection.
        user = await get_current_user(token)
    except HTTPException:
        # Close the connection if the token is invalid.
        await websocket.close(code=1008)  # Policy Violation
        return

    # Accept the WebSocket connection.
    await websocket.accept()
    # Send a welcome message to the authenticated client.
    await websocket.send_text(f"Hello, {user['user']}! Welcome to the secure WebSocket.")
    try:
        while True:
            # Receive a message from the client and echo it back.
            data = await websocket.receive_text()
            await websocket.send_text(f"You said: {data}")
    except WebSocketDisconnect:
        # Handle client disconnection.
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

# Create a FastAPI application instance.
app = FastAPI()

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        # Simulate a streaming response by sending 10 messages with a 1-second delay.
        for i in range(10):
            await websocket.send_text(f"Streaming message {i+1}")
            await asyncio.sleep(1)
    except Exception as e:
        # Handle any errors during streaming.
        print("Error during streaming:", e)
    finally:
        # Close the WebSocket connection after streaming is complete.
        await websocket.close()

# To run: uvicorn stream_server:app --reload
```

**Client Example (Optional):**

```python
import asyncio
import websockets

async def stream_client():
    # Define the WebSocket server URI.
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as websocket:
        try:
            while True:
                # Receive streaming messages from the server.
                message = await websocket.recv()
                print(f"Streamed response: {message}")
        except websockets.exceptions.ConnectionClosed:
            # Handle the end of the stream.
            print("Stream ended.")

asyncio.run(stream_client())
```

---

## Example 5: Real-Time Chat Room

This example is similar to the chat room example but uses a distinct endpoint and file name for differentiation.

**File:** `chat_server_additional.py`

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

# Create a FastAPI application instance.
app = FastAPI()

# Define a connection manager to handle active WebSocket connections.
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

@app.websocket("/ws/chat_additional")
async def chat_endpoint_additional(websocket: WebSocket):
    # Connect the client to the chat room.
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"(Additional) Client says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("(Additional) A client left the chat.")

# To run: uvicorn chat_server_additional:app --reload
```

---

## Example 6: Secure Notifications System

This example secures a WebSocket endpoint for sending notifications only to authenticated clients. Note the endpoint is `/ws/notify` and the file name is different.

**File:** `secure_notifications.py`

```python
from fastapi import FastAPI, WebSocket, Query, HTTPException, WebSocketDisconnect

# Create a FastAPI application instance.
app = FastAPI()

# Function to verify the token.
async def get_current_user(token: str = Query(...)):
    if token != "secret-token":
        raise HTTPException(status_code=400, detail="Invalid token")
    return {"user": "authenticated_user"}

@app.websocket("/ws/notify")
async def secure_notify(websocket: WebSocket, token: str = Query(...)):
    try:
        user = await get_current_user(token)
    except HTTPException:
        await websocket.close(code=1008)  # Close connection due to policy violation.
        return

    await websocket.accept()
    await websocket.send_text(f"Hello, {user['user']}! You are now connected to secure notifications.")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Notification received: {data}")
    except WebSocketDisconnect:
        print("Client disconnected from secure notifications.")

# To run: uvicorn secure_notifications:app --reload
```

---

## Example 7: Live Data Streaming

This example simulates live data streaming (such as progress updates) by sending periodic progress messages.

**File:** `streaming_server_progress.py`

```python
import asyncio
from fastapi import FastAPI, WebSocket

# Create a FastAPI application instance.
app = FastAPI()

@app.websocket("/ws/stream_progress")
async def stream_progress(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        # Simulate progress updates for 10 seconds.
        for i in range(10):
            await websocket.send_text(f"Progress update: {(i+1)*10}% complete")
            await asyncio.sleep(1)
    except Exception as e:
        print("Streaming error:", e)
    finally:
        await websocket.close()

# To run: uvicorn streaming_server_progress:app --reload
```

---

## HTTP vs. WebSocket Endpoints

| **Feature**             | **HTTP Endpoints**                          | **WebSocket Endpoints**                             |
|-------------------------|---------------------------------------------|-----------------------------------------------------|
| **Protocol**            | Request/Response over HTTP                  | Full-duplex communication over TCP via WebSocket    |
| **Connection Lifespan** | Short-lived (one request, one response)     | Long-lived (persistent connection)                |
| **Communication Model** | Client sends a request and awaits a response | Both client and server can send messages at any time|
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

---

## Conclusion

FastAPI’s WebSocket support opens the door to building real-time applications—from simple echo servers to sophisticated chat rooms and streaming endpoints. Each example in this guide is a standalone script that you can run and test independently. Save the code in individual files and run them using Uvicorn (e.g., `uvicorn filename:app --reload`). Use the provided client examples or your favorite WebSocket client to interact with these endpoints.

---

## References

- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [MDN WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Python websockets Library](https://pypi.org/project/websockets/)
- [Introduction to WebSockets](https://www.websocket.org/aboutwebsocket.html)

