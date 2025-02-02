# Introduction to fastapi.WebSocket with Client Testing

FastAPI makes it straightforward to add real-time communication to your applications using WebSockets. Under the hood, FastAPI uses Starlette’s WebSocket implementation, which provides you with an easy-to-use API. In this article, we cover:

## Table of Contents

1. [What is a WebSocket?](#what-is-a-websocket)
2. [The fastapi.WebSocket Class](#the-fastapiwebsocket-class)
3. [API Reference Table](#api-reference-table)
4. [Examples](#examples)
    - [Example 1: Simple Echo using Text Messages (with graceful close)](#example-1-simple-echo-using-text-messages)
    - [Example 2: JSON Data Exchange](#example-2-json-data-exchange)
    - [Example 3: Echoing Binary Data](#example-3-echoing-binary-data)
    - [Example 4: Using Low-level receive and send Methods](#example-4-using-low-level-receive-and-send-methods)
    - [Example 5: Handling Client Disconnects](#example-5-handling-client-disconnects)
    - [Example 6: Broadcasting to Multiple Clients](#example-6-broadcasting-to-multiple-clients)
    - [Example 7: Custom Headers and Subprotocols](#example-7-custom-headers-and-subprotocols)
    - [Example 8: Iterating over JSON and Binary Messages](#example-8-iterating-over-json-and-binary-messages)
5. [Conclusion](#conclusion)

## What is a WebSocket?

A WebSocket is a protocol that provides a persistent, full-duplex communication channel between a client (for example, a browser or a Python script) and a server. Unlike HTTP—where communication is client-initiated and transient—WebSockets allow both the client and the server to send data at any time. This makes them ideal for real-time applications such as chat systems, live dashboards, and collaborative tools.

## The fastapi.WebSocket Class

In FastAPI, WebSocket endpoints are defined by declaring a parameter of type `fastapi.WebSocket`. This object provides methods to:

- **Receive Data:**
  - `receive()`: Low-level method to receive ASGI messages.
  - `receive_text()`: Receive a text message.
  - `receive_bytes()`: Receive a binary message.
  - `receive_json()`: Receive a JSON message.
- **Send Data:**
  - `send()`: Low-level method to send ASGI messages.
  - `send_text()`: Send a text message.
  - `send_bytes()`: Send a binary message.
  - `send_json()`: Send a JSON message.
- **Manage Connection State:**
  - `accept()`: Accept the WebSocket connection.
  - `close()`: Close the WebSocket connection gracefully.

FastAPI also supports defining dependencies that work with both HTTP and WebSocket connections by using `HTTPConnection` instead of `Request` or `WebSocket`.

## API Reference Table

Below is a summary of the main methods and properties available on a `fastapi.WebSocket` instance:

| **Method / Property** | **Type / Parameters** | **Description** |
| --------------------- | --------------------- | --------------- |
| **`WebSocket(scope, receive, send)`** | *scope*: `Scope`<br>*receive*: `Receive`<br>*send*: `Send` | Constructor used by the ASGI server. |
| **`scope`** | `Scope` | The connection scope (includes connection details). |
| **`app`** | *Property* | The FastAPI application instance. |
| **`url`** | *Property* | The URL of the connection. |
| **`base_url`** | *Property* | The base URL. |
| **`headers`** | *Property* | HTTP headers from the connection. |
| **`query_params`** | *Property* | Query parameters from the URL. |
| **`path_params`** | *Property* | Path parameters from the URL. |
| **`cookies`** | *Property* | Cookies from the connection. |
| **`client`** | *Property* | Client address details. |
| **`state`** | *Property* | Application-specific state. |
| **`client_state`** | *Attribute* | Current client state (initially `CONNECTING`). |
| **`application_state`** | *Attribute* | Current application state (initially `CONNECTING`). |
| **`url_for(name, **path_params)`** | *name*: `str`<br>**path_params**: `Any` | Returns the URL for a named endpoint. |
| **`receive()`** | *Async Method* | Receives a raw ASGI WebSocket message. |
| **`send(message)`** | *Async Method*<br>*message*: `Message` | Sends a raw ASGI WebSocket message. |
| **`accept(subprotocol=None, headers=None)`** | *subprotocol*: `str \| None` (default: `None`) | Accepts the WebSocket connection. |
| **`receive_text()`** | *Async Method* | Receives a text message. |
| **`receive_bytes()`** | *Async Method* | Receives a binary message. |
| **`receive_json(mode='text')`** | *Async Method*<br>*mode*: `str` (default: `'text'`) | Receives a JSON message. |
| **`send_text(data)`** | *Async Method*<br>*data*: `str` | Sends a text message. |
| **`send_bytes(data)`** | *Async Method*<br>*data*: `bytes` | Sends a binary message. |
| **`send_json(data, mode='text')`** | *Async Method*<br>*data*: `Any`<br>*mode*: `str` (default: `'text'`) | Sends a JSON message. |
| **`close(code=1000, reason=None)`** | *Async Method*<br>*code*: `int` (default: `1000`)<br>*reason*: `str \| None` (default: `None`) | Closes the WebSocket connection gracefully. |

Additionally, FastAPI provides the `WebSocketDisconnect` exception for handling disconnections.

## Examples

The following examples illustrate how to use various methods available in `fastapi.WebSocket`. Each example is paired with a client snippet using the Python `websockets` library. Some examples also demonstrate how to close the connection gracefully using the `close()` method.

---

### Example 1: Simple Echo using Text Messages

This example creates an echo server that uses `receive_text()` and `send_text()`. It also shows how to close the connection gracefully when the client sends the message `"close"`.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/echo")
async def websocket_echo_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        while True:
            # Receive a text message from the client.
            data = await websocket.receive_text()
            # If the client sends "close", send a final message and close the connection gracefully.
            if data.lower() == "close":
                await websocket.send_text("Closing connection gracefully.")
                await websocket.close(code=1000, reason="Client requested closure")
                break
            # Otherwise, echo the received message back to the client.
            await websocket.send_text(f"Echo: {data}")
        
        # Close the connection after breaking the loop.
        await websocket.close(code=1000, reason="Client requested closure")
    except WebSocketDisconnect:
        # Handle client disconnection.
        print("Client disconnected")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets

async def test_echo():
    uri = "ws://localhost:8000/ws/echo"
    async with websockets.connect(uri) as websocket:
        # Send a message to the server.
        message = "Hello, WebSocket!"
        print(f"Sending: {message}")
        await websocket.send(message)
        
        # Receive the echoed response from the server.
        response = await websocket.recv()
        print(f"Received: {response}")
        
        # Send "close" to trigger graceful closure.
        close_message = "close"
        print(f"Sending: {close_message}")
        await websocket.send(close_message)
        
        # Receive the final response before the connection closes.
        final_response = await websocket.recv()
        print(f"Final response: {final_response}")

asyncio.run(test_echo())
```

---

### Example 2: JSON Data Exchange

This example demonstrates receiving JSON data with `receive_json()` and sending JSON responses with `send_json()`.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/json")
async def websocket_json_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        while True:
            # Receive JSON data from the client.
            data = await websocket.receive_json()
            
            # Process the data (reverse the string under the key 'message').
            message = data.get("message", "")
            response = {"response": message[::-1]}
            
            # Send the processed JSON response back to the client.
            await websocket.send_json(response)
    except WebSocketDisconnect:
        # Handle client disconnection.
        print("Client disconnected")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets
import json

async def test_json():
    uri = "ws://localhost:8000/ws/json"
    async with websockets.connect(uri) as websocket:
        # Prepare JSON data to send.
        data = {"message": "Hello, JSON!"}
        print(f"Sending: {data}")
        
        # Send the JSON data to the server.
        await websocket.send(json.dumps(data))
        
        # Receive the JSON response from the server.
        response = await websocket.recv()
        print(f"Received: {json.loads(response)}")

asyncio.run(test_json())
```

---

### Example 3: Echoing Binary Data

This example shows how to use `receive_bytes()` and `send_bytes()` to echo binary data.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/binary")
async def websocket_binary_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        while True:
            # Receive binary data from the client.
            data = await websocket.receive_bytes()
            # Echo the binary data back to the client.
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        # Handle client disconnection.
        print("Client disconnected")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets

async def test_binary():
    uri = "ws://localhost:8000/ws/binary"
    async with websockets.connect(uri) as websocket:
        # Prepare binary data to send.
        message = b"Binary data test"
        print(f"Sending binary data: {message}")
        # Send the binary data to the server.
        await websocket.send(message)
        # Receive the echoed binary data from the server.
        response = await websocket.recv()
        print(f"Received binary data: {response}")

asyncio.run(test_binary())
```

---

### Example 4: Using Low-level receive and send Methods

This example demonstrates how to use the low-level `receive()` and `send()` methods. The low-level `receive()` returns a dictionary containing either the `"text"` or `"bytes"` key. The corresponding `send()` method expects a dictionary with the message type and payload.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/generic")
async def websocket_generic_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        while True:
            # Use the low-level receive() method to get a message.
            message = await websocket.receive()
            
            # Check if the received message is text.
            if "text" in message:
                text = message["text"]
                # Prepare a response dictionary for text messages.
                response = {"type": "websocket.send", "text": f"Echo: {text}"}
                await websocket.send(response)
            
            # Check if the received message is binary.
            elif "bytes" in message:
                # Prepare a response dictionary for binary messages.
                response = {"type": "websocket.send", "bytes": message["bytes"]}
                await websocket.send(response)
    except WebSocketDisconnect:
        # Handle client disconnection.
        print("Client disconnected")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets

async def test_generic():
    uri = "ws://localhost:8000/ws/generic"
    async with websockets.connect(uri) as websocket:
        # Send a text message to the server.
        text_message = "Hello via generic method"
        print(f"Sending text: {text_message}")
        await websocket.send(text_message)
        
        # Receive the echoed text response from the server.
        text_response = await websocket.recv()
        print(f"Text response: {text_response}")

        # Send binary data to the server.
        binary_message = b"Hello binary via generic method"
        print(f"Sending binary: {binary_message}")
        await websocket.send(binary_message)

        # Receive the echoed binary response from the server.
        binary_response = await websocket.recv()
        print(f"Binary response: {binary_response}")

asyncio.run(test_generic())
```

---

### Example 5: Handling Client Disconnects

This example demonstrates graceful handling of client disconnects using the `WebSocketDisconnect` exception. The server logs the disconnect details.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/handle_disconnect")
async def websocket_disconnect_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    try:
        while True:
            # Receive a text message from the client.
            message = await websocket.receive_text()
            # Send a response back to the client.
            await websocket.send_text(f"Received: {message}")
    except WebSocketDisconnect as e:
        # Handle client disconnection and log the disconnect details.
        print(f"Client disconnected with code {e.code} and reason: {e.reason}")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets

async def test_disconnect():
    uri = "ws://localhost:8000/ws/handle_disconnect"
    async with websockets.connect(uri) as websocket:
        # Send a test message to the server.
        await websocket.send("Test disconnect")
        # Receive the response from the server.
        response = await websocket.recv()
        print(f"Received: {response}")
    # Exiting the context automatically closes the connection.

asyncio.run(test_disconnect())
```

---

### Example 6: Broadcasting to Multiple Clients

This example demonstrates a simple chat server that broadcasts messages to all connected clients. When a client disconnects, it is removed from the active connection list.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

# List to store active WebSocket connections.
active_connections: List[WebSocket] = []

# Function to broadcast a message to all connected clients.
async def broadcast_message(message: str):
    for connection in active_connections:
        await connection.send_text(message)

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()
    # Add the new connection to the list of active connections.
    active_connections.append(websocket)
    try:
        while True:
            # Receive a text message from the client.
            data = await websocket.receive_text()
            # Broadcast the received message to all connected clients.
            await broadcast_message(f"Client says: {data}")
    except WebSocketDisconnect:
        # Remove the disconnected client from the list of active connections.
        active_connections.remove(websocket)
        # Notify remaining clients about the disconnection.
        await broadcast_message("A client has disconnected.")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets

async def test_chat():
    uri = "ws://localhost:8000/ws/chat"
    async with websockets.connect(uri) as websocket:
        # Send a message to the chat server.
        await websocket.send("Hello everyone!")
        # Receive the broadcasted message from the server.
        response = await websocket.recv()
        print(f"Broadcast received: {response}")

asyncio.run(test_chat())
```

---

### Example 7: Custom Headers and Subprotocols

This example demonstrates how to accept a connection with custom headers and an optional subprotocol.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/custom")
async def websocket_custom_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection with a custom subprotocol and headers.
    await websocket.accept(subprotocol="custom-protocol", headers=[(b"X-Custom-Header", b"MyValue")])
    try:
        while True:
            # Receive a text message from the client.
            data = await websocket.receive_text()
            # Send a custom response back to the client.
            await websocket.send_text(f"Custom response: {data}")
    except WebSocketDisconnect:
        # Handle client disconnection.
        print("Client disconnected")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets

async def test_custom():
    uri = "ws://localhost:8000/ws/custom"
    # Connect to the WebSocket server with a requested subprotocol.
    async with websockets.connect(uri, subprotocols=["custom-protocol"]) as websocket:
        # Send a test message to the server.
        await websocket.send("Test custom connection")
        # Receive the response from the server.
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_custom())
```

---

### Example 8: Iterating over JSON and Binary Messages

This example demonstrates how to use the asynchronous iterators `iter_json()` and `iter_bytes()` to continuously process incoming JSON and binary messages in a single connection. In this example, the server first iterates over JSON messages until it receives a `"done"` signal, then switches to iterating over binary messages until it receives the binary signal `b"done"`, and finally closes the connection gracefully.

#### Server Code

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/iter")
async def websocket_iter_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection.
    await websocket.accept()

    # Inform the client to send JSON messages.
    await websocket.send_text("Send JSON messages now. Send 'done' to finish JSON iteration.")

    # Iterate over incoming JSON messages.
    async for data in websocket.iter_json():
        if data == "done":
            break
        # Echo the received JSON message back to the client.
        await websocket.send_text(f"JSON Echo: {data}")

    # Inform the client to send binary messages.
    await websocket.send_text("Send binary messages now. Send b'done' to finish binary iteration.")

    # Iterate over incoming binary messages.
    async for data in websocket.iter_bytes():
        if data == b"done":
            break
        # Echo the received binary message back to the client.
        await websocket.send_bytes(data)

    # Close the connection gracefully after iteration is complete.
    await websocket.close(code=1000, reason="Iteration complete")
```

#### Client Code Using `websockets`

```python
import asyncio
import websockets
import json

async def test_iter():
    uri = "ws://localhost:8000/ws/iter"
    async with websockets.connect(uri) as websocket:
        # Receive the prompt for JSON messages.
        prompt = await websocket.recv()
        print(f"Received: {prompt}")
        
        # Send a few JSON messages to the server.
        for msg in ["hello", "world"]:
            print(f"Sending JSON: {msg}")
            await websocket.send(json.dumps(msg))
            # Receive the echoed JSON response from the server.
            response = await websocket.recv()
            print(f"Received: {response}")

        # Signal the end of JSON messages.
        await websocket.send(json.dumps("done"))
        # Receive the prompt for binary messages.
        prompt2 = await websocket.recv()
        print(f"Received: {prompt2}")
        
        # Send a few binary messages to the server.
        for bmsg in [b'\x00\x01\x02', b'\x03\x04\x05']:
            print(f"Sending binary: {bmsg}")
            await websocket.send(bmsg)
            # Receive the echoed binary response from the server.
            response = await websocket.recv()
            print(f"Received binary: {response}")
        
        # Signal the end of binary messages.
        await websocket.send(b"done")
        # Wait for the connection to close.
        try:
            response = await websocket.recv()
            print(f"Received: {response}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")

asyncio.run(test_iter())
```

---

## Conclusion

FastAPI’s WebSocket support, built on top of Starlette, offers an intuitive API for handling real-time communication. With a wide array of methods such as `receive`, `send`, `receive_text`, `receive_bytes`, `receive_json`, `send_text`, `send_bytes`, `send_json`, as well as asynchronous iterators like `iter_json()` and `iter_bytes()`, you can easily implement interactive applications—from echo servers to chat rooms and custom protocol handling. Additionally, the ability to gracefully close connections using `close()` helps ensure robust and clean shutdowns of WebSocket sessions.

By following the examples provided—and testing them with the Python `websockets` library—you can quickly set up endpoints for text, binary, and JSON data exchange, as well as experiment with low-level message handling and custom connection parameters.
