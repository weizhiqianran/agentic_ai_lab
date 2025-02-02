# Understanding Response in FastAPI

FastAPI provides a robust set of tools to craft outgoing HTTP responses, giving you full control over status codes, headers, cookies, media types, and even streamed content. In this article, we focus exclusively on the **Response** aspects of FastAPI. You'll learn how to tailor your responses to meet your application's requirements using various built-in response objects and customization techniques.

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Response Object in FastAPI](#the-response-object-in-fastapi)
   - [Example 1: Setting a Custom Status Code](#example-1-setting-a-custom-status-code)
   - [Example 2: Setting Response Headers](#example-2-setting-response-headers)
   - [Example 3: Setting Cookies](#example-3-setting-cookies)
   - [Example 4: Deleting Cookies](#example-4-deleting-cookies)
   - [Example 5: Returning Plain Text Response](#example-5-returning-plain-text-response)
   - [Example 6: Returning HTML Response](#example-6-returning-html-response)
   - [Example 7: Returning JSON Response Manually](#example-7-returning-json-response-manually)
   - [Example 8: Streaming a Response](#example-8-streaming-a-response)
   - [Example 9: Redirecting to Another URL](#example-9-redirecting-to-another-url)
   - [Example 10: Setting a Custom Media Type](#example-10-setting-a-custom-media-type)
   - [Example 11: Returning a File Response](#example-11-returning-a-file-response)
   - [Example 12: Setting a Custom Content-Disposition Header](#example-12-setting-a-custom-content-disposition-header)
3. [The JSON Response Object in FastAPI](#the-json-response-object-in-fastapi)
   - [Example 13: Basic JSON Response](#example-13-basic-json-response)
   - [Example 14: Custom JSON Response with Status Code and Headers](#example-14-custom-json-response-with-status-code-and-headers)
   - [Example 15: JSON Response with Metadata](#example-15-json-response-with-metadata)
4. [The Streaming Response Object in FastAPI](#the-streaming-response-object-in-fastapi)
   - [Example 16: Streaming a Large File](#example-16-streaming-a-large-file)
   - [Example 17: Streaming Dynamically Generated Content](#example-17-streaming-dynamically-generated-content)
   - [Example 18: Streaming Real-Time Data](#example-18-streaming-real-time-data)
   - [Example 19: Streaming with Custom Headers](#example-19-streaming-with-custom-headers)
   - [Example 20: Streaming a Video File](#example-20-streaming-a-video-file)
5. [Conclusion](#conclusion)

---

## Introduction

In this article, we explore how FastAPI allows you to customize HTTP responses using different response objects. We will cover basic responses, JSON responses, and streaming responses, along with various examples that illustrate the flexibility of FastAPI's response handling.

---

## The Response Object in FastAPI

The `Response` object in FastAPI allows you to customize the outgoing HTTP response. You can set:

- **Status Codes**
- **Headers**
- **Cookies**
- **Media Types (plain text, HTML, JSON, XML, etc.)**
- **Streamed Content**
- **Redirects**

The table below summarizes some common customization options:

| **Feature**            | **Usage**                                                       | **Example**                                      |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------ |
| Custom Status Code     | Set via `response.status_code`                                  | 201 Created, 307 Temporary Redirect              |
| Custom Headers         | Add headers via `response.headers[...]`                         | `X-Custom-Header: Hello, World!`                 |
| Cookies                | Set or delete cookies using `response.set_cookie()`/`delete_cookie()` | Set cookie "my_cookie"                           |
| Media Type             | Specify via `Response(media_type=...)`                          | `"text/plain"`, `"text/html"`, `"application/json"` |
| File Response          | Return file content with proper media type                      | `"application/octet-stream"`                     |

Below are examples demonstrating various customizations with the `Response` object.

---

### Example 1: Setting a Custom Status Code  
This endpoint sets a custom HTTP status code (201 Created) before returning the response.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/custom-status")
def custom_status(response: Response):
    response.status_code = 201
    return {"message": "Custom status code set to 201"}
```

**Output:**
```
{
  "message": "Custom status code set to 201"
}
```

---

### Example 2: Setting Response Headers  
This endpoint adds a custom header to the response.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/custom-headers")
def custom_headers(response: Response):
    response.headers["X-Custom-Header"] = "Hello, World!"
    return {"message": "Custom header added"}
```

**Output:**
```
{
  "message": "Custom header added"
}
```

---

### Example 3: Setting Cookies  
This endpoint sets a cookie in the response.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/set-cookie")
def set_cookie(response: Response):
    response.set_cookie(key="my_cookie", value="cookie_value")
    return {"message": "Cookie set"}
```

**Output:**
```
{
  "message": "Cookie set"
}
```

---

### Example 4: Deleting Cookies  
This endpoint deletes a specified cookie from the response.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/delete-cookie")
def delete_cookie(response: Response):
    response.delete_cookie(key="my_cookie")
    return {"message": "Cookie deleted"}
```

**Output:**
```
{
  "message": "Cookie deleted"
}
```

---

### Example 5: Returning Plain Text Response  
This endpoint returns a plain text response by setting the `media_type` to `"text/plain"`.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/plain-text")
def plain_text(response: Response):
    return Response(content="This is a plain text response", media_type="text/plain")
```

**Output:**
```
This is a plain text response
```

---

### Example 6: Returning HTML Response  
This endpoint returns an HTML response by specifying the `media_type` as `"text/html"`.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/html-response")
def html_response(response: Response):
    html_content = "<h1>Hello, World!</h1>"
    return Response(content=html_content, media_type="text/html")
```

**Output:**
```
<h1>Hello, World!</h1>
```

---

### Example 7: Returning JSON Response Manually  
This endpoint manually creates a JSON response using Python’s `json.dumps()`.

```python
from fastapi import FastAPI, Response
import json

app = FastAPI()

@app.get("/json-response")
def json_response(response: Response):
    data = {"message": "This is a JSON response"}
    return Response(content=json.dumps(data), media_type="application/json")
```

**Output:**
```
{"message": "This is a JSON response"}
```

---

### Example 8: Streaming a Response  
This endpoint streams text data in chunks using a generator function.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/stream")
def stream_response(response: Response):
    def generate():
        for i in range(5):
            yield f"Chunk {i}\n"
    return Response(content=generate(), media_type="text/plain")
```

**Output:**
```
Chunk 0
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

---

### Example 9: Redirecting to Another URL  
This endpoint sets a redirect by updating the `Location` header and using an appropriate status code.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/redirect")
def redirect(response: Response):
    response.headers["Location"] = "https://fastapi.tiangolo.com"
    response.status_code = 307
    return {"message": "Redirecting..."}
```

**Output:**
```
{
  "message": "Redirecting..."
}
```

---

### Example 10: Setting a Custom Media Type  
This endpoint returns content with a custom media type, such as XML.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/custom-media-type")
def custom_media_type(response: Response):
    content = "<xml><message>Hello</message></xml>"
    return Response(content=content, media_type="application/xml")
```

**Output:**
```
<xml><message>Hello</message></xml>
```

---

### Example 11: Returning a File Response  
This endpoint reads a file from disk and returns it as a binary response.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/file")
def file_response(response: Response):
    with open("example.txt", "rb") as file:
        content = file.read()
    return Response(content=content, media_type="application/octet-stream")
```

**Output:**  
*(The output will be the binary content of "example.txt")*

---

### Example 12: Setting a Custom Content-Disposition Header  
This endpoint returns a file response with a custom `Content-Disposition` header to prompt file download.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/download")
def download_file(response: Response):
    response.headers["Content-Disposition"] = 'attachment; filename="example.txt"'
    with open("example.txt", "rb") as file:
        content = file.read()
    return Response(content=content, media_type="application/octet-stream")
```

**Output:**  
*(The browser will prompt to download "example.txt")*

---

## The JSON Response Object in FastAPI

FastAPI’s `JSONResponse` simplifies returning JSON data by automatically setting the correct `Content-Type` header (`application/json`).

### Example 13: Basic JSON Response  
`JSONResponse` is used to return JSON data with the appropriate header. This example returns a simple JSON response.

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/json-basic")
def json_basic():
    data = {"message": "Hello, World!", "status": "success"}
    return JSONResponse(content=data)
```

**Output:**
```
{
  "message": "Hello, World!",
  "status": "success"
}
```

---

### Example 14: Custom JSON Response with Status Code and Headers  
This example demonstrates how to customize the status code and headers in a `JSONResponse`.

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/json-custom")
def json_custom():
    data = {"message": "Custom JSON response"}
    headers = {"X-Custom-Header": "12345"}
    return JSONResponse(content=data, status_code=201, headers=headers)
```

**Output:**
```
Status Code: 201
Headers: {"X-Custom-Header": "12345", "content-type": "application/json"}
Body:
{
  "message": "Custom JSON response"
}
```

---

### Example 15: JSON Response with Metadata  
This example shows how to include additional metadata in the JSON response.

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/json-metadata")
def json_metadata():
    data = {
        "message": "Response with metadata",
        "metadata": {
            "version": "1.0",
            "author": "FastAPI"
        }
    }
    return JSONResponse(content=data)
```

**Output:**
```
{
  "message": "Response with metadata",
  "metadata": {
    "version": "1.0",
    "author": "FastAPI"
  }
}
```

---

## The Streaming Response Object in FastAPI

FastAPI’s `StreamingResponse` enables efficient handling of large or continuously generated content without loading the entire response into memory.

### Example 16: Streaming a Large File  
`StreamingResponse` streams a large file to the client without loading it entirely into memory.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream-file")
def stream_file():
    def iter_file():
        with open("large_file.txt", "rb") as file:
            while chunk := file.read(1024):  # Read in chunks of 1KB
                yield chunk
    return StreamingResponse(iter_file(), media_type="text/plain")
```

**Output:**  
*(The output will be the streamed content of "large_file.txt" in chunks.)*

---

### Example 17: Streaming Dynamically Generated Content  
This example streams dynamically generated CSV content to the client.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import csv
import io

app = FastAPI()

@app.get("/stream-csv")
def stream_csv():
    def generate_csv():
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["name", "age"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            buffer.seek(0)
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
    return StreamingResponse(generate_csv(), media_type="text/csv")
```

**Output:**  
```
name,age
Alice,30
name,age
Bob,25
name,age
Charlie,35
```

---

### Example 18: Streaming Real-Time Data  
This example streams simulated real-time log entries to the client.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

@app.get("/stream-logs")
def stream_logs():
    def generate_logs():
        for i in range(10):
            time.sleep(1)  # Simulate a delay
            yield f"Log entry {i}\n"
    return StreamingResponse(generate_logs(), media_type="text/plain")
```

**Output:**  
*(Every second, a new log entry is streamed, e.g., "Log entry 0", "Log entry 1", …)*

---

### Example 19: Streaming with Custom Headers  
This example streams data and adds custom headers to the `StreamingResponse`.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream-custom-headers")
def stream_custom_headers():
    def generate_data():
        for i in range(5):
            yield f"Data chunk {i}\n"
    headers = {"X-Custom-Header": "StreamingResponse"}
    return StreamingResponse(generate_data(), media_type="text/plain", headers=headers)
```

**Output:**  
```
Headers: {"X-Custom-Header": "StreamingResponse", ...}
Body:
Data chunk 0
Data chunk 1
Data chunk 2
Data chunk 3
Data chunk 4
```

---

### Example 20: Streaming a Video File  
This example streams a video file to the client in chunks.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream-video")
def stream_video():
    def iter_video():
        with open("sample_video.mp4", "rb") as video_file:
            while chunk := video_file.read(65536):  # Read in chunks of 64KB
                yield chunk
    return StreamingResponse(iter_video(), media_type="video/mp4")
```

**Output:**  
*(The video file "sample_video.mp4" is streamed to the client in binary chunks.)*

---

## Conclusion

FastAPI offers powerful and flexible response objects that allow you to tailor HTTP responses to your application’s needs. Whether you need to customize a plain text response, return JSON data, or stream large files and dynamic content, FastAPI provides a straightforward approach to handling various scenarios. Use these examples as a guide to implement custom responses in your own FastAPI applications.

---