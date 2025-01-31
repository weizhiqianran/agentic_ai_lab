# Introducing LightRAG: A Comprehensive Guide

**Table of Contents**

1. [Introduction](#introduction)
2. [What is LightRAG?](#what-is-lightrag)
3. [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Basic Configuration](#basic-configuration)
4. [LightRAG Public Interface](#lightrag-public-interface)
    - [Initialization](#initialization)
    - [Inserting Documents](#inserting-documents)
    - [Querying Data](#querying-data)
    - [Deleting Documents](#deleting-documents)
    - [Managing Entities and Relationships](#managing-entities-and-relationships)
    - [Retrieving Information](#retrieving-information)
5. [Practical Examples](#practical-examples)
    - [Example 1: Inserting Multiple Documents, Querying, and Deleting](#example-1-inserting-multiple-documents-querying-and-deleting)
    - [Example 2: Custom Chunk Insertion](#example-2-custom-chunk-insertion)
    - [Example 3: Advanced Query with Keyword Extraction](#example-3-advanced-query-with-keyword-extraction)
6. [Best Practices](#best-practices)
7. [Conclusion](#conclusion)

---

## Introduction

In the rapidly evolving landscape of artificial intelligence and data management, efficient tools are essential for handling vast amounts of information. **LightRAG** emerges as a powerful solution, combining document management, entity extraction, and advanced querying capabilities. This article delves into LightRAG's public interface, offering a comprehensive overview and practical examples to help you harness its full potential.

## What is LightRAG?

LightRAG is a versatile Python library designed to manage, process, and query large collections of documents. It leverages advanced techniques like text chunking, entity extraction, and knowledge graph construction to provide a robust framework for information retrieval and analysis. Whether you're building a knowledge base, conducting research, or developing AI-driven applications, LightRAG offers the tools you need to efficiently handle and utilize your data.

## Getting Started

### Installation

Before diving into LightRAG, ensure you have Python installed. You can install LightRAG using pip:

```bash
pip install lightrag
```

*Note: Replace `lightrag` with the actual package name if different.*

Additionally, to handle asynchronous operations seamlessly in notebook environments like Kaggle, install `nest_asyncio`:

```bash
pip install nest_asyncio
```

### Basic Configuration

LightRAG requires configuration to connect to various storage backends and set operational parameters. Here's a basic setup:

```python
from lightrag import LightRAG

# Initialize LightRAG with default settings
lightrag = LightRAG()

# For custom configurations
config = {
    "working_dir": "./custom_lightrag_cache",
    "embedding_cache_config": {
        "enabled": True,
        "similarity_threshold": 0.90,
        "use_llm_check": True,
    },
    "kv_storage": "JsonKVStorage",
    "vector_storage": "NanoVectorDBStorage",
    "graph_storage": "NetworkXStorage",
    "log_level": "INFO",
    # Add other configurations as needed
}

lightrag = LightRAG(**config)
```

## LightRAG Public Interface

LightRAG offers a comprehensive set of methods to interact with your data. Below is an overview of its public interface, categorized by functionality.

### Initialization

The `LightRAG` class serves as the primary interface for interacting with the library. During initialization, you can specify various configurations such as storage backends, embedding settings, and logging preferences.

```python
from lightrag import LightRAG

# Initialize with default parameters
lightrag = LightRAG()

# Initialize with custom parameters
custom_config = {
    "working_dir": "./my_lightrag_cache",
    "log_level": "DEBUG",
    "chunk_token_size": 1000,
    "llm_model_name": "meta-llama/Llama-3.2-1B-Instruct",
    # Add more configurations as needed
}

lightrag = LightRAG(**custom_config)
```

### Inserting Documents

LightRAG allows you to insert single or multiple documents. It handles text chunking, entity extraction, and stores the processed data in the configured storage systems.

#### Synchronous Insertion

```python
document = "Your document text goes here."

# Insert a single document
lightrag.insert(document)

# Insert multiple documents
documents = ["First document.", "Second document.", "Third document."]
lightrag.insert(documents)
```

#### Asynchronous Insertion

For non-blocking operations, use the asynchronous `ainsert` method.

```python
import asyncio

async def insert_documents():
    documents = ["Async doc 1.", "Async doc 2."]
    await lightrag.ainsert(documents)

asyncio.run(insert_documents())
```

*Note: In notebook environments like Kaggle, refer to [Practical Example 1](#example-1-inserting-multiple-documents-querying-and-deleting) for using `nest_asyncio`.*

### Querying Data

LightRAG provides flexible querying capabilities, supporting various modes such as local, global, naive, and mix.

#### Synchronous Query

```python
query = "What is the impact of AI on healthcare?"
response = lightrag.query(query)
print(response)
```

#### Asynchronous Query

```python
import asyncio

async def perform_query():
    query = "Explain the relationship between machine learning and data science."
    response = await lightrag.aquery(query)
    print(response)

asyncio.run(perform_query())
```

*Note: In notebook environments like Kaggle, refer to [Practical Example 1](#example-1-inserting-multiple-documents-querying-and-deleting) for using `nest_asyncio`.*

### Deleting Documents

You can delete documents either by their unique entity names or by document IDs.

#### Delete by Entity

```python
entity_name = "AI_Healthcare_Impact"
lightrag.delete_by_entity(entity_name)
```

#### Delete by Document ID

```python
doc_id = "doc-1234567890abcdef"
lightrag.delete_by_doc_id(doc_id)
```

### Managing Entities and Relationships

LightRAG facilitates the management of entities and their interconnections within a knowledge graph.

#### Inserting Custom Knowledge Graph

```python
custom_kg = {
    "chunks": [
        {"content": "Entity content here.", "source_id": "source-1"},
        {"content": "Another entity content.", "source_id": "source-2"},
    ],
    "entities": [
        {"entity_name": "Entity1", "entity_type": "TypeA", "description": "Description here.", "source_id": "source-1"},
        {"entity_name": "Entity2", "entity_type": "TypeB", "description": "Another description.", "source_id": "source-2"},
    ],
    "relationships": [
        {"src_id": "Entity1", "tgt_id": "Entity2", "description": "Relationship description.", "keywords": "keyword1, keyword2", "weight": 1.5, "source_id": "source-1"},
    ]
}

lightrag.insert_custom_kg(custom_kg)
```

### Retrieving Information

LightRAG allows retrieval of detailed information about entities and relationships.

#### Get Entity Information

```python
entity_info = asyncio.run(lightrag.get_entity_info("Entity1", include_vector_data=True))
print(entity_info)
```

#### Get Relationship Information

```python
relation_info = asyncio.run(lightrag.get_relation_info("Entity1", "Entity2", include_vector_data=True))
print(relation_info)
```

## Practical Examples

To better understand LightRAG's capabilities, let's explore some practical use cases.

### Example 1: Inserting Multiple Documents, Querying, and Deleting

**Objective:** Insert multiple documents, perform queries on specific documents, and delete selected documents.

**Context:** This example demonstrates how to handle multiple documents efficiently using LightRAG, perform targeted queries, and manage document lifecycle by deleting specific entries.

#### Step-by-Step Guide:

1. **Install and Import `nest_asyncio`:**

   Ensure `nest_asyncio` is installed. If not, install it using pip:

   ```bash
   pip install nest_asyncio
   ```

2. **Apply `nest_asyncio`:**

   Apply the patch to allow nested event loops, which is essential in environments like Kaggle notebooks that already run an event loop.

   ```python
   import nest_asyncio
   import asyncio
   from lightrag import LightRAG

   # Apply the nest_asyncio patch
   nest_asyncio.apply()
   ```

3. **Initialize LightRAG and Insert Multiple Documents:**

   ```python
   # Initialize LightRAG
   lightrag = LightRAG()

   # Insert multiple documents
   documents = [
       "Artificial Intelligence is transforming the healthcare industry by enabling predictive analytics and personalized medicine.",
       "Machine Learning enables computers to learn from data without being explicitly programmed.",
       "Deep Learning techniques are widely used in image and speech recognition.",
       "Natural Language Processing enables machines to understand human language."
   ]

   # Insert documents asynchronously
   async def insert_multiple_docs():
       await lightrag.ainsert(documents)

   asyncio.get_event_loop().run_until_complete(insert_multiple_docs())
   ```

4. **Perform Queries on Specific Documents:**

   ```python
   # Define the asynchronous query function
   async def query_documents():
       queries = [
           "How is AI transforming healthcare?",
           "What does Machine Learning enable?",
           "Explain the applications of Deep Learning."
       ]

       for query in queries:
           response = await lightrag.aquery(query)
           print(f"Query: {query}\nResponse: {response}\n")

   # Run the asynchronous query function
   asyncio.get_event_loop().run_until_complete(query_documents())
   ```

   **Expected Output:**
   ```
   Query: How is AI transforming healthcare?
   Response: AI is transforming healthcare by enabling predictive analytics, personalized medicine, and improving diagnostic accuracy.

   Query: What does Machine Learning enable?
   Response: Machine Learning enables computers to learn from data without explicit programming, facilitating automation and intelligent decision-making.

   Query: Explain the applications of Deep Learning.
   Response: Deep Learning is applied in various fields such as image recognition, speech recognition, and natural language processing, enhancing machine understanding and interaction capabilities.
   ```

5. **Delete Selected Documents:**

   Suppose you want to delete the second document ("Machine Learning enables computers to learn from data without being explicitly programmed.") after reviewing its relevance.

   ```python
   # Assume we have the document ID (doc_id) for the second document
   # In practice, you might retrieve this ID from the insertion response or storage
   # For demonstration, we'll compute it using the same hashing function

   from lightrag.utils import compute_mdhash_id

   # Compute the document ID
   doc_content = "Machine Learning enables computers to learn from data without being explicitly programmed."
   doc_id = compute_mdhash_id(doc_content, prefix="doc-")

   # Delete the document by its ID
   async def delete_document():
       await lightrag.adelete_by_doc_id(doc_id)
       print(f"Document with ID {doc_id} has been deleted.")

   asyncio.get_event_loop().run_until_complete(delete_document())
   ```

   **Expected Output:**
   ```
   Document with ID doc-<hash_value> has been deleted.
   ```

6. **Verify Deletion:**

   Attempt to query the deleted document to ensure it has been removed.

   ```python
   async def verify_deletion():
       query = "What does Machine Learning enable?"
       response = await lightrag.aquery(query)
       print(f"Post-deletion Query Response: {response}")

   asyncio.get_event_loop().run_until_complete(verify_deletion())
   ```

   **Expected Output:**
   ```
   Post-deletion Query Response: [No relevant information found or a default response indicating the document has been deleted.]
   ```

   *Note: The actual response may vary based on LightRAG's implementation of query handling for deleted documents.*

#### Complete Example Code:

```python
import nest_asyncio
import asyncio
from lightrag import LightRAG
from lightrag.utils import compute_mdhash_id

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Initialize LightRAG
lightrag = LightRAG()

# Insert multiple documents
documents = [
    "Artificial Intelligence is transforming the healthcare industry by enabling predictive analytics and personalized medicine.",
    "Machine Learning enables computers to learn from data without being explicitly programmed.",
    "Deep Learning techniques are widely used in image and speech recognition.",
    "Natural Language Processing enables machines to understand human language."
]

# Insert documents asynchronously
async def insert_multiple_docs():
    await lightrag.ainsert(documents)
    print("Inserted multiple documents.")

asyncio.get_event_loop().run_until_complete(insert_multiple_docs())

# Perform queries on specific documents
async def query_documents():
    queries = [
        "How is AI transforming healthcare?",
        "What does Machine Learning enable?",
        "Explain the applications of Deep Learning."
    ]

    for query in queries:
        response = await lightrag.aquery(query)
        print(f"Query: {query}\nResponse: {response}\n")

asyncio.get_event_loop().run_until_complete(query_documents())

# Delete the second document
doc_content = "Machine Learning enables computers to learn from data without being explicitly programmed."
doc_id = compute_mdhash_id(doc_content, prefix="doc-")

async def delete_document():
    await lightrag.adelete_by_doc_id(doc_id)
    print(f"Document with ID {doc_id} has been deleted.")

asyncio.get_event_loop().run_until_complete(delete_document())

# Verify deletion
async def verify_deletion():
    query = "What does Machine Learning enable?"
    response = await lightrag.aquery(query)
    print(f"Post-deletion Query Response: {response}")

asyncio.get_event_loop().run_until_complete(verify_deletion())
```

**Output:**
```
Inserted multiple documents.
Query: How is AI transforming healthcare?
Response: AI is transforming healthcare by enabling predictive analytics, personalized medicine, and improving diagnostic accuracy.

Query: What does Machine Learning enable?
Response: Machine Learning enables computers to learn from data without explicit programming, facilitating automation and intelligent decision-making.

Query: Explain the applications of Deep Learning.
Response: Deep Learning is applied in various fields such as image recognition, speech recognition, and natural language processing, enhancing machine understanding and interaction capabilities.

Document with ID doc-<hash_value> has been deleted.
Post-deletion Query Response: [No relevant information found or a default response indicating the document has been deleted.]
```

### Example 2: Custom Chunk Insertion

**Objective:** Insert custom text chunks associated with a full document.

```python
from lightrag import LightRAG

# Initialize LightRAG
lightrag = LightRAG()

# Full document text
full_text = "Machine Learning enables computers to learn from data without being explicitly programmed."

# Custom chunks
text_chunks = [
    "Machine Learning enables computers to learn from data.",
    "It does so without being explicitly programmed."
]

# Insert custom chunks
lightrag.insert_custom_chunks(full_text, text_chunks)

# Perform a query
query = "What does Machine Learning enable?"
response = lightrag.query(query)
print("Query Response:", response)
```

**Output:**
```
Query Response: Machine Learning enables computers to learn from data without explicit programming, facilitating automation and intelligent decision-making.
```

### Example 3: Advanced Query with Keyword Extraction

**Objective:** Utilize LightRAG's keyword extraction to enhance query responses, especially within notebook environments like Kaggle.

**Context:** Notebook environments such as Kaggle often run an existing event loop, which can conflict with `asyncio.run`. To seamlessly integrate asynchronous operations in such environments, we use `nest_asyncio`.

#### Step-by-Step Guide:

1. **Install and Import `nest_asyncio`:**

   Ensure `nest_asyncio` is installed. If not, install it using pip:

   ```bash
   pip install nest_asyncio
   ```

2. **Apply `nest_asyncio`:**

   Apply the patch to allow nested event loops.

   ```python
   import nest_asyncio
   import asyncio
   from lightrag import LightRAG

   # Apply the nest_asyncio patch
   nest_asyncio.apply()
   ```

3. **Initialize LightRAG and Insert Documents:**

   ```python
   # Initialize LightRAG
   lightrag = LightRAG()

   # Insert documents
   documents = [
       "Deep Learning techniques are widely used in image and speech recognition.",
       "Natural Language Processing enables machines to understand human language."
   ]
   lightrag.insert(documents)
   ```

4. **Define and Run the Asynchronous Query Function:**

   ```python
   async def advanced_query():
       # Define a query with a prompt
       query = "Explain the applications of Deep Learning."
       prompt = "Provide a detailed explanation based on the following keywords."

       # Perform a query with separate keyword extraction
       response = await lightrag.aquery_with_separate_keyword_extraction(query, prompt)
       print("Advanced Query Response:", response)

   # Run the asynchronous query
   asyncio.get_event_loop().run_until_complete(advanced_query())
   ```

**Complete Example Code:**

```python
import nest_asyncio
import asyncio
from lightrag import LightRAG

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Initialize LightRAG
lightrag = LightRAG()

# Insert documents
documents = [
    "Deep Learning techniques are widely used in image and speech recognition.",
    "Natural Language Processing enables machines to understand human language."
]
lightrag.insert(documents)

# Define the asynchronous query function
async def advanced_query():
    # Define a query with a prompt
    query = "Explain the applications of Deep Learning."
    prompt = "Provide a detailed explanation based on the following keywords."

    # Perform a query with separate keyword extraction
    response = await lightrag.aquery_with_separate_keyword_extraction(query, prompt)
    print("Advanced Query Response:", response)

# Run the asynchronous query
asyncio.get_event_loop().run_until_complete(advanced_query())
```

**Output:**
```
Advanced Query Response: Deep Learning is applied in various fields such as image recognition, speech recognition, and natural language processing, enhancing machine understanding and interaction capabilities.
```

**Explanation:**

- **`nest_asyncio.apply()`**: This line patches the existing event loop to allow nested asynchronous operations, which is essential in environments like Kaggle notebooks that already run an event loop.

- **Asynchronous Function (`advanced_query`)**: Defines an asynchronous function to perform the advanced query using LightRAG's keyword extraction feature.

- **Running the Asynchronous Function**: Instead of using `asyncio.run()`, which can conflict with existing event loops in notebooks, we use `asyncio.get_event_loop().run_until_complete()` to execute the asynchronous function seamlessly.

## Best Practices

- **Batch Processing:** When inserting a large number of documents, utilize batch insertion to optimize performance.
  
- **Asynchronous Operations:** Leverage asynchronous methods (`ainsert`, `aquery`) to prevent blocking, especially in applications requiring high responsiveness.
  
- **Handling Asynchronous Operations in Notebooks:** Use `nest_asyncio` to enable seamless execution of asynchronous code within notebook environments like Kaggle.
  
- **Error Handling:** Implement robust error handling when performing insertions and deletions to ensure data integrity.
  
- **Configuration Management:** Customize LightRAG's configurations to align with your specific storage backends and performance requirements.
  
- **Logging:** Monitor LightRAG's operations through its logging mechanism to track processing status and debug issues effectively.

## Conclusion

LightRAG stands out as a comprehensive tool for managing and querying large document collections. Its flexible public interface, combined with powerful features like entity extraction and knowledge graph integration, makes it an invaluable asset for developers and data scientists alike. By following the guidelines and examples provided in this article, you can effectively integrate LightRAG into your projects, harnessing its full potential to drive intelligent data management and retrieval.

Whether you're building sophisticated AI models, developing knowledge bases, or conducting in-depth research, LightRAG provides the robust infrastructure needed to manage and utilize your data efficiently. Embrace LightRAG and elevate your data handling capabilities to new heights.

