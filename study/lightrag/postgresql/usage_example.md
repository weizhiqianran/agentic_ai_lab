# Implementing LightRAG with Zhipu AI, Ollama, and PostgreSQL

## Table of Contents

1. [Overview](#1-overview)
2. [Code Breakdown](#2-code-breakdown)
   - [Importing Dependencies](#importing-dependencies)
   - [Configuration Setup](#configuration-setup)
   - [Initializing PostgreSQL Database](#initializing-postgresql-database)
   - [Configuring LightRAG](#configuring-lightrag)
   - [Ingesting Data](#ingesting-data)
   - [Performing Queries](#performing-queries)
3. [Key Takeaways](#3-key-takeaways)
4. [Conclusion](#4-conclusion)

## 1. Overview

This script demonstrates how to integrate **LightRAG** with **Zhipu AI**, **Ollama embeddings**, and **PostgreSQL**. It sets up a **retrieval-augmented generation (RAG)** system capable of querying large text data efficiently with vector search and knowledge graph storage.

## 2. Code Breakdown

### Importing Dependencies

The script begins by importing necessary modules:

```python
import asyncio
import logging
import os
import time
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.kg.postgres_impl import PostgreSQLDB
from lightrag.llm.zhipu import zhipu_complete
from lightrag.llm.ollama import ollama_embedding
from lightrag.utils import EmbeddingFunc
```

- **LightRAG**: Core framework for retrieval-augmented generation.
- **Zhipu AI & Ollama**: Used for **LLM-based text completion** and **text embeddings**.
- **PostgreSQLDB**: Backend storage for structured and vector data.
- **QueryParam**: Defines query parameters such as search modes (naive, local, global, hybrid).

### Configuration Setup

```python
load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
WORKING_DIR = f"{ROOT_DIR}/dickens-pg"
```

- Loads environment variables using `dotenv`.
- Defines `WORKING_DIR` where all generated data will be stored.

Set up logging for debugging:

```python
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
```

Create the working directory if it does not exist:

```python
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
```

---

### Initializing PostgreSQL Database

```python
postgres_db = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 15432,
        "user": "rag",
        "password": "rag",
        "database": "rag",
        "workspace": "default"  # Set the workspace (namespace) for this instance
    }
)
```

- Configures PostgreSQL for storage of vectors, knowledge graphs, and document statuses.

### Creating Multiple Workspaces

To create more than one workspace, simply initialize separate instances of `PostgreSQLDB` with different workspace values, or use the same instance to manage multiple workspaces. For example:

```python
# Workspace for Project A
postgres_db_project_a = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 15432,
        "user": "rag",
        "password": "rag",
        "database": "rag",
        "workspace": "project_a"  # Set the workspace (namespace) for project-a
    }
)

# Workspace for Project B
postgres_db_project_b = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 15432,
        "user": "rag",
        "password": "rag",
        "database": "rag",
        "workspace": "project_b"  # Set the workspace (namespace) for project-b
    }
)
```

In this way, records inserted into each workspace will be stored with the respective workspace identifier. The primary keys in the tables are composite keys (e.g., `(workspace, id)`), which ensures that data from different workspaces remains isolated. This approach is especially useful if you want to support multiple projects or testing environments without interference.

---

### Configuring LightRAG

After initializing the database, we set up **LightRAG**:

```python
async def main():
    # Initialize PostgreSQL database and create tables if they do not exist
    await postgres_db.initdb()
    await postgres_db.check_tables()

    # Create an instance of LightRAG with configured settings
    rag = LightRAG(
        working_dir=WORKING_DIR,                   # Directory where LightRAG stores processed data
        llm_model_func=zhipu_complete,             # Function for LLM-based text generation (Zhipu AI)
        llm_model_name="glm-4-flashx",             # Name of the language model used
        llm_model_max_async=4,                     # Maximum number of concurrent LLM queries
        llm_model_max_token_size=32768,            # Maximum token length allowed in the LLM response
        enable_llm_cache_for_entity_extract=True,  # Enables caching for entity extraction to optimize queries

        # Define the embedding function using Ollama for text embedding
        embedding_func=EmbeddingFunc(
            embedding_dim=768,                     # The size of the embedding vector
            max_token_size=8192,                   # Maximum number of tokens that can be processed
            func=lambda texts: ollama_embedding(   # Embedding function using Ollama
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),

        # Define storage types for key-value, document status, graph, and vector data
        kv_storage="PGKVStorage",                  # PostgreSQL-based key-value storage
        doc_status_storage="PGDocStatusStorage",   # PostgreSQL storage for document processing statuses
        graph_storage="PGGraphStorage",            # PostgreSQL-based knowledge graph storage
        vector_storage="PGVectorStorage",          # PostgreSQL-based vector database storage
    )

```

- Uses **Zhipu AI** (`zhipu_complete`) as the language model.
- Uses **Ollama embeddings** (`ollama_embedding`) for vector similarity search.
- Sets up **PostgreSQL-based storage** for document, vector, and knowledge graph management.
- Configures **async execution** to maximize efficiency.

---

### Linking LightRAG to PostgreSQL

```python
    # Assign the PostgreSQL database connection to various storage components in LightRAG
    # This ensures that all storage operations use the same database connection pool

    rag.doc_status.db = postgres_db              # Stores document processing statuses
    rag.full_docs.db = postgres_db               # Stores full documents before chunking
    rag.text_chunks.db = postgres_db             # Stores document chunks used for retrieval

    rag.llm_response_cache.db = postgres_db      # Caches LLM responses for efficiency
    
    rag.chunks_vdb.db = postgres_db              # Stores vector embeddings of document chunks
    rag.relationships_vdb.db = postgres_db       # Stores vector embeddings of relationships between entities
    rag.entities_vdb.db = postgres_db            # Stores vector embeddings of extracted entities
    
    rag.key_string_value_json_storage_cls.db = postgres_db  # Stores key-value JSON data
    rag.graph_storage_cls.db = postgres_db                  # Stores the structured knowledge graph data
    rag.chunk_entity_relation_graph.db = postgres_db        # Stores entity-relationship graph built from document chunks

    # Assigns the embedding function to the chunk-entity relation graph to enable vector operations
    rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func
```

- Ensures that all storage operations use the same PostgreSQL connection.
- Assigns an **embedding function** to the knowledge graph.

---

### Ingesting Data

```python
    with open(f"{ROOT_DIR}/book.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())
```

- Reads and inserts the contents of `book.txt` into the RAG system.
- This step is essential for enabling **document retrieval and entity linking**.

---

### Performing Queries

Once data is ingested, we execute different query types:

#### Naive Query (Simple Vector Search)

```python
    print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="naive")))
```

#### Local Query (Retrieves Information from Entities)

```python
    print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="local")))
```

#### Global Query (Knowledge Graph Retrieval)

```python
    print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="global")))
```

#### Hybrid Query (Combines Multiple Retrieval Techniques)

```python
    print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
```

## 3. Key Takeaways

- **LightRAG** provides an efficient **retrieval-augmented generation (RAG)** solution.
- **PostgreSQL** serves as a **scalable backend** for both vector search and knowledge graphs.
- **Zhipu AI** enables **LLM-powered text generation**.
- **Ollama embeddings** are used for **semantic similarity**.
- The **asynchronous architecture** ensures smooth execution.

## 4. Conclusion

This script provides a **complete implementation** of LightRAG using **Zhipu AI, Ollama embeddings, and PostgreSQL**. The combination of **vector search, knowledge graphs, and LLMs** makes it ideal for **context-aware AI assistants** and **intelligent document retrieval**.
