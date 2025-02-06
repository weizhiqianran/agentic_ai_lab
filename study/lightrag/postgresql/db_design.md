# Introducing PostgreSQLDB: A Comprehensive PostgreSQL Database Handler for Lightrag

**Table of Contents**

1. [Overview](#overview)  
2. [Database Schema Overview](#database-schema-overview)  
3. [SQL Templates Overview](#sql-templates-overview)  
4. [Class Structure and Purpose](#class-structure-and-purpose)  
5. [Detailed Code Walkthrough](#detailed-code-walkthrough)  
   - [Initialization (`__init__`)](#initialization)  
   - [Database Connection (`initdb`)](#database-connection-initdb)  
   - [Table Checking (`check_tables`)](#table-checking-check_tables)  
   - [Query Execution (`query`)](#query-execution-query)  
   - [SQL Execution (`execute`)](#sql-execution-execute)  
   - [Prerequisite Setup (`_prerequisite`)](#prerequisite-setup-prerequisite)  
6. [Summary Table of Key Methods](#summary-table-of-key-methods)  
7. [Conclusion](#conclusion)

---

## Overview

`PostgreSQLDB` is a central class in Lightrag's PostgreSQL implementation. It handles creating connection pools, executing queries, managing table existence, and setting up prerequisites required by the Apache AGE extension for graph operations. This article explains the code step by step and introduces both the underlying database schema and SQL templates used by Lightrag.

---

## Database Schema Overview

The Lightrag framework organizes its data using a structured database schema. Two important dictionaries that define this schema are `NAMESPACE_TABLE_MAP` and `TABLES`.

### NAMESPACE_TABLE_MAP

This mapping connects logical namespaces used by Lightrag to their corresponding table names in the PostgreSQL database.

```python
NAMESPACE_TABLE_MAP = {
    "full_docs": "LIGHTRAG_DOC_FULL",
    "text_chunks": "LIGHTRAG_DOC_CHUNKS",
    "chunks": "LIGHTRAG_DOC_CHUNKS",
    "entities": "LIGHTRAG_VDB_ENTITY",
    "relationships": "LIGHTRAG_VDB_RELATION",
    "doc_status": "LIGHTRAG_DOC_STATUS",
    "llm_response_cache": "LIGHTRAG_LLM_CACHE",
}
```

*Explanation:*  
- Each key represents a logical data category.  
- The value is the corresponding table name in the database.

### TABLES

The `TABLES` dictionary defines the Data Definition Language (DDL) statements for each table. This schema ensures that all required fields are present and that the primary keys are correctly defined.

```sql
CREATE TABLE LIGHTRAG_DOC_FULL (
    id VARCHAR(255),                                  -- Unique identifier for the full document
    workspace VARCHAR(255),                           -- Identifier for the workspace or namespace
    doc_name VARCHAR(1024),                           -- Name of the document
    content TEXT,                                     -- Full text content of the document
    meta JSONB,                                       -- Additional metadata stored in JSONB format
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Timestamp when the document was created
    update_time TIMESTAMP,                            -- Timestamp when the document was last updated
    CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
);

CREATE TABLE LIGHTRAG_DOC_CHUNKS (
    id VARCHAR(255),                                  -- Unique identifier for the document chunk
    workspace VARCHAR(255),                           -- Identifier for the workspace or namespace
    full_doc_id VARCHAR(256),                         -- Identifier of the full document this chunk belongs to
    chunk_order_index INTEGER,                        -- The order index of the chunk within the full document
    tokens INTEGER,                                   -- Number of tokens in the chunk
    content TEXT,                                     -- Text content of the chunk
    content_vector VECTOR,                            -- Vector embedding representing the chunk content
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Timestamp when the chunk was created
    update_time TIMESTAMP,                            -- Timestamp when the chunk was last updated
    CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
);

CREATE TABLE LIGHTRAG_VDB_ENTITY (
    id VARCHAR(255),                                  -- Unique identifier for the entity
    workspace VARCHAR(255),                           -- Identifier for the workspace or namespace
    entity_name VARCHAR(255),                         -- Name of the entity
    content TEXT,                                     -- Description or content related to the entity
    content_vector VECTOR,                            -- Vector embedding representing the entity content
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Timestamp when the entity was created
    update_time TIMESTAMP,                            -- Timestamp when the entity was last updated
    CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
);

CREATE TABLE LIGHTRAG_VDB_RELATION (
    id VARCHAR(255),                                  -- Unique identifier for the relationship
    workspace VARCHAR(255),                           -- Identifier for the workspace or namespace
    source_id VARCHAR(256),                           -- Identifier for the source entity of the relationship
    target_id VARCHAR(256),                           -- Identifier for the target entity of the relationship
    content TEXT,                                     -- Description or content of the relationship
    content_vector VECTOR,                            -- Vector embedding representing the relationship content
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Timestamp when the relationship was created
    update_time TIMESTAMP,                            -- Timestamp when the relationship was last updated
    CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
);

CREATE TABLE LIGHTRAG_LLM_CACHE (
    workspace VARCHAR(255) NOT NULL,                  -- Identifier for the workspace or namespace
    id VARCHAR(255) NOT NULL,                         -- Unique identifier for the cache entry
    mode VARCHAR(32) NOT NULL,                        -- Mode of operation (e.g., global, local)
    original_prompt TEXT,                             -- Original prompt used in the LLM request
    return_value TEXT,                                -- Cached response returned by the LLM
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Timestamp when the cache entry was created
    update_time TIMESTAMP,                            -- Timestamp when the cache entry was last updated
    CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, mode, id)
);

CREATE TABLE LIGHTRAG_DOC_STATUS (
    workspace VARCHAR(255) NOT NULL,                      -- Identifier for the workspace or namespace
    id VARCHAR(255) NOT NULL,                             -- Unique identifier for the document status record
    content_summary VARCHAR(255) NULL,                    -- A brief summary of the document content (first 100 characters)
    content_length INT4 NULL,                             -- Total length of the document (in characters or tokens)
    chunks_count INT4 NULL,                               -- Number of chunks into which the document was split
    status VARCHAR(64) NULL,                              -- Current processing status (e.g., pending, processed, failed)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NULL,  -- Timestamp when the status record was created
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NULL,  -- Timestamp when the status record was last updated
    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
);
```

Here summarizes the key features of each SQL table defined in the schema:

| **Table Name**              | **Description**                                                                                     | **Primary Key**           | **Key Columns / Features**                                                                                     |
|-----------------------------|-----------------------------------------------------------------------------------------------------|---------------------------|----------------------------------------------------------------------------------------------------------------|
| `LIGHTRAG_DOC_FULL`         | Stores the `complete documents` with their full text content and metadata.                           | `workspace, id`           | `doc_name` (document name), `content` (full text), `meta` (JSONB metadata), timestamps for creation and update.  |
| `LIGHTRAG_DOC_CHUNKS`       | Contains `chunks of full documents`, useful for granular retrieval and processing.                    | `workspace, id`           | `full_doc_id` (reference to full document), `chunk_order_index` (order within document), `tokens`, `content`, and a vector representation.  |
| `LIGHTRAG_VDB_ENTITY`       | Holds information about `entities` extracted from documents.                                        | `workspace, id`           | `entity_name` (name of the entity), `content` (entity description), and a vector embedding (`content_vector`).  |
| `LIGHTRAG_VDB_RELATION`     | Captures `relationships` between entities to build a knowledge graph.                               | `workspace, id`           | `source_id` and `target_id` (linking entities), `content` (relationship description), and its vector embedding. |
| `LIGHTRAG_LLM_CACHE`        | Caches `responses from LLM` interactions for efficient reuse.                      | `workspace, mode, id`     | `original_prompt` (LLM input), `return_value` (LLM output), with a mode indicator (e.g., global or local).      |
| `LIGHTRAG_DOC_STATUS`       | Tracks the `processing status` of documents (e.g., pending, processed, failed).                       | `workspace, id`           | `content_summary` (brief summary), `content_length`, `chunks_count` (number of chunks), `status`, and timestamps. |

---

## SQL Templates Overview

The SQL templates define the common SQL queries and commands used by Lightrag. These templates are stored in the `SQL_TEMPLATES` dictionary and cover various operations, such as retrieving data and upserting records.

```sql
-- Template ID: get_by_id_full_docs
-- Retrieves the full document content by its id.
SELECT id, COALESCE(content, '') as content
FROM LIGHTRAG_DOC_FULL 
WHERE workspace = $1 AND id = $2;

-- Template ID: get_by_id_text_chunks
-- Retrieves the text chunk data by its id, including token count, order index, and associated full document id.
SELECT id, tokens, COALESCE(content, '') as content,
       chunk_order_index, full_doc_id
FROM LIGHTRAG_DOC_CHUNKS 
WHERE workspace = $1 AND id = $2;

-- Template ID: get_by_id_llm_response_cache
-- Retrieves an LLM cache entry by its id and mode.
SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
FROM LIGHTRAG_LLM_CACHE 
WHERE workspace = $1 AND mode = $2;

-- Template ID: get_by_mode_id_llm_response_cache
-- Retrieves an LLM cache entry for a given mode and id.
SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
FROM LIGHTRAG_LLM_CACHE 
WHERE workspace = $1 AND mode = $2 AND id = $3;

-- Template ID: get_by_ids_full_docs
-- Retrieves multiple full documents by a list of ids.
SELECT id, COALESCE(content, '') as content
FROM LIGHTRAG_DOC_FULL 
WHERE workspace = $1 AND id IN ({ids});

-- Template ID: get_by_ids_text_chunks
-- Retrieves multiple text chunks by a list of ids.
SELECT id, tokens, COALESCE(content, '') as content,
       chunk_order_index, full_doc_id
FROM LIGHTRAG_DOC_CHUNKS 
WHERE workspace = $1 AND id IN ({ids});

-- Template ID: get_by_ids_llm_response_cache
-- Retrieves multiple LLM cache entries by a list of ids.
SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
FROM LIGHTRAG_LLM_CACHE 
WHERE workspace = $1 AND mode IN ({ids});

-- Template ID: filter_keys
-- Returns the ids that exist in a specific table for a given workspace.
SELECT id 
FROM {table_name} 
WHERE workspace = $1 AND id IN ({ids});

-- Template ID: upsert_doc_full
-- Inserts a full document; if a conflict occurs (same workspace and id), updates the content and update_time.
INSERT INTO LIGHTRAG_DOC_FULL (id, content, workspace)
VALUES ($1, $2, $3)
ON CONFLICT (workspace, id) DO UPDATE
   SET content = $2, update_time = CURRENT_TIMESTAMP;

-- Template ID: upsert_llm_response_cache
-- Inserts an LLM cache entry; on conflict, updates the prompt, return value, mode, and update_time.
INSERT INTO LIGHTRAG_LLM_CACHE(workspace, id, original_prompt, return_value, mode)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (workspace, mode, id) DO UPDATE
   SET original_prompt = EXCLUDED.original_prompt,
       return_value = EXCLUDED.return_value,
       mode = EXCLUDED.mode,
       update_time = CURRENT_TIMESTAMP;

-- Template ID: upsert_chunk
-- Inserts a text chunk; on conflict, updates tokens, order, full_doc_id, content, vector, and update_time.
INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens, chunk_order_index, full_doc_id, content, content_vector)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (workspace, id) DO UPDATE
   SET tokens = EXCLUDED.tokens,
       chunk_order_index = EXCLUDED.chunk_order_index,
       full_doc_id = EXCLUDED.full_doc_id,
       content = EXCLUDED.content,
       content_vector = EXCLUDED.content_vector,
       update_time = CURRENT_TIMESTAMP;

-- Template ID: upsert_entity
-- Inserts an entity record; on conflict, updates entity_name, content, vector, and update_time.
INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content, content_vector)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (workspace, id) DO UPDATE
   SET entity_name = EXCLUDED.entity_name,
       content = EXCLUDED.content,
       content_vector = EXCLUDED.content_vector,
       update_time = CURRENT_TIMESTAMP;

-- Template ID: upsert_relationship
-- Inserts a relationship record; on conflict, updates source_id, target_id, content, vector, and update_time.
INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id, target_id, content, content_vector)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (workspace, id) DO UPDATE
   SET source_id = EXCLUDED.source_id,
       target_id = EXCLUDED.target_id,
       content = EXCLUDED.content,
       content_vector = EXCLUDED.content_vector,
       update_time = CURRENT_TIMESTAMP;

-- Template ID: entities
-- Queries entities based on vector similarity. The placeholder [{embedding_string}] is replaced with the query embedding.
SELECT entity_name FROM
    (SELECT id, entity_name, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
     FROM LIGHTRAG_VDB_ENTITY where workspace = $1)
WHERE distance > $2
ORDER BY distance DESC
LIMIT $3;

-- Template ID: relationships
-- Queries relationships based on vector similarity.
SELECT source_id as src_id, target_id as tgt_id FROM
    (SELECT id, source_id, target_id, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
     FROM LIGHTRAG_VDB_RELATION where workspace = $1)
WHERE distance > $2
ORDER BY distance DESC
LIMIT $3;

-- Template ID: chunks
-- Queries text chunks based on vector similarity.
SELECT id FROM
    (SELECT id, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
     FROM LIGHTRAG_DOC_CHUNKS where workspace = $1)
WHERE distance > $2
ORDER BY distance DESC
LIMIT $3;
```

*Explanation:*  
- The SQL templates cover operations for key-value storage (KVStorage) and vector storage.  
- They include commands for retrieving data by ID(s), filtering keys, and upserting records for different namespaces such as full documents, text chunks, entities, relationships, and LLM cache.  
- Templates for vector storage queries (e.g., `"entities"`, `"relationships"`, `"chunks"`) use a placeholder (`[{embedding_string}]`) to inject computed embeddings into the SQL query for similarity search based on cosine distance.

---

## Class Structure and Purpose

The `PostgreSQLDB` class is designed to:

- **Establish Database Connections:** Create and manage a pool of asynchronous connections to a PostgreSQL database.  
- **Verify and Create Tables:** Check for the existence of required tables and create them if they do not exist.  
- **Execute Queries:** Provide methods for running SQL queries and non-query commands (e.g., INSERT, UPDATE).  
- **Support Graph Operations:** Set up prerequisites for Apache AGE by configuring the connection's search path and ensuring that a graph exists.

---

## Detailed Code Walkthrough

### Initialization

```python
class PostgreSQLDB:
    def __init__(self, config, **kwargs):
        # Initialize database connection parameters from the config dictionary.
        self.pool = None
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.user = config.get("user", "postgres")
        self.password = config.get("password", None)
        self.database = config.get("database", "postgres")
        self.workspace = config.get("workspace", "default")
        self.max = 12       # Maximum number of connections in the pool.
        self.increment = 1  # Additional configuration (not explicitly used).
        logger.info(f"Using the label {self.workspace} for PostgreSQL as identifier")

        # Validate that essential parameters are provided.
        if self.user is None or self.password is None or self.database is None:
            raise ValueError("Missing database user, password, or database in addon_params")
```

*Explanation:*  
- The constructor reads configuration values and sets defaults where necessary.  
- It ensures that critical connection parameters (user, password, database) are present.

---

### Database Connection

```python
async def initdb(self):
    try:
        # Create a connection pool using asyncpg's create_pool function.
        # This pool will manage multiple connections to the PostgreSQL database.
        self.pool = await asyncpg.create_pool(
            user=self.user,             # Database username
            password=self.password,     # Database password
            database=self.database,     # Name of the database to connect to
            host=self.host,             # Database host address
            port=self.port,             # Port number on which PostgreSQL is running
            min_size=1,                 # Minimum number of connections in the pool
            max_size=self.max,          # Maximum number of connections in the pool
        )
        # Log a message indicating a successful connection.
        logger.info(f"Connected to PostgreSQL database at {self.host}:{self.port}/{self.database}")
    except Exception as e:
        # If an error occurs during pool creation, log an error message.
        logger.error(f"Failed to connect to PostgreSQL database at {self.host}:{self.port}/{self.database}")
        logger.error(f"PostgreSQL database error: {e}")
        # Reraise the exception so that the calling code can handle it.
        raise
```

*Explanation:*  
- `initdb` asynchronously creates a pool of connections using the `asyncpg` library.  
- In case of failure, it logs the error and re-raises the exception.

---

### Table Checking

```python
async def check_tables(self):
    for k, v in TABLES.items():
        try:
            # Try executing a simple query on each table to verify its existence.
            await self.query("SELECT 1 FROM {k} LIMIT 1".format(k=k))
        except Exception as e:
            logger.error(f"Failed to check table {k} in PostgreSQL database")
            logger.error(f"PostgreSQL database error: {e}")
            try:
                # If the table does not exist, create it using the DDL.
                await self.execute(v["ddl"])
                logger.info(f"Created table {k} in PostgreSQL database")
            except Exception as e:
                logger.error(f"Failed to create table {k} in PostgreSQL database")
                logger.error(f"PostgreSQL database error: {e}")
    logger.info("Finished checking all tables in PostgreSQL database")
```

*Explanation:*  
- This method iterates through the `TABLES` dictionary, attempting to query each table.  
- If a table is missing, it creates the table using its Data Definition Language (DDL) statement.

---

### Query Execution

```python
async def query(self, sql: str, params: dict = None, multirows: bool = False, for_age: bool = False, graph_name: str = None) -> Union[dict, None, list[dict]]:
    # Acquire a connection from the pool
    async with self.pool.acquire() as connection:
        try:
            # If the query is intended for Apache AGE (graph queries), set up prerequisites
            if for_age:
                await PostgreSQLDB._prerequisite(connection, graph_name)
            
            # Execute the query:
            # - If parameters are provided, unpack them into the query.
            # - Otherwise, execute the query as-is.
            if params:
                rows = await connection.fetch(sql, *params.values())
            else:
                rows = await connection.fetch(sql)
            
            # Process the fetched rows:
            if multirows:
                # When expecting multiple rows, if rows exist,
                # extract the column names from the first row and convert each row to a dictionary.
                if rows:
                    columns = [col for col in rows[0].keys()]
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    # If no rows were returned, assign an empty list.
                    data = []
            else:
                # When expecting a single row, convert the first row to a dictionary if it exists.
                if rows:
                    columns = rows[0].keys()
                    data = dict(zip(columns, rows[0]))
                else:
                    # If no rows were returned, assign None.
                    data = None
            
            # Return the processed data.
            return data
        except Exception as e:
            # Log the error and print the SQL statement and parameters for debugging.
            logger.error(f"PostgreSQL database error: {e}")
            print(sql)
            print(params)
            # Reraise the exception to signal failure.
            raise
```

*Explanation:*  
- The `query` method executes SQL statements and returns the result as either a dictionary (for a single row) or a list of dictionaries (for multiple rows).  
- It supports AGE-specific queries by setting up prerequisites if `for_age` is True.

---

### SQL Execution

```python
async def execute(self, sql: str, data: Union[list, dict] = None, for_age: bool = False, graph_name: str = None, upsert: bool = False):
    try:
        async with self.pool.acquire() as connection:
            if for_age:
                await PostgreSQLDB._prerequisite(connection, graph_name)
            if data is None:
                await connection.execute(sql)
            else:
                await connection.execute(sql, *data.values())
    except (asyncpg.exceptions.UniqueViolationError, asyncpg.exceptions.DuplicateTableError) as e:
        if upsert:
            print("Key value duplicate, but upsert succeeded.")
        else:
            logger.error(f"Upsert error: {e}")
    except Exception as e:
        logger.error(f"PostgreSQL database error: {e.__class__} - {e}")
        print(sql)
        print(data)
        raise
```

*Explanation:*  
- The `execute` method handles SQL commands that modify data (e.g., INSERT, UPDATE).  
- It also handles specific exceptions for unique constraints or duplicate tables separately.

---

### Prerequisite Setup

```python
@staticmethod
async def _prerequisite(conn: asyncpg.Connection, graph_name: str):
    try:
        # Set the search_path to include the AGE catalog.
        await conn.execute('SET search_path = ag_catalog, "$user", public')
        # Create the graph if it doesn't exist.
        await conn.execute(f"""select create_graph('{graph_name}')""")
    except (asyncpg.exceptions.InvalidSchemaNameError, asyncpg.exceptions.UniqueViolationError):
        # Ignore errors that indicate the graph or schema already exists.
        pass
```

*Explanation:*  
- The `_prerequisite` method prepares a connection for AGE-specific queries by setting the search path and ensuring the graph exists.  
- It handles exceptions gracefully if the graph already exists.

---

## Summary Table of Key Methods

| **Method**           | **Purpose**                                                        | **Key Features**                                                                                   |
|----------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `__init__`           | Initializes connection parameters and validates configuration      | Sets default values and validates that essential parameters are provided                           |
| `initdb`             | Creates an asynchronous connection pool to the PostgreSQL database   | Uses `asyncpg.create_pool` and logs success or error messages                                       |
| `check_tables`       | Checks for existence of required tables and creates them if missing  | Iterates over table definitions; runs a test query and executes DDL if needed                        |
| `query`              | Executes SQL queries and returns results as dictionaries             | Supports single/multiple rows; handles AGE-specific prerequisites; logs errors                      |
| `execute`            | Executes SQL commands that modify data (INSERT, UPDATE, etc.)          | Handles unique constraint errors and logs SQL/data on errors                                       |
| `_prerequisite`      | Sets up the PostgreSQL connection for AGE queries                      | Configures the search path and creates the graph if it doesn't exist                                |

---

## Conclusion

The `PostgreSQLDB` class is a robust component of the Lightrag framework that provides a unified approach to managing PostgreSQL database connections, executing queries, and ensuring that necessary tables and graph contexts exist. The class not only supports traditional SQL operations but also sets the stage for graph-specific commands through Apache AGE.

Additionally, the database schema and SQL templates play crucial roles:

- **Database Schema:**  
  The schema is defined by the `NAMESPACE_TABLE_MAP` and `TABLES` dictionaries, which map logical namespaces to actual table names and provide the necessary DDL for creating these tables. This ensures that Lightrag's various data types—documents, chunks, entities, relationships, and more—are stored in a structured and consistent manner.

- **SQL Templates:**  
  The `SQL_TEMPLATES` dictionary contains pre-defined SQL queries and commands that are used throughout Lightrag. These templates cover data retrieval, filtering, and upsert operations for both key-value storage and vector similarity queries. They provide a centralized way to manage SQL commands and help ensure consistency across the system.

Understanding these components will help you maintain, extend, and troubleshoot your Lightrag deployments more effectively.
