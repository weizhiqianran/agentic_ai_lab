# Introducing PGDocStatusStorage: A PostgreSQL Implementation for Document Status Management

**Table of Contents**

1. [Overview](#overview)
2. [PGDocStatusStorage in Context](#pgdocstatusstorage-in-context)
3. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
   - [Class Initialization and Inheritance](#class-initialization-and-inheritance)
   - [Filtering Keys](#filtering-keys)
   - [Retrieving Document Status by ID](#retrieving-document-status-by-id)
   - [Aggregating Status Counts](#aggregating-status-counts)
   - [Retrieving Documents by Status](#retrieving-documents-by-status)
   - [Upserting Document Status](#upserting-document-status)
   - [Index Callback](#index-callback)
4. [Summary Table of Key Methods](#summary-table-of-key-methods)
5. [Conclusion](#conclusion)

---

## Overview

`PGDocStatusStorage` is a concrete implementation of the abstract `DocStatusStorage` class that manages document processing statuses in a PostgreSQL database. It is part of the Lightrag framework, designed to keep track of the status (e.g., pending, processing, processed, failed) of documents being ingested and processed. This storage class interacts with PostgreSQL via an instance of the `PostgreSQLDB` class and leverages SQL queries to manage document status records.

---

## PGDocStatusStorage in Context

Within Lightrag, tracking the document processing status is essential for monitoring, debugging, and retrying failed document processes. `PGDocStatusStorage` provides the following functionalities:

- **Filtering Keys:** Identify which document IDs are not yet present in the database.
- **Retrieving Document Status:** Get the status of a single document by its ID.
- **Aggregating Status Counts:** Summarize the number of documents in each status category.
- **Retrieving Documents by Status:** Fetch all documents that are pending, failed, or in any specified status.
- **Upserting Document Status:** Insert new document statuses or update existing ones.
- **Index Callback:** Notify when indexing is complete (although in PostgreSQL the changes are committed during upsert).

---

## Step-by-Step Code Walkthrough

Below is a detailed explanation of the `PGDocStatusStorage` class. Each section includes code snippets with added comments and a description of the logic behind it.

### Class Initialization and Inheritance

`PGDocStatusStorage` inherits from the abstract base class `DocStatusStorage`, which itself is a subclass of `BaseKVStorage`. It defines PostgreSQL-specific behavior for managing document statuses.

```python
@dataclass
class PGDocStatusStorage(DocStatusStorage):
    """PostgreSQL implementation of document status storage"""

    db: PostgreSQLDB = None

    def __post_init__(self):
        # Post-initialization hook; additional initialization logic could be placed here.
        pass
```

*Explanation:*  
- The `@dataclass` decorator simplifies initialization.
- The `db` attribute holds a reference to the PostgreSQLDB instance.
- The `__post_init__` method is available for any extra setup, though in this implementation it does nothing extra.

---

### Filtering Keys

The `filter_keys` method checks which document IDs do not exist in the database. This prevents duplicate entries when new documents are processed.

```python
async def filter_keys(self, data: list[str]) -> set[str]:
    """Return keys that don't exist in storage"""
    # Format the list of keys into a string for SQL IN clause
    keys = ",".join([f"'{_id}'" for _id in data])
    sql = (
        f"SELECT id FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND id IN ({keys})"
    )
    # Execute the query with workspace parameter
    result = await self.db.query(sql, {"workspace": self.db.workspace}, True)
    # If no result is returned, all keys are new; otherwise, subtract existing keys
    if result is None:
        return set(data)
    else:
        existed = set([element["id"] for element in result])
        return set(data) - existed
```

*Explanation:*  
- The keys are wrapped in quotes and concatenated to form the SQL IN clause.
- The SQL query checks for existing IDs in the `LIGHTRAG_DOC_STATUS` table.
- The method returns the set of IDs that are not found in the database.

---

### Retrieving Document Status by ID

The `get_by_id` method retrieves the status record of a single document by its ID.

```python
async def get_by_id(self, id: str) -> Union[T, None]:
    # Prepare SQL query to fetch document status
    sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and id=$2"
    params = {"workspace": self.db.workspace, "id": id}
    result = await self.db.query(sql, params, True)
    # Return None if no record is found; otherwise, create a DocProcessingStatus object
    if result is None or result == []:
        return None
    else:
        return DocProcessingStatus(
            content_length=result[0]["content_length"],
            content_summary=result[0]["content_summary"],
            status=result[0]["status"],
            chunks_count=result[0]["chunks_count"],
            created_at=result[0]["created_at"],
            updated_at=result[0]["updated_at"],
        )
```

*Explanation:*  
- A SQL query retrieves the document status record from the table.
- If the query returns an empty result, the method returns `None`.
- Otherwise, it constructs a `DocProcessingStatus` instance populated with the returned values.

---

### Aggregating Status Counts

The `get_status_counts` method summarizes the number of documents in each status category.

```python
async def get_status_counts(self) -> Dict[str, int]:
    """Get counts of documents in each status"""
    sql = """SELECT status as "status", COUNT(1) as "count"
             FROM LIGHTRAG_DOC_STATUS
             where workspace=$1 GROUP BY STATUS
          """
    result = await self.db.query(sql, {"workspace": self.db.workspace}, True)
    counts = {}
    # Iterate over each record and store counts in a dictionary
    for doc in result:
        counts[doc["status"]] = doc["count"]
    return counts
```

*Explanation:*  
- The SQL query groups documents by their status and counts them.
- The result is parsed into a Python dictionary mapping each status to its count.

---

### Retrieving Documents by Status

The `get_docs_by_status` method retrieves all documents with a specific processing status (e.g., failed or pending).

```python
async def get_docs_by_status(self, status: DocStatus) -> Dict[str, DocProcessingStatus]:
    """Get all documents by status"""
    sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and status=$1"
    params = {"workspace": self.db.workspace, "status": status}
    result = await self.db.query(sql, params, True)
    # Convert each record into a DocProcessingStatus instance and build a dictionary keyed by document ID
    return {
        element["id"]: DocProcessingStatus(
            content_summary=element["content_summary"],
            content_length=element["content_length"],
            status=element["status"],
            created_at=element["created_at"],
            updated_at=element["updated_at"],
            chunks_count=element["chunks_count"],
        )
        for element in result
    }
```

*Explanation:*  
- The method takes a status parameter (using the `DocStatus` enum) and retrieves all matching records.
- The results are converted to a dictionary where the keys are document IDs and the values are `DocProcessingStatus` objects.

There are also two convenience methods that utilize `get_docs_by_status`:

```python
async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
    """Get all failed documents"""
    return await self.get_docs_by_status(DocStatus.FAILED)

async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
    """Get all pending documents"""
    return await self.get_docs_by_status(DocStatus.PENDING)
```

*Explanation:*  
- These methods simply pass the appropriate `DocStatus` enum value to fetch failed or pending documents, respectively.

---

### Upserting Document Status

The `upsert` method inserts a new status record or updates an existing one based on a document ID.

```python
async def upsert(self, data: dict[str, dict]):
    """Update or insert document status

    Args:
        data: Dictionary of document IDs and their status data
    """
    sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content_summary,content_length,chunks_count,status)
             values($1,$2,$3,$4,$5,$6)
              on conflict(id,workspace) do update set
              content_summary = EXCLUDED.content_summary,
              content_length = EXCLUDED.content_length,
              chunks_count = EXCLUDED.chunks_count,
              status = EXCLUDED.status,
              updated_at = CURRENT_TIMESTAMP"""
    # Iterate over each key-value pair in the data dictionary
    for k, v in data.items():
        # chunks_count is optional; if missing, use -1 as a placeholder
        await self.db.execute(
            sql,
            {
                "workspace": self.db.workspace,
                "id": k,
                "content_summary": v["content_summary"],
                "content_length": v["content_length"],
                "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                "status": v["status"],
            },
        )
    return data
```

*Explanation:*  
- The SQL statement uses PostgreSQL's `ON CONFLICT` clause to update an existing record if the document ID and workspace combination already exists.
- For each document, the method constructs the parameters and executes the SQL command.
- Finally, it returns the input data.

---

### Index Callback

The `index_done_callback` method is intended to be called once the indexing or status update operation is complete. In this PostgreSQL implementation, the data is already committed during the upsert stage, so the method simply logs an informational message.

```python
async def index_done_callback(self):
    """Save data after indexing, but for PostgreSQL, we already saved them during the upsert stage, so no action to take here"""
    logger.info("Doc status had been saved into postgresql db!")
```

*Explanation:*  
- This callback serves as a notification hook for when the upsert operation is complete.
- It logs a confirmation message indicating that the document status has been saved.

---

## Summary Table of Key Methods

| **Method**             | **Purpose**                                               | **Key Behavior**                                                                                      |
|------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `filter_keys`          | Filters out existing document IDs                       | Queries the database to return only IDs that are not already present                                  |
| `get_by_id`            | Retrieves a document's processing status by its ID        | Constructs a `DocProcessingStatus` object if the record exists                                        |
| `get_status_counts`    | Aggregates counts of documents per status                 | Groups records by status and returns a dictionary mapping status to count                             |
| `get_docs_by_status`   | Retrieves all documents with a given processing status    | Returns a dictionary of document IDs and their corresponding `DocProcessingStatus` objects             |
| `get_failed_docs`      | Convenience method to get documents with a FAILED status  | Calls `get_docs_by_status` with `DocStatus.FAILED`                                                     |
| `get_pending_docs`     | Convenience method to get documents with a PENDING status | Calls `get_docs_by_status` with `DocStatus.PENDING`                                                    |
| `upsert`               | Inserts or updates document status records                | Uses SQL `ON CONFLICT` to update existing entries or insert new records                                |
| `index_done_callback`  | Callback after indexing/upsert is complete                | Logs a message indicating successful storage of document statuses                                     |

---

## Conclusion

`PGDocStatusStorage` is a PostgreSQL-specific implementation that efficiently handles document processing statuses within the Lightrag framework. By offering methods for filtering, retrieving, aggregating, and upserting document statuses, it provides a robust interface for managing the lifecycle of document processing tasks. The code is designed with clarity and maintainability in mind, leveraging Python data classes, asynchronous programming, and PostgreSQL features to achieve reliable storage operations.

We hope this step-by-step walkthrough, enriched with code comments and summary tables, has clarified how `PGDocStatusStorage` functions and how you can extend or utilize it within your own projects.
