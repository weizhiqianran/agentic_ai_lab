# Introduction to PGKVStorage: A PostgreSQL-based Key–Value Storage Implementation for Lightrag

**Table of Contents**

1. [Overview of PGKVStorage](#overview-of-pgkvstorage)
2. [The Role of PGKVStorage in Lightrag](#the-role-of-pgkvstorage-in-lightrag)
3. [Step-by-Step Code Explanation](#step-by-step-code-explanation)
   - [Initialization and Configuration](#initialization-and-configuration)
   - [Query Methods](#query-methods)
   - [Insert and Upsert Operations](#insert-and-upsert-operations)
   - [Utility Methods and Callbacks](#utility-methods-and-callbacks)
4. [Summary Table of Key Methods](#summary-table-of-key-methods)
5. [Conclusion](#conclusion)

---

## Overview of PGKVStorage

The `PGKVStorage` class is a concrete implementation of the abstract `BaseKVStorage` class. It is designed to interact with a PostgreSQL database through an abstraction called `PostgreSQLDB`. The key functionalities include:

- **Retrieval of Data:** Methods such as `get_by_id` and `get_by_ids` retrieve stored documents or cache items.
- **Filtering Keys:** The `filter_keys` method is used to check which keys do not already exist in the database.
- **Upserting Data:** The `upsert` method takes a dictionary of data and inserts or updates the corresponding records in the PostgreSQL tables.

## The Role of PGKVStorage in Lightrag

Within the Lightrag ecosystem, `PGKVStorage` supports various storage namespaces. For example:
  
- **Full Documents (`full_docs`):** Stores entire document content.
- **Text Chunks (`text_chunks`):** Stores chunks of documents.
- **LLM Response Cache (`llm_response_cache`):** Caches responses from language models.

Depending on the namespace, `PGKVStorage` uses different SQL templates to query or upsert the data.

---

## Step-by-Step Code Explanation

Below is a detailed walkthrough of the code in `PGKVStorage` with added comments and explanations.

### Initialization and Configuration

When a `PGKVStorage` instance is created, it receives a PostgreSQL database instance (`db`) and configuration details (such as namespace and global configuration). The class also sets the maximum batch size for embedding operations:

```python
@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB = None

    def __post_init__(self):
        # Set maximum number of items processed per batch for embedding generation.
        self._max_batch_size = self.global_config["embedding_batch_num"]
```

*Key Points:*
- The `__post_init__` method is called automatically after the data class is initialized.
- It reads the batch size from the global configuration to ensure consistency during embedding operations.

### Query Methods

The query methods in `PGKVStorage` are responsible for retrieving data by key or keys. There are two primary methods: `get_by_id` and `get_by_ids`.

#### `get_by_id`

This method retrieves a single record based on an ID. It selects a specific SQL template depending on the storage namespace:

```python
async def get_by_id(self, id: str) -> Union[dict, None]:
    """Get doc_full data by id."""
    # Select SQL template based on the namespace
    sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
    # Prepare parameters with workspace and id
    params = {"workspace": self.db.workspace, "id": id}
    
    if "llm_response_cache" == self.namespace:
        # If the namespace is 'llm_response_cache', multiple rows may be returned.
        array_res = await self.db.query(sql, params, multirows=True)
        res = {}
        for row in array_res:
            res[row["id"]] = row
    else:
        # For other namespaces, return a single record
        res = await self.db.query(sql, params)
    
    return res if res else None
```

*Key Points:*
- The method builds the SQL query using the namespace-specific template.
- It handles the special case for the LLM response cache, which may require assembling multiple rows into a dictionary.

#### `get_by_ids`

This method is used to retrieve multiple records at once. It constructs the SQL query by formatting a list of IDs into the SQL statement:

```python
async def get_by_ids(self, ids: List[str], fields=None) -> Union[List[dict], None]:
    """Get doc_chunks data by id"""
    # Format a list of IDs into a SQL IN clause string.
    # Each id is wrapped in single quotes and then joined with commas.
    sql = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
        ids=",".join([f"'{id}'" for id in ids])
    )
    # Set up the parameters for the query. Here, only the workspace is needed.
    params = {"workspace": self.db.workspace}
    
    # Special handling for the "llm_response_cache" namespace.
    if "llm_response_cache" == self.namespace:
        # Execute the query expecting multiple rows.
        array_res = await self.db.query(sql, params, multirows=True)
        # Create an empty set to collect unique modes.
        modes = set()
        # Create a dictionary to store results grouped by mode.
        dict_res: dict[str, dict] = {}
        # Iterate over each returned row to collect all unique modes.
        for row in array_res:
            modes.add(row["mode"])
        # Initialize the result dictionary for each unique mode.
        for mode in modes:
            if mode not in dict_res:
                dict_res[mode] = {}
        # Group the rows by mode, using the row's id as key.
        for row in array_res:
            dict_res[row["mode"]][row["id"]] = row
        # Convert the grouped dictionary into a list of dictionaries.
        res = [{k: v} for k, v in dict_res.items()]
    else:
        # For other namespaces, execute the query and expect multiple rows.
        res = await self.db.query(sql, params, multirows=True)
    
    # Return the result if it exists, otherwise return None.
    return res if res else None
```

*Key Points:*
- IDs are formatted into a string suitable for the SQL IN clause.
- As with `get_by_id`, special handling is provided for `llm_response_cache`.

### Insert and Upsert Operations

The `upsert` method in `PGKVStorage` handles the insertion or update of records. It chooses the SQL template based on the namespace of the data:

```python
async def upsert(self, data: Dict[str, dict]):
    # Determine the action based on the namespace.
    if self.namespace == "text_chunks":
        # Additional logic can be added for text chunks if needed.
        pass
    elif self.namespace == "full_docs":
        # Iterate over each document, preparing and executing an upsert SQL command.
        for k, v in data.items():
            upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
            _data = {
                "id": k,
                "content": v["content"],
                "workspace": self.db.workspace,
            }
            await self.db.execute(upsert_sql, _data)
    elif self.namespace == "llm_response_cache":
        # Process each mode and its associated items.
        for mode, items in data.items():
            for k, v in items.items():
                upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
                _data = {
                    "workspace": self.db.workspace,
                    "id": k,
                    "original_prompt": v["original_prompt"],
                    "return_value": v["return"],
                    "mode": mode,
                }
                await self.db.execute(upsert_sql, _data)
```

*Key Points:*
- Depending on the storage namespace, different SQL templates and parameter mappings are used.
- The `execute` method from the `PostgreSQLDB` class is responsible for running the query.
- For the `llm_response_cache` namespace, the data is structured by mode, and the method loops through each item accordingly.

### Utility Methods and Callbacks

The `index_done_callback` method provides a hook that can be used after indexing is completed:

```python
async def index_done_callback(self):
    if self.namespace in ["full_docs", "text_chunks"]:
        logger.info("full doc and chunk data had been saved into postgresql db!")
```

*Key Points:*
- This callback is a notification hook, logging a message when data indexing is complete.
- For some namespaces (e.g., `llm_response_cache`), a callback might not be necessary or is handled differently.

---

## Summary Table of Key Methods

| **Method**             | **Purpose**                                                   | **Key Behavior**                                                                                       |
|------------------------|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `__post_init__`        | Initialize configuration parameters                         | Sets the maximum batch size from the global configuration                                              |
| `get_by_id`            | Retrieve a single record by its ID                            | Uses namespace-specific SQL templates; handles LLM response cache differently                          |
| `get_by_ids`           | Retrieve multiple records by a list of IDs                    | Formats IDs into SQL IN clause; aggregates results by mode for `llm_response_cache`                      |
| `upsert`               | Insert or update records in the database                      | Selects SQL template based on namespace; iterates over data and executes upsert commands                |
| `index_done_callback`  | Callback hook post-indexing                                   | Logs a message indicating that data has been saved                                                     |
| `filter_keys` (inherited) | Filter out existing keys before inserting new data           | Queries the database and returns the set of keys that do not exist                                     |

---

## Conclusion

The `PGKVStorage` class is a well-structured, namespace-aware storage solution built on top of PostgreSQL. By abstracting query and upsert operations into separate methods, it provides flexibility and modularity for handling different types of document-related data in Lightrag. With clear SQL templates and configuration-driven behavior, developers can easily extend or customize this class to meet specific requirements.

The inline comments and summary tables provided in this article are designed to help you understand the step-by-step logic behind `PGKVStorage` and how it fits into the larger Lightrag framework.
