# Introducing PGGraphStorage: A PostgreSQL Graph Storage Implementation for Lightrag

**Table of Contents**

1. [Overview](#overview)
2. [Role of PGGraphStorage in Lightrag](#role-of-pggraphstorage-in-lightrag)
3. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)  
   3.1. [Class Initialization and Setup](#class-initialization-and-setup)  
   3.2. [Graph Query Execution and Record Conversion](#graph-query-execution-and-record-conversion)  
   3.3. [Node and Edge Operations](#node-and-edge-operations)  
   3.4. [Error Handling and Retrying](#error-handling-and-retrying)
4. [Summary Table of Key Methods](#summary-table-of-key-methods)
5. [Conclusion](#conclusion)

---

## Overview

`PGGraphStorage` is a PostgreSQL-based implementation of the abstract `BaseGraphStorage` class within the Lightrag framework. This storage class leverages the Apache AGE extension for PostgreSQL to manage graph data, allowing you to perform graph queries, insert nodes and edges, and even execute node embedding algorithms (although some algorithms may be placeholders).

In this article, we break down the code of `PGGraphStorage` step by step, explain the purpose of each section, and provide inline comments and summary tables for clarity.

---

## Role of PGGraphStorage in Lightrag

Within Lightrag, graph storage is essential for managing relationships and entities extracted from documents. `PGGraphStorage` allows you to:
- Execute graph queries using a Cypher-like language.
- Upsert (insert/update) nodes and edges.
- Query node and edge properties.
- Handle errors gracefully with retries.

This functionality makes it a robust choice for knowledge graph applications integrated into the Lightrag ecosystem.

---

## Step-by-Step Code Walkthrough

### 3.1 Class Initialization and Setup

The `PGGraphStorage` class inherits from `BaseGraphStorage` and is initialized with the necessary parameters: namespace, global configuration, and an embedding function. The constructor also sets up the graph name using an environment variable.

```python
@dataclass
class PGGraphStorage(BaseGraphStorage):
    db: PostgreSQLDB = None  # PostgreSQL database instance

    # Static method for preloading a networkx graph (not used in production)
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with AGE in production")

    def __init__(self, namespace, global_config, embedding_func):
        # Initialize the base graph storage with provided parameters
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        # Set the graph name from the environment variable
        self.graph_name = os.environ["AGE_GRAPH_NAME"]
        # Define available node embedding algorithms; currently only node2vec is referenced
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
```

*Key Points:*
- The class inherits from `BaseGraphStorage`, receiving standard configuration.
- `self.graph_name` is set from an environment variable (`AGE_GRAPH_NAME`).
- The `_node_embed_algorithms` dictionary allows for potential expansion of node embedding options.

---

### 3.2 Graph Query Execution and Record Conversion

The class provides a private method `_query` that executes graph queries. This method converts a Cypher-like query into an AGE-compatible PostgreSQL query, handles errors, and converts results to Python dictionaries.

```python
async def _query(self, query: str, readonly: bool = True, upsert: bool = False) -> List[Dict[str, Any]]:
    """
    Execute a graph query and convert the result records to dictionaries.

    Args:
        query (str): The Cypher-like query to execute.
        readonly (bool): Determines if the query is read-only.
        upsert (bool): Flag for upsert operations.

    Returns:
        List[Dict[str, Any]]: List of results as dictionaries.
    """
    # Use the wrapped query directly (could be modified for AGE-specific conversion)
    wrapped_query = query

    try:
        if readonly:
            # Execute a read-only query with AGE prerequisites
            data = await self.db.query(
                wrapped_query,
                multirows=True,
                for_age=True,
                graph_name=self.graph_name,
            )
        else:
            # Execute an upsert or write query
            data = await self.db.execute(
                wrapped_query,
                for_age=True,
                graph_name=self.graph_name,
                upsert=upsert,
            )
    except Exception as e:
        raise PGGraphQueryException({
            "message": f"Error executing graph query: {query}",
            "wrapped": wrapped_query,
            "detail": str(e),
        }) from e

    if data is None:
        result = []
    else:
        # Convert each record using the helper method _record_to_dict
        result = [PGGraphStorage._record_to_dict(d) for d in data]

    return result
```

*Key Points:*
- The `_query` method centralizes query execution and error handling.
- It calls the `query` or `execute` methods from the PostgreSQLDB instance.
- Results are passed through `_record_to_dict` to convert AGE data types into Python types.

#### Record Conversion Helper: `_record_to_dict`

This helper method processes the raw `asyncpg.Record` objects, parsing AGTYPE strings for vertices and edges.

```python
@staticmethod
def _record_to_dict(record: asyncpg.Record) -> Dict[str, Any]:
    """
    Convert an AGE query record into a Python dictionary.

    Args:
        record (asyncpg.Record): Record from a graph query.

    Returns:
        Dict[str, Any]: Converted dictionary with proper types.
    """
    d = {}  # Result holder

    # Build a mapping from vertex IDs to their properties
    vertices = {}
    for k in record.keys():
        v = record[k]
        if isinstance(v, str) and "::" in v:
            dtype = v.split("::")[-1]
            v = v.split("::")[0]
            if dtype == "vertex":
                vertex = json.loads(v)
                vertices[vertex["id"]] = vertex.get("properties")

    # Iterate over each field in the record and parse values
    for k in record.keys():
        v = record[k]
        if isinstance(v, str) and "::" in v:
            dtype = v.split("::")[-1]
            v = v.split("::")[0]
        else:
            dtype = ""

        if dtype == "vertex":
            vertex = json.loads(v)
            field = vertex.get("properties") or {}
            field["label"] = PGGraphStorage._decode_graph_label(field["node_id"])
            d[k] = field
        elif dtype == "edge":
            # For edges, return a tuple containing source properties, edge label, and target properties
            edge = json.loads(v)
            d[k] = (
                vertices.get(edge["start_id"], {}),
                edge["label"],
                vertices.get(edge["end_id"], {}),
            )
        else:
            d[k] = json.loads(v) if isinstance(v, str) else v

    return d
```

*Key Points:*
- The method distinguishes between vertex and edge data.
- For vertices, it decodes the label using `_decode_graph_label`.
- For edges, it returns a tuple with source, label, and target.

---

### 3.3 Node and Edge Operations

`PGGraphStorage` includes methods to check existence, upsert, and retrieve nodes and edges.

#### Checking Node and Edge Existence

```python
async def has_node(self, node_id: str) -> bool:
    # Encode the node_id for AGE compatibility
    entity_name_label = PGGraphStorage._encode_graph_label(node_id.strip('"'))

    query = """SELECT * FROM cypher('%s', $$
                 MATCH (n:Entity {node_id: "%s"})
                 RETURN count(n) > 0 AS node_exists
               $$) AS (node_exists bool)""" % (self.graph_name, entity_name_label)

    single_result = (await self._query(query))[0]
    logger.debug(
        "{%s}:query:%s:result:%s",
        inspect.currentframe().f_code.co_name,
        query,
        single_result["node_exists"],
    )

    return single_result["node_exists"]

async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
    src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
    tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))

    query = """SELECT * FROM cypher('%s', $$
                 MATCH (a:Entity {node_id: "%s"})-[r]-(b:Entity {node_id: "%s"})
                 RETURN COUNT(r) > 0 AS edge_exists
               $$) AS (edge_exists bool)""" % (self.graph_name, src_label, tgt_label)

    single_result = (await self._query(query))[0]
    logger.debug(
        "{%s}:query:%s:result:%s",
        inspect.currentframe().f_code.co_name,
        query,
        single_result["edge_exists"],
    )
    return single_result["edge_exists"]
```

*Key Points:*
- Methods `has_node` and `has_edge` encode identifiers and execute queries to verify existence.
- They log debug information for troubleshooting.

#### Upserting Nodes and Edges

Upsert methods ensure that nodes or edges are inserted or updated in the AGE graph.

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type((PGGraphQueryException,)))
async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
    """
    Upsert a node in the AGE database.

    Args:
        node_id (str): Unique identifier for the node.
        node_data (Dict[str, Any]): Node properties.
    """
    label = PGGraphStorage._encode_graph_label(node_id.strip('"'))
    properties = node_data

    query = """SELECT * FROM cypher('%s', $$
                 MERGE (n:Entity {node_id: "%s"})
                 SET n += %s
                 RETURN n
               $$) AS (n agtype)""" % (
        self.graph_name,
        label,
        PGGraphStorage._format_properties(properties),
    )

    try:
        await self._query(query, readonly=False, upsert=True)
        logger.debug("Upserted node with label '%s' and properties: %s", label, properties)
    except Exception as e:
        logger.error("Error during upsert: %s", e)
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type((PGGraphQueryException,)))
async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]):
    """
    Upsert an edge between two nodes.

    Args:
        source_node_id (str): Identifier of the source node.
        target_node_id (str): Identifier of the target node.
        edge_data (Dict[str, Any]): Edge properties.
    """
    src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
    tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))
    edge_properties = edge_data

    query = """SELECT * FROM cypher('%s', $$
                 MATCH (source:Entity {node_id: "%s"})
                 WITH source
                 MATCH (target:Entity {node_id: "%s"})
                 MERGE (source)-[r:DIRECTED]->(target)
                 SET r += %s
                 RETURN r
               $$) AS (r agtype)""" % (
        self.graph_name,
        src_label,
        tgt_label,
        PGGraphStorage._format_properties(edge_properties),
    )
    try:
        await self._query(query, readonly=False, upsert=True)
        logger.debug("Upserted edge from '%s' to '%s' with properties: %s", src_label, tgt_label, edge_properties)
    except Exception as e:
        logger.error("Error during edge upsert: %s", e)
        raise
```

*Key Points:*
- Both upsert methods use the `MERGE` clause to create or update graph elements.
- They include retry logic to handle transient errors.
- The helper `_format_properties` method is used to format node/edge properties for the query.

---

### 3.4 Error Handling and Retrying

The class employs the `tenacity` library to retry operations in case of failures (e.g., transient network issues). The decorators on `upsert_node` and `upsert_edge` ensure that the query is retried up to 3 times with exponential backoff.

*Key Points:*
- Retries help increase reliability in distributed database environments.
- Specific exception types (`PGGraphQueryException`) trigger the retry logic.

---

## Summary Table of Key Methods

| **Method**               | **Purpose**                                                     | **Key Features**                                                                     |
|--------------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `__init__`               | Initializes PGGraphStorage with namespace, config, embedding_func | Sets graph name; initializes embedding algorithm mapping                             |
| `_query`                 | Executes a graph query and converts results to dictionaries      | Handles AGE-specific prerequisites; error handling and conversion via `_record_to_dict` |
| `_record_to_dict`        | Converts raw AGE query records to Python dictionaries            | Parses vertices and edges; decodes graph labels using `_decode_graph_label`          |
| `has_node`               | Checks if a node exists in the graph                             | Encodes node ID; returns boolean                                                     |
| `has_edge`               | Checks if an edge exists between two nodes                       | Encodes node IDs; returns boolean                                                    |
| `upsert_node`            | Inserts or updates a node in the graph                           | Uses MERGE in Cypher query; includes retry logic                                     |
| `upsert_edge`            | Inserts or updates an edge between nodes                         | Uses MERGE in Cypher query; includes retry logic                                     |

---

## Conclusion

`PGGraphStorage` provides a comprehensive PostgreSQL-based graph storage solution for Lightrag, leveraging Apache AGE. Through its well-structured methods for query execution, record conversion, and node/edge upsert operations, it enables efficient graph management. The use of retry logic and helper functions ensures robustness and maintainability.

This article has walked through the code step by step, highlighting key functionalities and offering detailed inline comments to facilitate understanding and further development.
