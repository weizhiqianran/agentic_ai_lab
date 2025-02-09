# Deep Dive into AIDE's `agent.py`: Understanding Prompt Engineering and Workflow

In the realm of artificial intelligence (AI) development, particularly when leveraging large language models (LLMs) like OpenAI's GPT series, crafting effective prompts is crucial for guiding model behavior and achieving desired outcomes. The `agent.py` module within the AIDE (Artificial Intelligence Development Environment) library exemplifies sophisticated prompt engineering to orchestrate AI-driven solutions effectively.

This article provides an in-depth exploration of the `agent.py` script, focusing on how prompts are constructed, utilized, and integrated into the agent's workflow. Through detailed explanations and illustrative examples, you'll gain a comprehensive understanding of the prompt engineering strategies employed in AIDE and how they contribute to the library's robust AI solutions.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Overview of `agent.py`](#overview-of-agentpy)
3. [Detailed Explanation](#detailed-explanation)
    - [Imports and Dependencies](#imports-and-dependencies)
    - [Class: `Agent`](#class-agent)
        - [Initialization](#initialization)
        - [Prompt Components](#prompt-components)
            - [`_prompt_environment`](#prompt_environment)
            - [`_prompt_impl_guideline`](#prompt_impl_guideline)
            - [`_prompt_resp_fmt`](#prompt_resp_fmt)
        - [Prompt Construction Methods](#prompt-construction-methods)
            - [`plan_and_code_query`](#plan_and_code_query)
            - [`_draft`](#_draft)
            - [`_improve`](#_improve)
            - [`_debug`](#_debug)
        - [Search Policy](#search-policy)
        - [Execution Workflow](#execution-workflow)
            - [`step`](#step)
            - [`parse_exec_result`](#parse_exec_result)
    4. [Function: `query`](#function-query)
    5. [Example Usage](#example-usage)
        - [Scenario: Iterative Model Improvement](#scenario-iterative-model-improvement)
        - [Sample Code Walkthrough](#sample-code-walkthrough)
        - [Expected Output](#expected-output)
    6. [Comparison Table](#comparison-table)
    7. [Best Practices and Key Takeaways](#best-practices-and-key-takeaways)
    8. [Conclusion](#conclusion)

---

## Introduction

Effective prompt engineering is the cornerstone of harnessing the full potential of LLMs. By meticulously crafting prompts, developers can guide AI models to produce accurate, relevant, and actionable outputs. The `agent.py` module in AIDE leverages advanced prompt engineering techniques to manage the lifecycle of AI agents, facilitating tasks such as drafting solutions, improving existing code, and debugging faulty implementations.

Understanding how prompts are constructed and utilized within `agent.py` provides valuable insights into building intelligent, autonomous agents capable of iterative improvement and problem-solving. This article dissects the prompt engineering strategies employed in `agent.py`, elucidating their roles and functionalities within the agent's workflow.

---

## Overview of `agent.py`

The `agent.py` script defines the `Agent` class, which orchestrates the interactions between the AI model and the task at hand. The agent's responsibilities include generating initial solutions, iteratively improving them, handling buggy implementations, and maintaining a journal of its actions and decisions.

Central to the agent's operations are carefully constructed prompts that guide the AI model through various stages of problem-solving. These prompts are dynamically built based on the agent's current state, the task description, and past interactions, ensuring that each step is informed by previous outcomes and aligned with the overarching objectives.

---

## Detailed Explanation

To comprehensively understand how prompts work within `agent.py`, let's dissect the module's components, focusing on prompt construction, utilization, and integration into the agent's workflow.

### Imports and Dependencies

```python
import logging
import random
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")
```

- **Standard Libraries**:
  - `logging`: Facilitates logging of events, errors, and debug information.
  - `random`: Enables random selection processes within the agent.
  - `typing`: Supports type hinting for better code clarity and error checking.
  
- **Third-Party Libraries**:
  - `humanize`: Provides human-readable representations of data, enhancing log messages.

- **Local Modules**:
  - `.backend`: Contains backend integrations and function specifications.
  - `.interpreter`: Manages the execution results of code.
  - `.journal`: Maintains a history of solutions and their evaluations.
  - `.utils`: Houses various utility functions for configuration, data preview, metric handling, and response processing.

### Class: `Agent`

The `Agent` class is the core of `agent.py`, encapsulating the logic for generating, improving, and debugging solutions.

#### Initialization

```python
class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
```

- **Parameters**:
  - `task_desc` (`str`): A description of the task the agent is to perform.
  - `cfg` (`Config`): Configuration settings loaded from `config.py`.
  - `journal` (`Journal`): Maintains a log of the agent's actions and solutions.

- **Attributes**:
  - `self.acfg`: Shortcut to the agent-specific configurations.
  - `self.data_preview`: Stores a preview of the data, utilized in prompt construction.

#### Prompt Components

The agent constructs prompts by combining various components, each serving a specific purpose in guiding the AI model.

##### `_prompt_environment`

```python
@property
def _prompt_environment(self):
    pkgs = [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "lightGBM",
        "torch",
        "torchvision",
        "torch-geometric",
        "bayesian-optimization",
        "timm",
    ]
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    env_prompt = {
        "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
    }
    return env_prompt
```

- **Purpose**: Specifies the environment in which the agent's code will run, including available packages.

- **Functionality**:
  - Defines a list of commonly used machine learning packages.
  - Shuffles the list to introduce randomness, preventing the model from relying on package order.
  - Constructs a string listing the packages in markdown inline code format.
  - Returns a dictionary with the key `"Installed Packages"` containing instructions about package usage.

- **Example Output**:

  ```json
  {
      "Installed Packages": "Your solution can use any relevant machine learning packages such as: `pandas`, `torch`, `numpy`, `scikit-learn`, ... . Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
  }
  ```

##### `_prompt_impl_guideline`

```python
@property
def _prompt_impl_guideline(self):
    impl_guideline = [
        "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
        "The code should be a single-file python program that is self-contained and can be executed as-is.",
        "No parts of the code should be skipped, don't terminate the before finishing the script.",
        "Your response should only contain a single code block.",
        f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
        'All the provided input data is stored in "./input" directory.',
        '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description. This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!**',
        'You can also use the "./working" directory to store any temporary files that your code needs to create.',
    ]
    if self.acfg.expose_prediction:
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

    if self.acfg.k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}
```

- **Purpose**: Provides detailed guidelines for implementing the solution in code.

- **Functionality**:
  - Lists a series of instructions to ensure the generated code aligns with the task requirements.
  - Conditionally adds guidelines based on configuration settings:
    - If `expose_prediction` is enabled, it instructs the inclusion of a `predict()` function.
    - If `k_fold_validation` is greater than 1, it specifies the use of k-fold cross-validation.
  - Returns a dictionary with the key `"Implementation guideline"` containing the list of guidelines.

- **Example Output**:

  ```json
  {
      "Implementation guideline": [
          "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
          "The code should be a single-file python program that is self-contained and can be executed as-is.",
          "No parts of the code should be skipped, don't terminate the before finishing the script.",
          "Your response should only contain a single code block.",
          "Be aware of the running time of the code, it should complete within 1 hour.",
          'All the provided input data is stored in "./input" directory.',
          '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description. This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!**',
          'You can also use the "./working" directory to store any temporary files that your code needs to create.',
          "The implementation should include a predict() function, allowing users to seamlessly reuse the code to make predictions on new data. The prediction function should be well-documented, especially the function signature.",
          "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
      ]
  }
  ```

##### `_prompt_resp_fmt`

```python
@property
def _prompt_resp_fmt(self):
    return {
        "Response format": (
            "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
            "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
            "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
        )
    }
```

- **Purpose**: Defines the expected format of the AI model's response.

- **Functionality**:
  - Instructs the AI to provide a concise natural language outline followed by a single markdown-formatted code block.
  - Emphasizes the absence of additional text or headings, ensuring a clean and structured response.

- **Example Output**:

  ```json
  {
      "Response format": "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block."
  }
  ```

#### Prompt Construction Methods

The `Agent` class employs several methods to construct and manage prompts tailored to different stages of solution development: drafting, improving, and debugging.

##### `plan_and_code_query`

```python
def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
    """Generate a natural language plan + code in the same LLM call and split them apart."""
    completion_text = None
    for _ in range(retries):
        completion_text = query(
            system_message=prompt,
            user_message=None,
            model=self.acfg.code.model,
            temperature=self.acfg.code.temp,
        )

        code = extract_code(completion_text)
        nl_text = extract_text_up_to_code(completion_text)

        if code and nl_text:
            # merge all code blocks into a single string
            return nl_text, code

        print("Plan + code extraction failed, retrying...")
    print("Final plan + code extraction attempt failed, giving up...")
    return "", completion_text  # type: ignore
```

- **Purpose**: Sends a constructed prompt to the AI model and retrieves both a natural language plan and corresponding code.

- **Functionality**:
  - Attempts to query the AI model up to a specified number of retries.
  - Utilizes the `query` function to send the prompt and receive the AI's response.
  - Extracts code blocks and natural language text from the response using `extract_code` and `extract_text_up_to_code`.
  - Returns a tuple containing the natural language plan and the code if both are successfully extracted.
  - If extraction fails, it retries until the maximum number of attempts is reached.

- **Example Usage**:

  ```python
  prompt = {
      "Introduction": "You are...",
      "Task description": "Predict house prices...",
      "Instructions": {...},
      ...
  }

  plan, code = agent.plan_and_code_query(prompt)
  ```

##### `_draft`

```python
def _draft(self) -> Node:
    prompt: Any = {
        "Introduction": (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        ),
        "Task description": self.task_desc,
        "Memory": self.journal.generate_summary(),
        "Instructions": {},
    }
    prompt["Instructions"] |= self._prompt_resp_fmt
    prompt["Instructions"] |= {
        "Solution sketch guideline": [
            "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
            "Take the Memory section into consideration when proposing the design,"
            " don't propose the same modelling solution but keep the evaluation the same.",
            "The solution sketch should be 3-5 sentences.",
            "Propose an evaluation metric that is reasonable for this task.",
            "Don't suggest to do EDA.",
            "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
        ],
    }
    prompt["Instructions"] |= self._prompt_impl_guideline
    prompt["Instructions"] |= self._prompt_environment

    if self.acfg.data_preview:
        prompt["Data Overview"] = self.data_preview

    plan, code = self.plan_and_code_query(prompt)
    return Node(plan=plan, code=code)
```

- **Purpose**: Initiates the creation of an initial solution by drafting a plan and corresponding code based on the task description.

- **Functionality**:
  - Constructs a prompt that includes:
    - **Introduction**: Sets the context for the AI model, positioning it as a Kaggle grandmaster.
    - **Task Description**: Provides the specific task the agent aims to solve.
    - **Memory**: Summarizes past interactions or solutions from the journal.
    - **Instructions**: Combines response format guidelines, solution sketch guidelines, implementation guidelines, and environment specifications.
  - Optionally includes a data preview if configured.
  - Utilizes `plan_and_code_query` to obtain the AI-generated plan and code.
  - Returns a `Node` object encapsulating the plan and code.

- **Example Output**:

  ```python
  Node(
      plan="To predict house prices, we will start by performing feature selection to identify the most impactful variables. "
           "Next, we'll implement a linear regression model to establish a baseline performance. "
           "Finally, we'll evaluate the model using RMSE on a hold-out validation set.",
      code="""
      def train_model():
          import pandas as pd
          from sklearn.model_selection import train_test_split
          from sklearn.linear_model import LinearRegression
          from sklearn.metrics import mean_squared_error

          # Load data
          data = pd.read_csv('./input/house_prices.csv')
          X = data.drop('SalePrice', axis=1)
          y = data['SalePrice']

          # Split data
          X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

          # Train model
          model = LinearRegression()
          model.fit(X_train, y_train)

          # Predict and evaluate
          predictions = model.predict(X_val)
          rmse = mean_squared_error(y_val, predictions, squared=False)
          print(f"RMSE: {rmse}")
      """
  )
  ```

##### `_improve`

```python
def _improve(self, parent_node: Node) -> Node:
    prompt: Any = {
        "Introduction": (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        ),
        "Task description": self.task_desc,
        "Memory": self.journal.generate_summary(),
        "Instructions": {},
    }
    prompt["Previous solution"] = {
        "Code": wrap_code(parent_node.code),
    }

    prompt["Instructions"] |= self._prompt_resp_fmt
    prompt["Instructions"] |= {
        "Solution improvement sketch guideline": [
            "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
            "You should be very specific and should only propose a single actionable improvement.",
            "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
            "Take the Memory section into consideration when proposing the improvement.",
            "The solution sketch should be 3-5 sentences.",
            "Don't suggest to do EDA.",
        ],
    }
    prompt["Instructions"] |= self._prompt_impl_guideline

    plan, code = self.plan_and_code_query(prompt)
    return Node(
        plan=plan,
        code=code,
        parent=parent_node,
    )
```

- **Purpose**: Enhances an existing solution by proposing and implementing a specific improvement.

- **Functionality**:
  - Constructs a prompt that includes:
    - **Introduction**: Instructs the AI to improve upon a provided solution.
    - **Task Description**: Reiterates the task at hand.
    - **Memory**: Summarizes past solutions and interactions.
    - **Previous Solution**: Provides the existing code wrapped in markdown.
    - **Instructions**: Combines response format guidelines, solution improvement guidelines, implementation guidelines, and environment specifications.
  - Utilizes `plan_and_code_query` to obtain the AI-generated improvement plan and code.
  - Returns a `Node` object that includes the plan, code, and a reference to the parent node.

- **Example Output**:

  ```python
  Node(
      plan="To enhance the existing linear regression model, we will introduce feature engineering by creating interaction terms between key variables. "
           "This should capture more complex relationships in the data, potentially improving model performance.",
      code="""
      def train_model():
          import pandas as pd
          from sklearn.model_selection import train_test_split
          from sklearn.linear_model import LinearRegression
          from sklearn.metrics import mean_squared_error

          # Load data
          data = pd.read_csv('./input/house_prices.csv')
          X = data.drop('SalePrice', axis=1)
          y = data['SalePrice']

          # Feature engineering
          X['OverallQual_LotArea'] = X['OverallQual'] * X['LotArea']

          # Split data
          X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

          # Train model
          model = LinearRegression()
          model.fit(X_train, y_train)

          # Predict and evaluate
          predictions = model.predict(X_val)
          rmse = mean_squared_error(y_val, predictions, squared=False)
          print(f"RMSE: {rmse}")
      """,
      parent=parent_node
  )
  ```

##### `_debug`

```python
def _debug(self, parent_node: Node) -> Node:
    prompt: Any = {
        "Introduction": (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        ),
        "Task description": self.task_desc,
        "Previous (buggy) implementation": wrap_code(parent_node.code),
        "Execution output": wrap_code(parent_node.term_out, lang=""),
        "Instructions": {},
    }
    prompt["Instructions"] |= self._prompt_resp_fmt
    prompt["Instructions"] |= {
        "Bugfix improvement sketch guideline": [
            "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
            "Don't suggest to do EDA.",
        ],
    }
    prompt["Instructions"] |= self._prompt_impl_guideline

    if self.acfg.data_preview:
        prompt["Data Overview"] = self.data_preview

    plan, code = self.plan_and_code_query(prompt)
    return Node(plan=plan, code=code, parent=parent_node)
```

- **Purpose**: Addresses and fixes bugs present in an existing solution.

- **Functionality**:
  - Constructs a prompt that includes:
    - **Introduction**: Instructs the AI to fix bugs in the provided solution.
    - **Task Description**: Reiterates the task at hand.
    - **Previous (Buggy) Implementation**: Provides the flawed code wrapped in markdown.
    - **Execution Output**: Includes the output from executing the buggy code, wrapped in markdown.
    - **Instructions**: Combines response format guidelines, bugfix guidelines, implementation guidelines, and environment specifications.
  - Optionally includes a data preview if configured.
  - Utilizes `plan_and_code_query` to obtain the AI-generated bugfix plan and code.
  - Returns a `Node` object that includes the plan, code, and a reference to the parent node.

- **Example Output**:

  ```python
  Node(
      plan="The previous implementation fails due to a missing import for `LinearRegression`. "
           "To fix this, we'll add the necessary import statement at the beginning of the script.",
      code="""
      def train_model():
          import pandas as pd
          from sklearn.model_selection import train_test_split
          from sklearn.linear_model import LinearRegression  # Added import
          from sklearn.metrics import mean_squared_error

          # Load data
          data = pd.read_csv('./input/house_prices.csv')
          X = data.drop('SalePrice', axis=1)
          y = data['SalePrice']

          # Split data
          X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

          # Train model
          model = LinearRegression()
          model.fit(X_train, y_train)

          # Predict and evaluate
          predictions = model.predict(X_val)
          rmse = mean_squared_error(y_val, predictions, squared=False)
          print(f"RMSE: {rmse}")
      """,
      parent=parent_node
  )
  ```

#### Search Policy

```python
def search_policy(self) -> Node | None:
    """Select a node to work on (or None to draft a new node)."""
    search_cfg = self.acfg.search

    # initial drafting
    if len(self.journal.draft_nodes) < search_cfg.num_drafts:
        logger.debug("[search policy] drafting new node (not enough drafts)")
        return None

    # debugging
    if random.random() < search_cfg.debug_prob:
        # nodes that are buggy + leaf nodes + debug depth < max debug depth
        debuggable_nodes = [
            n
            for n in self.journal.buggy_nodes
            if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
        ]
        if debuggable_nodes:
            logger.debug("[search policy] debugging")
            return random.choice(debuggable_nodes)
        logger.debug("[search policy] not debugging by chance")

    # back to drafting if no nodes to improve
    good_nodes = self.journal.good_nodes
    if not good_nodes:
        logger.debug("[search policy] drafting new node (no good nodes)")
        return None

    # greedy
    greedy_node = self.journal.get_best_node()
    logger.debug("[search policy] greedy node selected")
    return greedy_node
```

- **Purpose**: Determines the next action for the agent—whether to draft a new solution, improve an existing one, or debug a faulty implementation.

- **Functionality**:
  - **Initial Drafting**: If the number of draft nodes is below the configured threshold, the agent opts to draft a new solution.
  - **Debugging**: With a probability defined by `debug_prob`, the agent selects a buggy node to debug, provided it hasn't exceeded the maximum debug depth.
  - **Improvement**: If there are no nodes to debug and sufficient drafts exist, the agent selects the best existing node (greedy approach) for further improvement.
  - **Fallback**: If no good nodes are available, the agent resorts to drafting a new solution.

- **Example Scenario**:
  - **Case 1**: Few draft nodes → Agent drafts a new solution.
  - **Case 2**: High probability and presence of debuggable nodes → Agent debugs.
  - **Case 3**: Existing good nodes → Agent improves the best node.

#### Execution Workflow

The `Agent` class orchestrates the entire lifecycle of solution development through its `step` and `parse_exec_result` methods.

##### `step`

```python
def step(self, exec_callback: Callable[[str, bool], ExecutionResult]):
    if not self.journal.nodes or self.data_preview is None:
        self.update_data_preview()

    parent_node = self.search_policy()
    logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

    if parent_node is None:
        result_node = self._draft()
    elif parent_node.is_buggy:
        result_node = self._debug(parent_node)
    else:
        result_node = self._improve(parent_node)

    self.parse_exec_result(
        node=result_node,
        exec_result=exec_callback(result_node.code, True),
    )
    self.journal.append(result_node)
```

- **Purpose**: Executes a single iteration of the agent's workflow—selecting the next action, generating or improving code, executing it, and parsing the results.

- **Functionality**:
  - **Data Preview Update**: Ensures that the data preview is available before proceeding.
  - **Search Policy Application**: Determines whether to draft, improve, or debug.
  - **Action Execution**:
    - **Drafting**: Calls `_draft` to create a new solution.
    - **Debugging**: Calls `_debug` to fix a faulty solution.
    - **Improving**: Calls `_improve` to enhance an existing solution.
  - **Execution Callback**: Executes the generated code via the provided callback function.
  - **Result Parsing**: Parses the execution results to evaluate the solution.
  - **Journal Update**: Appends the new node (solution) to the journal for tracking.

- **Example Flow**:
  1. **Agent selects action** based on search policy.
  2. **Generates or improves code** using the appropriate method.
  3. **Executes the code** via `exec_callback`.
  4. **Parses and evaluates** the results.
  5. **Logs the solution** in the journal.

##### `parse_exec_result`

```python
def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
    logger.info(f"Agent is parsing execution results for node {node.id}")

    node.absorb_exec_result(exec_result)

    prompt = {
        "Introduction": (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        ),
        "Task description": self.task_desc,
        "Implementation": wrap_code(node.code),
        "Execution output": wrap_code(node.term_out, lang=""),
    }

    response = cast(
        dict,
        query(
            system_message=prompt,
            user_message=None,
            func_spec=review_func_spec,
            model=self.acfg.feedback.model,
            temperature=self.acfg.feedback.temp,
        ),
    )

    # if the metric isn't a float then fill the metric with the worst metric
    if not isinstance(response["metric"], float):
        response["metric"] = None

    node.analysis = response["summary"]
    node.is_buggy = (
        response["is_bug"]
        or node.exc_type is not None
        or response["metric"] is None
    )

    if node.is_buggy:
        node.metric = WorstMetricValue()
    else:
        node.metric = MetricValue(
            response["metric"], maximize=not response["lower_is_better"]
        )
```

- **Purpose**: Analyzes the results of code execution to determine the success of the solution and identify any bugs.

- **Functionality**:
  - **Absorbing Execution Results**: Integrates the execution output into the node.
  - **Prompt Construction**: Builds a prompt instructing the AI to evaluate the execution output and identify bugs.
    - **Introduction**: Sets the context for evaluation.
    - **Task Description**: Reiterates the task.
    - **Implementation**: Includes the executed code.
    - **Execution Output**: Provides the output from running the code.
  - **Function Specification**: Utilizes `review_func_spec` to structure the expected response, ensuring it includes necessary fields like `is_bug`, `summary`, `metric`, and `lower_is_better`.
  - **Query Execution**: Sends the prompt to the AI model and retrieves the structured response.
  - **Response Handling**:
    - Validates the `metric` value, assigning a `WorstMetricValue` if invalid.
    - Updates the node's analysis, bug status, and metric accordingly.

- **Example Response**:

  ```json
  {
      "is_bug": false,
      "summary": "The model achieved an RMSE of 0.35 on the validation set, indicating a good fit.",
      "metric": 0.35,
      "lower_is_better": true
  }
  ```

- **Outcome**:
  - **No Bug Detected**: Assigns a `MetricValue` with the reported metric.
  - **Bug Detected**: Assigns a `WorstMetricValue`, prompting the agent to debug the solution in the next iteration.

### Prompt Construction Strategy

The agent employs a hierarchical and modular approach to prompt construction, ensuring that each prompt is contextually relevant and aligned with the task objectives. By compartmentalizing different aspects of the prompt (e.g., environment, guidelines, response format), the agent maintains clarity and consistency in its interactions with the AI model.

- **Modularity**: Breaking down prompts into components like environment setup, implementation guidelines, and response formats allows for easy adjustments and scalability.
  
- **Dynamic Adaptation**: The agent tailors prompts based on the current state (e.g., drafting, improving, debugging), ensuring that each interaction is purpose-driven.
  
- **Feedback Integration**: By incorporating past solutions and their evaluations (memory), the agent builds upon previous knowledge, fostering continuous improvement.

---

## Function: `query`

The `query` function is a pivotal component within the AIDE library's backend integration. It abstracts the complexities of interacting with different large language model (LLM) backends, providing a streamlined interface for sending prompts and receiving responses.

```python
from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1-"):
        if system_message:
            user_message = system_message
        system_message = None
        model_kwargs["temperature"] = 1

    query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
```

### Explanation

- **Purpose**: The `query` function standardizes the way AIDE interacts with different LLM backends, allowing seamless integration and flexibility in model selection.

- **Parameters**:
  - `system_message` (`PromptType | None`): An optional system-level prompt that sets the context for the AI model.
  - `user_message` (`PromptType | None`): An optional user-level prompt that represents the main query or instruction.
  - `model` (`str`): The identifier for the model to use (e.g., `"gpt-4-turbo"`, `"claude-2"`).
  - `temperature` (`float | None`, optional): Controls the randomness of the model's output. Higher values like 0.8 make the output more random, while lower values like 0.2 make it more focused and deterministic.
  - `max_tokens` (`int | None`, optional): Specifies the maximum number of tokens to generate in the response.
  - `func_spec` (`FunctionSpec | None`, optional): Defines a function call structure if the backend supports function calling. If provided, the response will be a dictionary containing function call details instead of a plain string.
  - `**model_kwargs`: Additional keyword arguments to pass directly to the backend's query function.

- **Functionality**:
  1. **Parameter Consolidation**:
     
     ```python
     model_kwargs = model_kwargs | {
         "model": model,
         "temperature": temperature,
         "max_tokens": max_tokens,
     }
     ```
  
     - Merges the `model`, `temperature`, and `max_tokens` into the `model_kwargs` dictionary, ensuring they are passed to the backend correctly.
  
  2. **Handling Beta Limitations**:
     
     ```python
     if model.startswith("o1-"):
         if system_message:
             user_message = system_message
         system_message = None
         model_kwargs["temperature"] = 1
     ```
  
     - For models with names starting with `"o1-"`, which might have specific backend limitations or requirements, the function adjusts the `system_message` and sets the `temperature` to `1`. This ensures compatibility with such models.
  
  3. **Backend Selection**:
     
     ```python
     query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
     ```
  
     - Determines which backend function to use based on the model name. If the model name contains `"claude-"`, it selects the Anthropic backend; otherwise, it defaults to the OpenAI backend.
  
  4. **Prompt Compilation**:
     
     ```python
     output, req_time, in_tok_count, out_tok_count, info = query_func(
         system_message=compile_prompt_to_md(system_message) if system_message else None,
         user_message=compile_prompt_to_md(user_message) if user_message else None,
         func_spec=func_spec,
         **model_kwargs,
     )
     ```
  
     - Compiles the `system_message` and `user_message` from their structured `PromptType` formats to markdown strings using `compile_prompt_to_md`.
     - Ensures that the prompts adhere to the backend's expected input format.
     - Calls the selected backend's `query_func` with the compiled messages, function specification, and additional model parameters.
     - Receives the output along with metadata like request time and token counts.
  
  5. **Output Return**:
     
     ```python
     return output
     ```
  
     - Returns the generated output, which is either a string completion (if `func_spec` is `None`) or a dictionary containing function call details (if `func_spec` is provided).

- **Example Usage**:

  ```python
  from aide.backend import query, FunctionSpec

  # Define a function specification for function calling
  submit_review_spec = FunctionSpec(
      name="submit_review",
      json_schema={
          "type": "object",
          "properties": {
              "is_bug": {"type": "boolean"},
              "summary": {"type": "string"},
              "metric": {"type": "number"},
              "lower_is_better": {"type": "boolean"},
          },
          "required": ["is_bug", "summary", "metric", "lower_is_better"],
      },
      description="Submit a review evaluating the output of the training script.",
  )

  # Query the OpenAI backend
  response = query(
      system_message="You are an AI assistant.",
      user_message="Provide a summary of the following code.",
      model="gpt-4-turbo",
      temperature=0.5,
      max_tokens=150,
  )

  print(response)  # Outputs the generated summary as a string

  # Query the Anthropic backend with function calling
  response_with_func = query(
      system_message=None,
      user_message="Evaluate the following script.",
      model="claude-2",
      func_spec=submit_review_spec,
  )

  print(response_with_func)  # Outputs a dict with function call details
  ```

- **Output Examples**:

  - **String Completion**:

    ```plaintext
    The provided code implements a linear regression model to predict house prices based on various features. It loads the dataset, splits it into training and validation sets, trains the model, and evaluates its performance using RMSE.
    ```

  - **Function Call Details**:

    ```json
    {
        "is_bug": false,
        "summary": "The model achieved an RMSE of 0.35 on the validation set, indicating a good fit.",
        "metric": 0.35,
        "lower_is_better": true
    }
    ```

- **Advantages**:

  - **Unified Interface**: Simplifies interactions with multiple backends through a single function.
  
  - **Flexibility**: Supports both standard text completions and function calling, catering to diverse application needs.
  
  - **Configurability**: Allows customization of model parameters like temperature and max tokens, providing control over response generation.
  
  - **Extensibility**: Easily accommodates additional backends by extending the backend selection logic.

---

## Example Usage

To illustrate how prompts work within `agent.py`, let's walk through a practical scenario where the agent iteratively develops and refines a solution for predicting house prices.

### Scenario: Iterative Model Improvement

Imagine you're participating in a Kaggle competition to predict house prices. The agent's goal is to develop a robust model by iteratively drafting solutions, improving them, and debugging any issues that arise during execution.

### Sample Code Walkthrough

```python
import aide
from aide.journal import Journal
from aide.utils.config import load_cfg

# Load configuration
cfg = load_cfg()

# Initialize Journal
journal = Journal()

# Create task description
task_description = "Predict the sales price for each house based on various features."

# Initialize Agent
agent = aide.Agent(
    task_desc=task_description,
    cfg=cfg,
    journal=journal,
)

# Define an execution callback function
def execute_code(code: str, run: bool) -> aide.interpreter.ExecutionResult:
    # Execute the code and capture the output
    # This is a placeholder; implementation will depend on your execution environment
    import subprocess
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=cfg.exec.timeout,
        )
        return aide.interpreter.ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            exc_type=None if result.returncode == 0 else "ExecutionError",
        )
    except subprocess.TimeoutExpired as e:
        return aide.interpreter.ExecutionResult(
            stdout=e.stdout or "",
            stderr=e.stderr or "Execution timed out.",
            returncode=-1,
            exc_type="TimeoutExpired",
        )

# Run the agent for a specified number of steps
best_solution = None
for _ in range(10):
    agent.step(exec_callback=execute_code)
    # Optionally, retrieve the best solution so far
    best_solution = journal.get_best_node()

if best_solution:
    print(f"Best solution has validation metric: {best_solution.metric}")
    print(f"Best solution code:\n{best_solution.code}")
```

- **Initialization**:
  - **Configuration Loading**: Loads settings from `config.yaml` and command-line arguments using `load_cfg()`.
  - **Journal Initialization**: Prepares a journal to log solutions and evaluations.
  - **Agent Initialization**: Sets up the agent with the task description, configuration, and journal.

- **Execution Loop**:
  - **Agent Steps**: Iteratively calls the `step` method, allowing the agent to:
    - **Select Action**: Decide whether to draft, improve, or debug a solution.
    - **Generate Solution**: Use prompts to generate a plan and code.
    - **Execute Code**: Run the generated code via `exec_callback`.
    - **Parse Results**: Analyze execution outcomes to assess solution quality.

### Expected Output

After running the agent for several steps, the journal will contain a history of solutions, their evaluations, and any identified bugs. Here's an example of what a node in the journal might look like:

```python
Node(
    id=1,
    plan="To predict house prices, we'll start by performing feature selection to identify the most impactful variables. "
         "Next, we'll implement a linear regression model to establish a baseline performance. "
         "Finally, we'll evaluate the model using RMSE on a hold-out validation set.",
    code="""
    def train_model():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        # Load data
        data = pd.read_csv('./input/house_prices.csv')
        X = data.drop('SalePrice', axis=1)
        y = data['SalePrice']

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        print(f"RMSE: {rmse}")
    """,
    term_out="RMSE: 0.35",
    analysis="The model achieved an RMSE of 0.35 on the validation set, indicating a good fit.",
    is_buggy=False,
    metric=MetricValue(value=0.35, maximize=False)
)
```

- **Node Attributes**:
  - `id`: Unique identifier for the solution.
  - `plan`: Natural language outline of the proposed solution.
  - `code`: Generated Python code implementing the solution.
  - `term_out`: Output from executing the code (e.g., evaluation metric).
  - `analysis`: Summary of the execution results.
  - `is_buggy`: Boolean indicating if the solution has bugs.
  - `metric`: Represents the evaluation metric value.

- **Next Steps**:
  - If `is_buggy` is `False`, the agent may proceed to improve the solution further.
  - If `is_buggy` is `True`, the agent will initiate debugging in the next step.

---

## Comparison Table

To contextualize AIDE's prompt engineering strategies within the broader landscape of AI development tools, let's compare `agent.py` with standard prompt construction approaches.

### Comparing AIDE's Prompt Engineering with Standard Approaches

| **Feature**                           | **AIDE's `agent.py`**                                                | **Standard Prompt Construction**                          |
|---------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------|
| **Modular Prompt Components**         | Utilizes property-based modular components (`_prompt_environment`, `_prompt_impl_guideline`, `_prompt_resp_fmt`) | Typically uses monolithic prompts without modularity     |
| **Dynamic Adaptation**                | Adjusts prompts based on agent's state (drafting, improving, debugging) | Static prompts tailored per use-case without state awareness |
| **Structured Formatting**             | Constructs prompts as dictionaries, allowing for organized and nested structures | Often constructs prompts as plain strings, leading to less structure |
| **Integration with Configuration**    | Leverages configuration settings (`self.cfg`) to customize prompt components | Requires manual adjustments to prompts based on configurations |
| **Feedback Loop Integration**         | Incorporates feedback from execution results to inform future prompts | Limited or no integration of feedback into prompt adjustments |
| **Guideline Enforcement**            | Embeds detailed guidelines within prompts to ensure desired output formats and content | Relies on the model's interpretation of less detailed instructions |
| **Response Parsing**                  | Utilizes function specifications (`FunctionSpec`) to parse and validate responses | Typically relies on manual parsing or basic regex without validation |
| **Iterative Improvement**             | Facilitates iterative drafting and refining through systematic prompts | Lacks built-in mechanisms for iterative prompt-based improvements |
| **Environment Specification**         | Clearly defines available packages and environment constraints within prompts | May omit detailed environment specifications, leading to potential inconsistencies |
| **Automated Error Handling**         | Detects and responds to bugs through specialized prompts and `WorstMetricValue` | Requires manual detection and handling of errors without prompt-based strategies |

**Key Observations**:

- **Modularity and Structure**: AIDE's approach emphasizes modularity and structured prompt construction, enhancing clarity and maintainability.
  
- **Dynamic and Adaptive**: The agent dynamically adapts prompts based on its current state, fostering a responsive and intelligent workflow.
  
- **Guideline Enforcement**: Embedding detailed guidelines ensures that the AI model adheres to expected response formats and content requirements.
  
- **Feedback Integration**: By incorporating execution results into future prompts, AIDE establishes a feedback loop that drives continuous improvement.

---

## Best Practices and Key Takeaways

Leveraging the `agent.py` module's prompt engineering strategies can significantly enhance the effectiveness and reliability of AI-driven solutions. Here are some best practices and key takeaways:

1. **Modular Prompt Design**:
   - **Separation of Concerns**: Break down prompts into distinct components (environment, guidelines, response formats) to enhance clarity and manageability.
     ```python
     prompt["Instructions"] |= self._prompt_resp_fmt
     prompt["Instructions"] |= self._prompt_impl_guideline
     prompt["Instructions"] |= self._prompt_environment
     ```
   
2. **Dynamic Adaptation Based on State**:
   - **Contextual Prompts**: Tailor prompts based on whether the agent is drafting, improving, or debugging, ensuring relevance and specificity.
     ```python
     if parent_node is None:
         result_node = self._draft()
     elif parent_node.is_buggy:
         result_node = self._debug(parent_node)
     else:
         result_node = self._improve(parent_node)
     ```
   
3. **Incorporate Detailed Guidelines**:
   - **Clarity and Precision**: Provide explicit instructions within prompts to guide the AI model towards desired outputs and behaviors.
     ```python
     "Solution sketch guideline": [
         "This first solution design should be relatively simple...",
         ...
     ],
     ```
   
4. **Leverage Configuration Settings**:
   - **Customization**: Utilize configuration files to adjust prompt components dynamically, allowing for flexible and adaptable prompt engineering.
     ```python
     if self.acfg.expose_prediction:
         impl_guideline.append("The implementation should include a predict() function...")
     ```
   
5. **Implement Feedback Loops**:
   - **Continuous Improvement**: Use execution results to inform and refine future prompts, fostering an iterative enhancement process.
     ```python
     response = cast(
         dict,
         query(
             system_message=prompt,
             user_message=None,
             func_spec=review_func_spec,
             model=self.acfg.feedback.model,
             temperature=self.acfg.feedback.temp,
         ),
     )
     ```
   
6. **Ensure Response Validation**:
   - **Robustness**: Validate the AI model's responses to prevent the incorporation of invalid or erroneous data into the agent's workflow.
     ```python
     if not isinstance(response["metric"], float):
         response["metric"] = None
     ```
   
7. **Utilize Environment Specifications**:
   - **Consistency**: Clearly define the available packages and environment constraints within prompts to ensure that generated code is executable and compatible.
     ```python
     env_prompt = {
         "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}..."
     }
     ```

**Key Takeaways**:

- **Structured and Modular Prompts**: Organizing prompts into distinct, reusable components enhances maintainability and clarity.
  
- **Adaptive Prompting**: Adjusting prompts based on the agent's current state ensures that interactions remain relevant and purposeful.
  
- **Feedback-Driven Iteration**: Integrating execution results into future prompts creates a feedback loop that drives continuous improvement and refinement of solutions.
  
- **Explicit Guidelines**: Providing detailed instructions within prompts steers the AI model towards producing consistent and accurate outputs.

---

## Conclusion

The `agent.py` module within the AIDE library showcases sophisticated prompt engineering techniques that are pivotal for orchestrating AI-driven solutions. By meticulously constructing and managing prompts, the agent ensures that each step of the solution development process is guided, validated, and iteratively improved.

Understanding the intricacies of prompt construction in `agent.py` provides valuable insights into effective AI model interaction, enabling developers to harness the full potential of LLMs within their projects. Whether it's drafting initial solutions, enhancing existing code, or debugging faulty implementations, AIDE's agent employs a robust and dynamic prompt engineering strategy that fosters reliable and high-performing AI solutions.

Embrace the strategies elucidated in this article to elevate your AI development workflows, ensuring that your interactions with language models are both efficient and effective. For further exploration and advanced configurations, refer to the official AIDE documentation and engage with the community to share experiences and insights.

---