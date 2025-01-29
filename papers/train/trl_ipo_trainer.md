# Understanding IPO: A New Approach to Learning from Human Preferences

## Table of Contents
1. [Introduction](#introduction)
2. [Learning from Human Preferences](#learning-from-human-preferences)
3. [Challenges with Existing Methods](#challenges-with-existing-methods)
4. [Introducing IPO](#introducing-ipo)
5. [Breaking Down Objective Function](#breaking-down-objective-function)
6. [Theoretical Foundations of IPO](#theoretical-foundations-of-ipo)
7. [Implementing IPO with DPOTrainer](#implementing-ipo-with-dpotrainer)
8. [Conclusion and Future Work](#conclusion-and-future-work)
9. [References](#references)

## Introduction

In the realm of machine learning, aligning models with human preferences is crucial for creating systems that behave as desired. Traditional methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimisation (DPO) have paved the way in this area. However, they come with their own set of challenges, notably overfitting and reliance on specific assumptions. Enter **Identity Preference Optimisation (IPO)**, a novel approach that aims to overcome these limitations by directly learning from human preferences without the need for intermediate reward modeling. This article delves into the intricacies of IPO, comparing it with existing methods, and highlighting its theoretical and empirical benefits.

## Learning from Human Preferences

### What is IPO?

**Identity Preference Optimisation (IPO)** is a method designed to learn policies directly from human preference data. Unlike traditional approaches that rely on intermediary steps like reward modeling, IPO aims to optimize the policy by directly considering pairwise human preferences. This direct approach helps in mitigating issues like overfitting and ensures that the learned policy remains close to a reference policy, enhancing its generalizability.

### Traditional Methods

#### Reinforcement Learning from Human Feedback (RLHF)

RLHF is a widely adopted paradigm in natural language processing and other areas. It involves two main stages:

1. **Learning the Reward Model:** A binary classifier is trained to distinguish between preferred and dispreferred actions based on human feedback.
2. **Policy Optimisation:** Using the learned reward model, reinforcement learning techniques like Proximal Policy Optimization (PPO) are employed to optimize the policy.

**Example:** Imagine training a chatbot. RLHF would first involve collecting pairs of responses, labeling them as preferred or not, training a reward model on these labels, and then using RL to fine-tune the chatbot to generate preferred responses.

#### Direct Preference Optimisation (DPO)

DPO offers an alternative by eliminating the need for a separate reward model. Instead, it directly optimizes the policy based on the observed preferences in the dataset.

**Example:** Continuing with the chatbot scenario, DPO would directly adjust the chatbot's responses to align with the preferred responses without training a separate model to predict preference scores.

---

## Challenges with Existing Methods

### Overfitting Issues

Both RLHF and DPO can suffer from overfitting, especially when dealing with deterministic or nearly deterministic preferences. Overfitting occurs when the model becomes too tailored to the training data, performing poorly on unseen data.

**Table 1: Overfitting in RLHF vs. DPO**

| Method | Overfitting Risk | Description |
|--------|-------------------|-------------|
| RLHF   | Moderate          | Overfitting can occur if the reward model becomes too specialized. However, regularisation techniques like KL divergence help mitigate this. |
| DPO    | High               | Without an explicit reward model, DPO may overfit by assigning extreme probabilities to preferred actions, ignoring the reference policy. |

### Assumptions and Limitations

RLHF relies on the **Bradley-Terry (BT)** model, assuming that pairwise preferences can be transformed into pointwise rewards. This assumption may not hold in all scenarios, leading to potential inaccuracies. DPO, while simplifying the process by removing the reward model, still relies on this assumption and is more prone to overfitting.

---

#### Understanding the Bradley-Terry (BT) Model Assumption

The **Bradley-Terry (BT)** model is a mathematical framework used to convert **pairwise preferences** into **pointwise rewards**. It assumes that if a person prefers **option A over option B**, then we can assign **numerical scores** (or "Elo-like rewards") to A and B such that:

$$
P(A \succ B) = \frac{e^{r(A)}}{e^{r(A)} + e^{r(B)}}
$$

where:
- $ P(A \succ B) $ is the probability of A being preferred over B.
- $ r(A) $ and $ r(B) $ are numerical rewards assigned to A and B.

#### **Example: Movie Ratings**
Imagine you are training an AI to **recommend movies** based on human preferences. You collect pairwise comparisons where users express preferences between two movies:

| Movie A | Movie B | Preferred Movie |
|---------|---------|----------------|
| Inception | Titanic | Inception |
| Titanic | The Godfather | The Godfather |
| Inception | The Godfather | The Godfather |

Using the BT model, we attempt to assign numerical scores:

- **Inception** has a certain Elo-score.
- **Titanic** has a different Elo-score.
- **The Godfather** has the highest score.

Now, RLHF assumes that these **pairwise preferences can be transformed into a numerical reward model**. However, this assumption **may not always hold**.

#### **Why the Assumption Fails?**

1. **Preferences are Contextual, Not Absolute**  
   - A user may prefer **Titanic over Inception** when watching with family, but prefer **Inception over Titanic** when watching alone.  
   - The BT model forces a single global ranking, which may **oversimplify the complexity** of human preferences.

2. **Loss of Fine-Grained Information**  
   - Suppose a user slightly prefers **Inception over Titanic** but strongly prefers **The Godfather over both**.  
   - The BT model does not differentiate between "strong" and "weak" preferences; it only learns a **numerical gap** between movies.

3. **Overfitting in DPO**  
   - DPO avoids training a separate reward model but still **relies on the BT assumption**.  
   - If the dataset contains **very few preference samples**, DPO may overfit by assigning extreme probabilities (e.g., **always selecting The Godfather** and ignoring other movies).  
   - This makes the policy **too rigid** and unable to generalize to new movies.

---

## Introducing IPO

### The ΨPO Objective

IPO introduces a generalized objective known as **Ψ-preference optimisation (ΨPO)**. The ΨPO framework allows for the optimization of policies based solely on pairwise preferences, bypassing the need for pointwise rewards.

The ΨPO objective function can be rewritten conceptually as:

$$
\text{Optimized Policy} = \text{Maximize Preference Satisfaction} - \tau \times \text{Regularisation Penalty}
$$

Where:
- $\tau$ is the regularisation parameter.

### How IPO Differs from RLHF and DPO

IPO stands out by directly optimizing the policy based on pairwise preferences without the intermediary reward model. This direct approach allows IPO to avoid the overfitting issues inherent in DPO and the reliance on the BT model in RLHF.

**Figure 1: Comparison of Methods**

| Method | Reward Model | Direct Preference Optimization | Regularisation |
|--------|--------------|-------------------------------|-----------------|
| RLHF   | Yes          | No                            | Yes             |
| DPO    | No           | Yes                           | Limited         |
| IPO    | No           | Yes                           | Yes             |


### Advantages of IPO

#### Simplicity

IPO offers a straightforward approach by eliminating the need for an intermediate reward model. This simplification not only reduces computational overhead but also streamlines the training process.

#### Regularisation

By incorporating KL regularisation directly into the optimization objective, IPO ensures that the learned policy remains close to a reference policy. This regularisation is crucial for preventing overfitting and maintaining the generalizability of the policy.

---

## Breaking Down Objective Function 

The ΨPO (Psi-Preference Optimisation) objective function is a **generalized formulation** for learning from human preferences. It allows direct preference optimization without requiring a reward model. Let’s break it down into its **core components** for better understanding.

### **ΨPO Objective Function**
$$
\max_{\pi} \mathbb{E}_{x \sim \rho, y \sim \pi(.|x), y' \sim \mu(.|x)} [\Psi(p^*(y \succ y'|x))] - \tau D_{KL}(\pi || \pi_{ref})
$$

Where:
- $ x $ is the **context (input prompt)**, such as a prompt or environment state, representing different situations where decisions are made.
- $ y $ is the **chosen action (chosen response)** from the policy $ \pi $ given the context $ x $.
- $ y' $ is the **alternative action (rejected response)** from the behavior policy $ \mu $ given the context $ x $.
- $ \mathbb{E} $ denotes the **expectation operator**, which computes the average value of a function over a probability distribution.
- $ x \sim \rho $ is the **context**, sampled from a distribution $ \rho $, representing different situations or prompts where decisions are made.
- $ y \sim \pi(.|x) $ is an **action** sampled from the learned policy $ \pi $, given context $ x $.
- $ y' \sim \mu(.|x) $ is an **alternative action** sampled from the behavior policy $ \mu $, which serves as a reference for preference comparison.
- $ \Psi $ is a **non-decreasing function** that transforms the preference probability $ p^*(y \succ y'|x) $ into a form suitable for optimization.
- $ \pi $ is the **policy being optimized (policy model)**, which maps a given context $ x $ to a probability distribution over actions.
- $ \pi_{ref} $ is the **reference policy (reference model)**, a pre-existing or base policy that guides optimization and helps prevent overfitting.
- $ \mu $ is the **behavior policy**, representing the policy that originally generated the actions in the dataset.
- $ p^*(y \succ y'|x) $ is the **true human preference probability**, representing the likelihood that action $ y $ is preferred over action $ y' $ given the context $ x $.
- $ D_{KL}(\pi || \pi_{ref}) $ is the **Kullback-Leibler divergence**, a measure of how much the optimized policy $ \pi $ deviates from a reference policy $ \pi_{ref} $.
- $ \tau $ is the **regularisation parameter**, controlling the trade-off between preference satisfaction and staying close to the reference policy.

The function can be understood as a **trade-off** between two terms:  
1. **Maximizing preference satisfaction**  
2. **Minimizing divergence from a reference policy**

### **1. Expected Preference Satisfaction Term**
$$
\mathbb{E}_{x \sim \rho, y \sim \pi(.|x), y' \sim \mu(.|x)} [\Psi(p^*(y \succ y'|x))]
$$
This term ensures that the policy $\pi$ **prioritizes actions that align with human preferences**.

- **Expectation over $ x \sim \rho $:**  
  - The expectation is taken over **context $ x $**, representing real-world situations.
- **Action Selection:**
  - $ y \sim \pi(.|x) $ → The policy selects an action $ y $ based on the given context.
  - $ y' \sim \mu(.|x) $ → The behavior policy $ \mu $ selects a competing action $ y' $.
- **Pairwise Preference Evaluation:**  
  - $ p^*(y \succ y'|x) $ represents the probability that action $ y $ is preferred over $ y' $.
  - The function $ \Psi $ is applied to this probability to adjust its impact on optimization.

💡 **Intuition:** The model learns by comparing **actions in pairs** and prioritizing those that are consistently preferred.

### **2. KL Divergence Regularisation Term**
$$
- \tau D_{KL}(\pi || \pi_{ref})
$$
This term **ensures that the learned policy $\pi$ does not drift too far from a reference policy $\pi_{ref}$**.

- **KL Divergence Definition:**
  $$
  D_{KL}(\pi || \pi_{ref}) = \sum_{x} \pi(x) \log \frac{\pi(x)}{\pi_{ref}(x)}
  $$
  This measures how much $\pi$ differs from $\pi_{ref}$. A higher value means $\pi$ is deviating significantly.
  
- **Regularisation Parameter ($\tau$):**
  - Controls the balance between learning from preferences and staying close to $\pi_{ref}$.
  - A **high** $\tau$ encourages the policy to stay closer to $\pi_{ref}$, reducing overfitting.
  - A **low** $\tau$ allows more deviation, making the policy more adaptive.

💡 **Intuition:** Regularisation prevents overfitting by ensuring that the optimized policy does not stray too far from known behavior.

### **Final Takeaway**
The ΨPO objective function can be rewritten conceptually as:

$$
\text{Optimized Policy} = \text{Maximize Preference Satisfaction} - \tau \times \text{Regularisation Penalty}
$$

- **The first term ensures that the model prioritizes preferred actions.**  
- **The second term ensures that the model does not drift too far from the reference policy.**  

---

## Theoretical Foundations of IPO

### A Unifying Framework for Learning from Preferences

IPO’s ΨPO objective provides a **generalized framework** that unifies two key methods:  
1. **Reinforcement Learning from Human Feedback (RLHF)**  
2. **Direct Preference Optimisation (DPO)**  

By selecting different forms of the function $ \Psi $, IPO can **mimic the behavior** of these traditional approaches or introduce new optimization techniques.

#### **Example: How IPO Unifies RLHF and DPO**
Consider a scenario where we are training an AI assistant to rank **search engine results** based on human feedback. Users provide **pairwise preferences**, choosing which of two results is more relevant.  
- **RLHF Approach**:  
  - The system first trains a **reward model** using a classifier that predicts a score for each result.  
  - Then, the policy is optimized using reinforcement learning to maximize this reward while staying close to the original model.
- **DPO Approach**:  
  - Instead of learning a reward model, DPO directly **adjusts the policy** to prefer actions that align with human preferences.

**How IPO Bridges the Gap**  
IPO provides a single framework where the behavior of RLHF and DPO can be **derived as special cases**. Specifically:
- When $ \Psi(q) = \log \left( \frac{q}{1 - q} \right) $, IPO **reduces to RLHF and DPO under the Bradley-Terry model assumption**.
- $ q $ represents the **true human preference probability**, denoted as: $ q = p^*(y \succ y' | x) $

where:
- $ p^*(y \succ y' | x) $ is the **probability that action $ y $ is preferred over action $ y' $**, given the context $ x $.
- This probability is estimated from human preference data.

This means IPO **flexibly adapts** to different learning strategies based on the choice of $ \Psi $, offering a more generalized approach.

---

### Preventing Overfitting with IPO

One of the **key advantages** of IPO is its ability to prevent **overfitting** by making a **crucial adjustment**:  
- Instead of using a function like $ \log \left( \frac{q}{1 - q} \right) $, IPO **chooses a simple identity function** $ \Psi(q) = q $.  
- This prevents the optimization process from exaggerating small differences in preference scores.

#### **Example: Overfitting in Search Engine Ranking**
Imagine we are optimizing a search engine ranking model:  
- The model compares two search results, **Result A and Result B**, and human feedback prefers A over B 100% of the time.
- In DPO, the policy **overfits** by always choosing A and completely ignoring B.
- In IPO, regularisation ensures the model **does not completely discard** Result B, preventing **rigid and brittle decisions**.

**Mathematical Formulation**  
IPO modifies the ΨPO objective as:

$$
\max_{\pi} \mathbb{E}_{x \sim \rho, y \sim \pi(.|x), y' \sim \mu(.|x)} [p^*(y \succ y'|x)] - \tau D_{KL}(\pi || \pi_{ref})
$$

Here:
- The **first term** encourages the policy to prefer **higher-ranked actions**.
- The **second term** ensures the policy **does not diverge too far from a reference model**, avoiding **overfitting**.

---

## Implementing IPO with DPOTrainer

To effectively train a model using **IPO loss**, we can use `DPOTrainer` from the `trl` library. This section provides a **practical example** demonstrating how to apply IPO loss instead of the default DPO loss in a fine-tuning scenario.

### **Example: Training a Language Model with IPO Loss**
The following code snippet demonstrates **how to use IPO loss** for training a preference-based model:

```python
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel

# Define maximum sequence length
max_seq_length = 2048  # Supports automatic RoPE Scaling

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/zephyr-sft",
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect type: Float16 for T4/V100, BFloat16 for Ampere+
    load_in_4bit=True,  # 4-bit quantization to reduce memory usage
)

# Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Optimized for dropout = 0
    bias="none",     # Optimized for bias = "none"
    use_gradient_checkpointing=True,
    random_state=3407,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,                      # Enable mixed precision for faster training
)

# Initialize DPOTrainer with IPO loss
# Regularisation Parameter 'beta' (τ) of IPO:
# - Controls the balance between learning from preferences and staying close to reference model.
# - A high τ encourages the policy model to stay closer to reference model, reducing overfitting.
# - A low τ allows more deviation, making the policy model more adaptive.
dpo_trainer = DPOTrainer(
    model=model,
    model_ref=None,                 # IPO can be used without a reference model
    args=training_args,
    loss_type="ipo",                # Use IPO loss instead of default DPO loss
    beta=0.1,                       # Regularization parameter (τ in the IPO paper, defaults to 0.1)
    train_dataset=train_dataset,    # Ensure train_dataset is defined
    eval_dataset=eval_dataset,      # Optional evaluation dataset
    tokenizer=tokenizer,
)

# Start training
dpo_trainer.train()
```

## **Why Use IPO in DPOTrainer?**
The key difference in this implementation is setting `loss_type="ipo"` in `DPOTrainer`, which applies **IPO’s alternative loss function** instead of the default DPO loss.

### **Key Advantages of IPO Loss**
| **Feature**         | **DPO (Default Loss)** | **IPO (Alternative Loss)** |
|--------------------|---------------------|---------------------------|
| **Overfitting Risk** | High (prone to extreme log-likelihood shifts) | Lower (averages over log-likelihoods) |
| **Reference Model** | Required for implicit reward modeling | Optional (can train without a reference model) |
| **Preference Learning** | Based on normalized likelihood (logsigmoid) | Uses identity mapping to stabilize learning |
| **Use Case** | Sensitive to hyperparameters | More stable and generalizable |

## **Takeaways**
- **IPO reduces overfitting** by averaging log-likelihood differences rather than summing them.
- **IPO can work without a reference model**, making it **simpler to implement** than traditional RLHF or DPO approaches.
- **Regularisation in IPO is more stable**, ensuring that learned policies do not overfit human preferences too aggressively.

This implementation provides **a practical way to train preference-optimized models**, making IPO a more robust alternative to standard DPO loss functions.

---

## Conclusion and Future Work

**Identity Preference Optimisation (IPO)** presents a compelling advancement in the field of learning from human preferences. By unifying and improving upon existing methods like RLHF and DPO, IPO offers a more robust and theoretically sound approach. Its ability to prevent overfitting while maintaining alignment with human preferences makes it a valuable tool for training models in various applications.

Future research should focus on scaling IPO to more complex settings, such as large language models, and exploring its performance across diverse datasets. Additionally, integrating IPO with other machine learning paradigms could further enhance its effectiveness and applicability.

## References

1. "A General Theoretical Paradigm to Understand Learning from Human Preferences." [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
2. "IPO Trainer." [Hugging Face Documentation](https://huggingface.co/docs/trl/v0.7.11/en/dpo_trainer)
