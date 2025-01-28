# Contrastive Preference Optimization (CPO): Advancing Machine Translation

### Table of Contents
1. [Introduction](#introduction)  
2. [What is CPO?](#what-is-cpo)  
3. [Challenges in Supervised Fine-Tuning (SFT)](#challenges-in-supervised-fine-tuning-sft)  
4. [How CPO Works](#how-cpo-works)  
5. [Advantages of CPO](#advantages-of-cpo)  
6. [Experimental Results](#experimental-results)  
6. [Implementing CPO in Code](#implementing-cpo-in-code)  
7. [Conclusion](#conclusion)  
8. [References](#references)  

---

## Introduction

Moderate-sized large language models (LLMs) have shown potential in machine translation (MT) but still fall short compared to state-of-the-art translation systems like GPT-4 or WMT competition winners. **Contrastive Preference Optimization (CPO)** is a novel method introduced to bridge this performance gap by refining models' training techniques and leveraging both model-generated and reference data for improved results.

Machine translation (MT) is a field within natural language processing (NLP) that focuses on enabling machines to translate text between different languages. Traditional approaches in MT often rely on encoder-decoder architectures and supervised fine-tuning (SFT) to train models on large datasets of parallel text. While these methods have achieved remarkable success, they face challenges in scalability and quality, particularly when handling moderate-sized large language models (LLMs).

**CPO** leverages two core principles: `preference learning`, which teaches the model to distinguish high-quality translations from less desirable ones, and a `contrastive objective`, which helps the model prioritize preferred translations over near-perfect but flawed alternatives. By combining these strategies, **CPO** overcomes the limitations of SFT and significantly improves translation quality.

---

## What is CPO?

**Contrastive Preference Optimization (CPO)** is an advanced training approach that improves machine translation by teaching models to generate superior translations while avoiding near-perfect but flawed outputs. Unlike traditional methods, CPO uses `preference learning` combined with a `contrastive objective` to address inherent limitations in supervised fine-tuning (SFT).

### Key Formula of CPO

The **key formula of Contrastive Preference Optimization (CPO)** is its **loss function**, which combines two essential components:

1. **Preference Learning Loss $( L_{\text{prefer}} $):** Ensures the model generates translations that are preferred over suboptimal ones.
2. **Supervised Fine-Tuning Loss $( L_{\text{NLL}} $):** Prevents the model from deviating too far from the data distribution of high-quality translations.

The formula for the CPO loss is as follows:

$[
\text{Loss} = L_{\text{prefer}} + L_{\text{NLL}}
$]

---

### 1. **Preference Learning Loss $( L_{\text{prefer}} $)**

This term optimizes the model to prioritize the preferred (better) translations over the dis-preferred (worse) translations. It uses a **contrastive learning approach** based on a sigmoid function to evaluate the likelihood difference between preferred and dis-preferred translations.

$[
L_{\text{prefer}} = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w | x)}{\pi_\theta(y_l | x)} \right) \right]
$]

- $(x$): Source sentence.
- $(y_w$): Preferred translation (better quality).
- $(y_l$): Dis-preferred translation (lower quality).
- $(\pi_\theta(y | x)$): Model's predicted probability for a translation $(y$) given the source $(x$).
- $(\sigma$): Sigmoid function.
- $(\beta$): Scaling hyperparameter to control the strength of preference weighting.
- $-\mathbb{E}_{(x, y_w, y_l) \sim D}$: The expectation operator, averaging over all data triplets $(x, y_w, y_l)$ sampled from the dataset $(D)$.

This term ensures that the model gives higher probabilities to preferred translations compared to dis-preferred ones.

#### Example of $( L_{\text{prefer}} $)

1. **Source Sentence $( x $):**  
   *"The cat is on the mat."*

   The dataset provides three translations for this sentence:
   - **Preferred Translation $( y_w $):**  
     *"Die Katze ist auf der Matte."* (Correct and fluent German translation).
   - **Dis-preferred Translation $( y_l $):**  
     *"Die Katze sitzt auf der Matte."* (Almost correct but introduces a subtle error—"sits on the mat" instead of "is on the mat").
   - **Poor Translation:**  
     *"Die Katze ist unter der Matte."* ("Under the mat"—completely incorrect).

   For simplicity, CPO considers only $( y_w $) (preferred) and $( y_l $) (dis-preferred) translations.

2. The model assigns probabilities to each translation:
   - $( \pi_\theta(y_w | x) = 0.75 $) (75% confidence in $( y_w $)).
   - $( \pi_\theta(y_l | x) = 0.25 $) (25% confidence in $( y_l $)).

3. The contrastive objective in $( L_{\text{prefer}} $) focuses on increasing the gap between these probabilities by comparing their likelihoods:

   - **Logarithmic Comparison of Probabilities:**  
     $[
     \log \frac{\pi_\theta(y_w | x)}{\pi_\theta(y_l | x)} = \log \frac{0.75}{0.25} = \log(3) \approx 1.10
     $]

   - **Scaling with $( \beta $):**  
     If $( \beta = 0.5 $), this value is scaled:  
     $[
     \beta \cdot \log \frac{\pi_\theta(y_w | x)}{\pi_\theta(y_l | x)} = 0.5 \cdot 1.10 = 0.55
     $]

   - **Sigmoid Function ($( \sigma $)):**  
     The sigmoid function converts this value into a probability-like score, ensuring it falls between 0 and 1:  
     $[
     \sigma(0.55) \approx 0.63
     $]

   - **Logarithm of the Sigmoid Score:**  
     $[
     \log \sigma(0.55) \approx \log(0.63) \approx -0.46
     $]

4. The expectation $( -\mathbb{E}_{(x, y_w, y_l) \sim D} $) averages this process over the dataset. For this single example:  
   $[
   L_{\text{prefer}} = -(-0.46) = 0.46
   $]

   This loss value indicates that the model needs to improve its preference for $( y_w $) by further increasing the probability gap between $( y_w $) and $( y_l $).

5. The model adjusts its parameters to:
   - **Increase $( \pi_\theta(y_w | x) $):** This raises the likelihood of the preferred translation.
   - **Decrease $( \pi_\theta(y_l | x) $):** This lowers the likelihood of the dis-preferred translation.

   Over time, this process reduces $( L_{\text{prefer}} $) and helps the model consistently favor **better translations** over weaker ones.

---

### 2. **Supervised Fine-Tuning Loss $( L_{\text{NLL}} $)**

The **major purpose of $( L_{\text{NLL}} $)** is to train the model to **align its predictions closely with high-quality reference translations** by minimizing the difference between the model’s predicted output and the actual preferred translation. This ensures that the model learns from the high-quality training data and does not deviate significantly from it.

$[
L_{\text{NLL}} = - \mathbb{E}_{(x, y_w) \sim D} \left[ \log \pi_\theta(y_w | x) \right]
$]

- $(x)$: Source sentence.
- $(y_w)$: Preferred translation (better quality).
- $(\pi_\theta(y_w | x))$: Model's predicted probability for the preferred translation $(y_w)$ given the source $(x)$.
- $-\mathbb{E}_{(x, y_w) \sim D}$: The expectation operator, averaging over all data pairs $(x, y_w)$ sampled from the dataset $(D)$.

This is a standard supervised learning objective, ensuring the model learns from the preferred translations.

#### Example of $( L_{\text{NLL}} $)

The **negative log-likelihood (NLL)** is a measure of how "wrong" the model's predicted probabilities are for the correct translation. Lower NLL means the model is more confident in generating the correct output.

1. Model’s Predicted Probabilities $( \pi_\theta(y|x) $):  
   Let’s say the model predicts probabilities for possible translations:
   - *"Die Katze ist auf der Matte."* → $(0.8$) (80% confident this is correct)
   - *"Die Katze ist unter der Matte."* → $(0.1$) (10% confident this is correct)
   - *"Die Katze sitzt auf dem Boden."* → $(0.1$) (10% confident this is correct)

2. Log-Likelihood:  
   The likelihood of the preferred translation is its predicted probability, $( \pi_\theta(y_w|x) = 0.8 $).  
   The log-likelihood is the logarithm of this probability:
   $[
   \log \pi_\theta(y_w|x) = \log(0.8) \approx -0.22
   $]

3. Negative Log-Likelihood (NLL):  
   The NLL simply negates the log-likelihood:  
   $[
   -\log \pi_\theta(y_w|x) = -(-0.22) = 0.22
   $]

4. The model adjusts its parameters to:
   - If the model improves and assigns $(0.9$) probability to $(y_w$), the NLL drops to: $[ -\log(0.9) \approx 0.10 $]
   - If the model assigns $(1.0$) probability to $(y_w$), the NLL becomes $(0$), indicating perfect confidence in the correct translation.

---

### Combined CPO Loss

By combining these two components, the final CPO loss function becomes:

$[
\text{Loss} = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w | x)}{\pi_\theta(y_l | x)} \right) \right] - \mathbb{E}_{(x, y_w) \sim D} \left[ \log \pi_\theta(y_w | x) \right]
$]

### Explanation with an Example

Suppose a model is translating a sentence from English to German:
- **Source Sentence $(x$):** "The cat is on the mat."
- **Preferred Translation $(y_w$):** "Die Katze ist auf der Matte."
- **Dis-preferred Translation $(y_l$):** "Die Katze sitzt auf dem Boden." (Close but incorrect as it changes the meaning to "on the ground.")

Using CPO:
- The **$( L_{\text{prefer}} $)** term ensures the model strongly prefers $(y_w$) over $(y_l$), prioritizing the correct context ("on the mat").
- The **$( L_{\text{NLL}} $)** term reinforces the model to align with high-quality training examples (e.g., $(y_w$)).

---

## Challenges in Supervised Fine-Tuning (SFT)

SFT, a common technique in MT, mimics reference translations to minimize prediction errors. However, it has significant drawbacks:
- **Reference Limitations:** Even human-generated translations (gold references) may contain errors or omissions.
- **Performance Bottlenecks:** SFT restricts models to the quality level of the training data.
- **Error Rejection:** Models lack the ability to identify and avoid subtle translation errors.

For instance, in the FLORES-200 dataset, model-generated translations sometimes outperformed human references, highlighting the need for better training approaches.

---

## How CPO Works

### Key Features
1. **Preference Learning:** Trains models to prioritize high-quality translations.
2. **Contrastive Objective:** Helps models distinguish between "preferred" and "dis-preferred" translations.
3. **Efficiency:** Requires minimal additional parameters (0.1% of the original model) and a small dataset (22,000 sentences).

### Preference Data Construction

CPO uses a triplet format for training:
- **Preferred Translation:** The highest-quality output (e.g., from GPT-4).
- **Dis-preferred Translation:** A near-perfect but flawed output.
- **Reference Translation:** Human-provided translations.

The example below illustrates the process:
| **Source Sentence**        | **Translation Type**   | **Example Translation**                                       | **Score** |
|----------------------------|------------------------|---------------------------------------------------------------|----------|
| "Martelly's CEP"           | Preferred              | "Martelly's fifth Provisional Electoral Council (CEP)."       | High     |
|                            | Dis-preferred          | "Martelly's fifth CEP in four years."                         | Medium   |
|                            | Reference              | "It is Martelly's fifth CEP in four years."                   | Low      |

### CPO Loss Function

CPO optimizes a dual-objective loss function:
1. **Preference Learning Loss (Lprefer):** Encourages better translation quality.
2. **Supervised Fine-Tuning Loss (LNLL):** Ensures consistency with training data.

The combined loss function is:
$[
\text{Loss} = L_{\text{prefer}} + L_{\text{NLL}}
$]

---

## Advantages of CPO

1. **Improved Quality:** Surpasses traditional models by avoiding flawed translations.
2. **Memory & Speed Efficiency:** Reduces computational requirements compared to other methods like Direct Preference Optimization (DPO).
3. **Robust Evaluation:** Incorporates reference-free metrics like KIWI-XXL and XCOMET for unbiased assessment.

---

## Experimental Results

### Performance Against Benchmarks

CPO fine-tuning on the ALMA-13B-LoRA model resulted in significant improvements. It matched or outperformed GPT-4 and WMT competition winners across multiple datasets.

| **Model**          | **Average Score (KIWI-XXL)** | **Average Score (XCOMET)** |
|---------------------|-----------------------------|----------------------------|
| Gold Reference      | 83.47                      | 92.85                     |
| WMT Winners         | 84.81                      | 93.78                     |
| GPT-4              | 83.83                      | 93.23                     |
| **ALMA-13B-R**      | **85.74**                  | **94.05**                 |

### Human Evaluation

Human evaluators rated translations from CPO-trained models higher than those from baseline models.

| **Metric**                | **ALMA-13B-LoRA** | **ALMA-13B-R** |
|---------------------------|-------------------|----------------|
| Average Score             | 4.86             | 5.16          |
| Preferred Translation (%) | 62.5             | 77.8          |

### Ablation Studies

Key findings from ablation studies:
1. Both components of the CPO loss function (Lprefer and LNLL) are essential for optimal performance.
2. High-quality "dis-preferred" data significantly improves training outcomes.

---

## Implementing CPO in Code

The following example demonstrates how to implement CPO using the `trl` library with a moderate-sized LLM:

```python
# train_cpo.py
from datasets import load_dataset
from trl import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Load training dataset
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Define training configuration
training_args = CPOConfig(output_dir="Qwen2-0.5B-CPO", logging_steps=10)

# Initialize the trainer
trainer = CPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)

# Train the model with CPO
trainer.train()
```

### Explanation of the Code

1. **Model Initialization:**
   - A pretrained model (`Qwen/Qwen2-0.5B-Instruct`) is loaded using the `transformers` library.
   - The corresponding tokenizer is also initialized.

2. **Dataset Loading:**
   - A preference dataset, `ultrafeedback_binarized`, is loaded from the `datasets` library. This dataset contains examples of preferred and dis-preferred translations for contrastive learning.

3. **CPO Configuration:**
   - A `CPOConfig` object is created to define the training parameters such as `output_dir` (where the model will be saved) and `logging_steps` (frequency of logging during training).

4. **CPO Trainer:**
   - The `CPOTrainer` class is used to handle the training process. It integrates the CPO loss function and optimizes the model using preference data.

5. **Model Training:**
   - The `trainer.train()` method trains the model using the loaded dataset and the defined configuration.

---

## Conclusion

CPO introduces a paradigm shift in machine translation by overcoming the limitations of traditional methods. Its ability to leverage model-generated data and reject suboptimal translations makes it a robust solution for moderate-sized LLMs. The ALMA-13B-R model, trained using CPO, sets a new benchmark in translation quality, rivaling top-performing systems like GPT-4.

---

## References

1. [Contrastive Preference Optimization (arXiv:2401.08417)](https://arxiv.org/abs/2401.08417)
2. [HuggingFace CPO Trainer](https://huggingface.co/docs/trl/main/en/cpo_trainer)
