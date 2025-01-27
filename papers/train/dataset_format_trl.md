# Dataset Formats of HuggingFace TRL

## Table of Contents
1. [Introduction](#introduction)
2. [Type of Text Datasets](#type-of-text-datasets)
   - [Language Modeling](#language-modeling)
   - [Prompt-Only](#prompt-only)
   - [Prompt-Completion](#prompt-completion)
   - [Preference (Explicit)](#preference-explicit)
   - [Preference (Implicit)](#preference-implicit)
   - [Unpaired Preference](#unpaired-preference)
   - [Stepwise Supervision](#stepwise-supervision)
3. [Vision Datasets](#vision-datasets)
4. [Choosing the Right Dataset Type](#choosing-the-right-dataset-type)
5. [Conclusion](#conclusion)

## Introduction
HuggingFace's TRL (Transformer Reinforcement Learning) library provides a robust framework for fine-tuning large language models (LLMs) using reinforcement learning techniques. A critical aspect of this process is the preparation and formatting of datasets, which vary depending on the specific task and trainer being used. This article provides an overview of the dataset formats and types supported by TRL, along with guidance on how to structure your data for different tasks.

## Type of Text Datasets

Datasets in TRL are categorized based on their format and type:
- **Format**: Refers to how the data is structured, typically as either **standard** or **conversational**.
- **Type**: Refers to the specific task the dataset is designed for, such as **language modeling**, **preference**, or **prompt-completion**.

### Language Modeling
**Standard:**
```json
{
  "text": "The sky is blue."
}
```
**Conversational:**
```json
{
  "messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
  ]
}
```

### Prompt-Only
**Standard:**
```json
{
  "prompt": "The sky is"
}
```
**Conversational:**
```json
{
  "prompt": [
    {"role": "user", "content": "What color is the sky?"}
  ]
}
```

### Prompt-Completion
**Standard:**
```json
{
  "prompt": "The sky is",
  "completion": " blue."
}
```
**Conversational:**
```json
{
  "prompt": [
    {"role": "user", "content": "What color is the sky?"}
  ],
  "completion": [
    {"role": "assistant", "content": "It is blue."}
  ]
}
```

### Preference (Explicit)
**Standard:**
```json
{
  "prompt": "The sky is",
  "chosen": " blue.",
  "rejected": " green."
}
```
**Conversational:**
```json
{
  "prompt": [
    {"role": "user", "content": "What color is the sky?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "It is blue."}
  ],
  "rejected": [
    {"role": "assistant", "content": "It is green."}
  ]
}
```

### Preference (Implicit)
**Standard:**
```json
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}
```
**Conversational:**
```json
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```

### Unpaired Preference
**Standard:**
```json
{
  "prompt": "The sky is",
  "completion": " blue.",
  "label": true
}
```
**Conversational:**
```json
{
  "prompt": [
    {"role": "user", "content": "What color is the sky?"}
  ],
  "completion": [
    {"role": "assistant", "content": "It is green."}
  ],
  "label": false
}
```

### Stepwise Supervision
**Standard:**
```json
{
  "prompt": "Which number is larger, 9.8 or 9.11?",
  "completions": [
    "The fractional part of 9.8 is 0.8.",
    "The fractional part of 9.11 is 0.11.",
    "0.11 is greater than 0.8.",
    "Hence, 9.11 > 9.8."
  ],
  "labels": [true, true, false, false]
}
```
**Conversational:**
```json
N/A
```

---

## Vision Datasets
Some trainers in TRL support fine-tuning vision-language models (VLMs) using image-text pairs. For vision datasets, the conversational format is recommended, as it allows for the inclusion of image data alongside text.

### Key Differences:
- The dataset must include an **images** key with the image data.
- The **content** field in messages must be a list of dictionaries, specifying the type of data: **image** or **text**.

### Example:
```python
# Vision dataset:
"content": [
    {"type": "image"}, 
    {"type": "text", "text": "What color is the sky in the image?"}
]
```

## Choosing the Right Dataset Type
The choice of dataset type depends on the task and the specific TRL trainer being used. Below is a summary of the dataset types supported by each TRL trainer:

| Trainer               | RL Type      | Expected Dataset Type                         | ref_model    | reward_model | reward_funcs | peft_config  |
|-----------------------|--------------|-----------------------------------------------|--------------|--------------|--------------|--------------|
| `SFTTrainer`          | N/A          | [Language modeling](#language-modeling)       |              |              |              | Yes          |
| `RewardTrainer`       | Reward Model | [Preference (implicit)](#preference-implicit) |              |              |              | Yes          |
| `DPOTrainer`          | Offline      | [Preference (explicit)](#preference-explicit) | Yes          |              |              | Yes          |
| `CPOTrainer (SimPO)`  | Offline      | [Preference (explicit)](#preference-explicit) |              |              |              | Yes          |
| ORPOTrainer           | Offline      | [Preference (explicit)](#preference-explicit) |              |              |              | Yes          |
| KTOTrainer            | Offline      | [Unpaired preference](#unpaired-preference)   | Yes          |              |              | Yes          |
| `GRPOTrainer`         | Online       | [Prompt-only](#prompt-only)                   |              |              | Yes          | Yes          |
| OnlineDPOTrainer      | Online       | [Prompt-only](#prompt-only)                   | Yes          | Yes          |              | Yes          |
| XPOTrainer            | Online       | [Prompt-only](#prompt-only)                   | Yes          | Yes          |              | Yes          |
| NashMDTrainer         | Online       | [Prompt-only](#prompt-only)                   | Yes          | Yes          |              | Yes          |

## Conclusion
Understanding the dataset formats and types supported by HuggingFace TRL is essential for effectively fine-tuning LLMs and VLMs. By selecting the appropriate dataset type and format for your task, you can ensure optimal performance and compatibility with the TRL trainers. Whether you're working with text-based or vision-language models, TRL provides the flexibility and tools needed to achieve your goals.