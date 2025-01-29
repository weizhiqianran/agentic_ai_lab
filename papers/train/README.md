## Mastering LLM & VLM Training: From Supervised Fine-Tuning to Reinforcement Learning

This course covers various reinforcement learning techniques for optimizing large language models (LLMs), including both **offline** and **online reinforcement learning** strategies. You will explore key concepts such as **Supervised Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, and other **advanced RL methods** for preference alignment and model training. Each article delves into a specific approach, explaining its mechanics, benefits, and implementation.

---

### **1. Basic Knowledge**
Understanding the fundamentals of reinforcement learning (RL) in LLMs, including dataset formatting and key concepts for offline and online RL.

- **[LLM Instruction Tuning: Should You Use SFT, Offline RL, or Online RL?](sft_online_offline_rl.md)**

  Introduction to **Supervised Fine-Tuning (SFT)** and RL methods, explaining their trade-offs and use cases.

- **[Dataset Formats of Hugging Face TRL](trl_dataset_format.md)**

  Covers dataset structures used for reinforcement learning in TRL, including explicit and implicit preference datasets.

---

### **2. Offline Reinforcement Learning**
Explores **Offline RL** techniques, which use pre-collected data rather than real-time feedback, optimizing models based on stored preferences.

- **[Direct Preference Optimization (DPO)](trl_dpo_trainer.md)**

  A method that simplifies RLHF by removing reward models and optimizing policy through preference-based loss.

- **[Identity Preference Optimization (IPO)](trl_ipo_trainer.md)**

  Extends DPO with regularization techniques to prevent overfitting and improve preference alignment.

- **[Contrastive Preference Optimization (CPO)](trl_cpo_trainer.md)**

  Uses contrastive learning to refine model responses in tasks like machine translation.

- **[Simple Preference Optimization (SimPO)](trl_simpo_trainer.md)**

  A lightweight alternative to DPO that removes dependency on reference models.

- **[Odds Ratio Preference Optimization (ORPO)](trl_orpo_trainer.md)**

  A monolithic preference optimization approach without reference models, reducing computational complexity.

---

### **3. Online Reinforcement Learning**
Explores **Online RL**, where models receive real-time feedback and adapt dynamically.

- **[Direct Language Model Alignment from Online AI Feedback (OAIF)](trl_oaif_trainer.md)**

  Introduces **online preference learning**, using AI evaluators to refine responses in real-time.

- **[Group Relative Policy Optimization (GRPO)](trl_grpo_trainer.md)**

  A reinforcement learning method that enhances mathematical reasoning models through group-based scoring.

---
