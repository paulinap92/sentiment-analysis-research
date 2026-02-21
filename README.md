# 🧠 Sentiment Analysis – Model Benchmarking & Production Decision

This project presents a structured comparison of multiple NLP approaches for binary sentiment classification within a microservice-oriented architecture.

The objective was not only to achieve strong predictive performance, but to select the most appropriate model for deployment in a lightweight production environment.

---

# 📌 Project Structure

The experimentation process is organized into separate notebooks:

1. **Notebook 1 – Pretrained Transformer Baseline**
2. **Notebook 2 – Transformer Fine-Tuning**
3. **Notebook 3 – LSTM Neural Network**
4. **Notebook 4 – Classical ML Models (TF-IDF / CountVectorizer)**

Each notebook includes:

- Full training pipeline
- Train/test evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC (where applicable)
- Classification report

📊 **All detailed results are available inside each notebook.**
The README provides a summary, while full metrics and visualizations are documented directly in the corresponding notebook files.

---

# 🚀 Experimentation Timeline

## 1️⃣ Pretrained Transformer (Baseline)

We first evaluated a pretrained Transformer model (e.g., DistilBERT).

**Goal:**
Establish a strong modern NLP baseline without additional task-specific training.

**Outcome:**
- Good performance
- High computational cost
- Large model size
- Higher inference latency

While effective, it was relatively heavy for short review-style inputs typical for our microservice.

---

## 2️⃣ Transformer Fine-Tuning

Next, we fine-tuned the transformer model on our dataset.

**Goal:**
Improve domain adaptation and maximize performance.

**Outcome:**
- Slight performance improvement
- Increased training complexity
- Significant computational cost

The marginal gains did not justify the operational overhead for this specific problem.

---

## 3️⃣ LSTM Neural Network

We implemented a custom LSTM-based architecture.

**Goal:**
Evaluate a classical sequence-based deep learning model.

**Outcome:**
- Competitive results
- Greater architectural control
- Still heavier than classical ML
- Slower inference compared to linear models

The LSTM performed well but did not provide a significant advantage over simpler methods.

---

## 4️⃣ Classical Machine Learning Models

Finally, we benchmarked lightweight traditional models:

- Linear SVM (TF-IDF)
- Logistic Regression (TF-IDF)
- SGD Classifier (TF-IDF)
- Random Forest (CountVectorizer)
- Naive Bayes (CountVectorizer)

Evaluation was performed on:

- `REVIEWS` – smaller dataset
- `REVIEWS_REAL` – larger, more realistic dataset

---

# 📊 Key Findings

### On the small dataset:
- Near-perfect scores (~1.0 F1)
- Indicates high linear separability
- Suggests the task is structurally simple

### On the realistic dataset:
- Linear SVM achieved ~0.90 F1
- Logistic Regression and Random Forest performed slightly below
- Naive Bayes showed weaker generalization

This comparison revealed:

- The smaller dataset is not fully representative of real-world variability.
- Linear models generalize very effectively for this classification problem.
- Model complexity does not necessarily translate to meaningful performance gains.

---

# 🏆 Final Model Selection

The selected production model is:

> ✅ **Linear SVM with TF-IDF**

### Reasons for selection:

- Highest F1 score on the realistic dataset
- Strong generalization
- Lightweight and efficient
- Fast inference
- Minimal memory footprint
- Ideal for microservice deployment

Given the linear separability of the problem, complex deep learning architectures were unnecessary.

In a microservice context, simplicity, reliability, and performance efficiency are prioritized over architectural complexity.

---

# 🔬 Dataset Evolution Strategy

We recognize that dataset quality strongly impacts model performance.

Going forward, we will:

- Gradually expand and diversify the dataset
- Introduce more varied linguistic patterns
- Increase domain variability
- Continuously re-evaluate model performance

The goal is to ensure that the selected model remains robust as the dataset becomes more representative of real-world conditions.

---

# 🏗 Engineering Perspective

This project demonstrates:

- Structured model benchmarking
- Controlled experimentation across architectures
- Evidence-based model selection
- Avoidance of over-engineering

> The best model is not the most complex one —
> it is the one that solves the problem efficiently.

---

📁 For full training logs, metrics, confusion matrices, and ROC curves, please refer to the individual notebooks.
