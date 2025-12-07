# Article-Fact-Extraction

This repository contains a complete end‑to‑end pipeline for transforming raw C4-like web text into structured factual triples using a T5-based sequence-to-sequence model. The system covers **data extraction**, **cleaning**, **triple formatting**, **tokenization**, **dataset preparation**, and **model training**.

---

## 🚀 Project Overview

This project implements a modern fact-extraction system inspired by WebIE-style pipelines. It:

* Loads and filters large web‑scale datasets.
* Extracts metadata and annotated factual triples.
* Converts triples into a linearized text format suitable for seq2seq training.
* Tokenizes both inputs and labels with separate padding strategies.
* Stores efficiently in Arrow format for high-throughput training.
* Trains a T5 model to generate triples given raw text.

---

## 📦 Features

### ✔ Efficient dataset handling

* Uses `datasets` library with multiprocessing.
* Safely filters entries missing metadata or triples.
* Supports C4-style web documents.

### ✔ Triple linearization

Triples are converted into a structured token sequence:

```
<subj> SUBJ <rel> REL <obj> OBJ <et>
```

This format is stable, compact, and easy for T5 models to learn.

### ✔ Robust tokenization

* Input max length: **256 tokens**
* Label max length: **256 tokens**
* Label padding tokens are replaced with **-100** for proper loss masking.

### ✔ Training-ready Arrow dataset

Arrow dramatically reduces disk usage and memory overhead.

### ✔ Custom training loop

* Mini-batch training via PyTorch DataLoader
* Uses HuggingFace `DataCollatorForSeq2Seq`
* Supports GPUs and mixed-precision

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

Or manually ensure you have:

```
torch
transformers
datasets
yaml
```

---

## 🗂 Dataset Pipeline

### 1. Load raw dataset

```python
dataset = load_dataset("json", data_files=path, split="train")
```

### 2. Filter invalid entries

Ensures metadata exists and triples are present.

### 3. Convert triples

Adds a new column `triple_text`.

### 4. Tokenize

Creates `input_ids`, `attention_mask`, and `labels`.

### 5. Save

```python
tokenized.save_to_disk("data/model_input/train")
```

---

## 🧠 Model Training

Run:

```bash
python -m Fact_Extraction.model
```

The model learns to generate factual triples conditioned on raw text.

---

## 🧪 Example Input / Output

### Input Text

```
The Eiffel Tower is located in Paris and was completed in 1889.
```

### Model Output

```
<subj> Eiffel Tower <rel> located_in <obj> Paris <et>
<subj> Eiffel Tower <rel> completed_in <obj> 1889 <et>
```

---

## 🛠 Configuration

YAML configuration defines:

* dataset paths
* tokenization parameters
* training hyperparameters
* logging

Simplify experiments by editing only the config files.

---

## 📊 Logging

Centralized logging with clean formatting.
Fixes issues like Python's default `logger.info("text", value)` formatting bugs.

---

## 🤝 Contributions

PRs are welcome! This system can be extended with:

* Relation classification models
* Subject/object canonicalization
* Knowledge graph export
* Model evaluation scripts

---

## 📜 License

MIT License

---

## ⭐ Support

If this project helps you, feel free to star ⭐ the repository!
