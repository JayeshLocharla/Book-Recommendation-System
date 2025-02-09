# Book-Recommendation-System

## Description

This project leverages **Large Language Models (LLMs)** to perform **zero-shot text classification, sentiment analysis, data exploration, and vector search**, integrated into an interactive **Gradio dashboard**. It enables users to analyze text data without requiring pre-defined labels, making it highly adaptable for various NLP tasks.

## Features

### 1. **Data Exploration (`data-exploration.ipynb`)**
   - Conducts **Exploratory Data Analysis (EDA)** on text datasets.
   - Generates statistical summaries, word frequency distributions, and visual insights.

### 2. **Zero-Shot Text Classification (`text-classification.ipynb`)**
   - Uses **Hugging Face's `transformers` library** for **Zero-Shot Classification**.
   - Allows classification of text into user-defined categories without requiring pre-training.
   - Powered by **LLMs (e.g., `facebook/bart-large-mnli`, `roberta-large-mnli`)**.

### 3. **Sentiment Analysis (`sentiment-analysis.ipynb`)**
   - Classifies text as **positive, negative, or neutral** using a pre-trained LLM.
   - Uses Hugging Face sentiment models like `distilbert-base-uncased-finetuned-sst-2`.

### 4. **Vector Search (`vector-search.ipynb`)**
   - Implements **semantic search** using **vector embeddings**.
   - Uses models like **`sentence-transformers`** for generating vector representations.
   - Performs similarity search using **FAISS or Annoy**.

### 5. **Interactive Dashboard (`gradio-dashboard.py`)**
   - A web-based **Gradio dashboard** for easy interaction.
   - Supports:
     - Text input for **classification & sentiment analysis**.
     - **Vector-based search** for similar text retrieval.
     - Visualization of **data insights**.

## Technologies Used
- **Python**
- **Hugging Face Transformers (`transformers`, `sentence-transformers`)**
- **LLMs (BART, RoBERTa, DistilBERT, etc.)**
- **Zero-Shot Classification**
- **FAISS / Annoy (for vector search)**
- **Gradio (for UI/UX)**
- **Pandas, NumPy, Matplotlib, Seaborn (for data exploration)**

## Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd <project-folder>
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Gradio Dashboard
```bash
python gradio-dashboard.py
```

## Features

- Upload or input text to classify, analyze sentiment, or search similar texts.
- Run data exploration notebooks for deep insights.
- Use LLM-based zero-shot classification without pre-training.
- Access all functionalities through the Gradio dashboard.
