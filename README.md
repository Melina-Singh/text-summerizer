
# 📝 Text Summarization Web App

This project implements a **Sequence-to-Sequence (Seq2Seq)** model with **LSTM layers** for **abstractive text summarization**, using a subset of the CNN/Daily Mail dataset (150 articles). The application includes a **modular Python backend** and a **Flask-based web interface** for generating summaries.

---


---

## 📁 Directory Structure

```text
summarization_app/
├── models/
│   ├── __init__.py
│   └── seq2seq_model.py
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── evaluation.py
|   └──logger.py
|   
├── templates/
│   └── index.html
├── static/
│   └── css/
│       └── styles.css
|
├── app.py
├── train.py
├──transformer_train.py
├── requirements.txt
└── README.md


## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone <[repository-url](https://github.com/Melina-Singh/text-summerizer.git)>
cd summarization_app
```

![Screenshot 2025-05-09 212911](https://github.com/user-attachments/assets/f65e9ba2-ab78-427d-a424-07bd4f58e1cf)

