# 🏔️ AI-Based Intelligent Enquiry Chatbot for HP Colleges

> An offline, multi-college chatbot designed to provide fast and accurate responses to student queries.

---

## 📌 Overview

This project is a machine learning-based chatbot that helps students get instant information about colleges such as:

* Fees
* Courses
* Hostel facilities
* Placements
* Contact details

The system uses Natural Language Processing (NLP) and Machine Learning techniques to understand user queries and respond accurately.

---

## ✨ Key Features

* ✅ Works completely offline
* ✅ Supports multiple colleges
* ✅ Fast and accurate responses
* ✅ User-friendly chat interface
* ✅ Quick query buttons
* ✅ College comparison feature
* ✅ Analytics dashboard
* ✅ Clean UI (no technical complexity for users)

---

## 🏫 Supported Colleges

* NIT Hamirpur
* IIT Mandi
* HPU Shimla
* Jaypee University Waknaghat
* CSKHPKV Palampur
* JNGEC Mandi

---

## 🤖 Machine Learning Model

The system uses **Logistic Regression** for intent classification.

It was selected after comparing multiple models based on:

* Accuracy
* Consistency
* Performance on small datasets

---

## 🧠 System Workflow

```
User Input
   ↓
Text Preprocessing
   ↓
Vectorization (CountVectorizer)
   ↓
Model Prediction (Intent Classification)
   ↓
Response Generation
```

---

## ⚙️ Technologies Used

* **Python**
* **NLTK** (Text preprocessing)
* **Scikit-learn** (Machine learning)
* **Streamlit** (User interface)
* **Pandas & NumPy**

---

## 🚀 How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the chatbot

```bash
streamlit run app.py
```

### 3. Open in browser

```
http://localhost:8501
```

---

## 📊 Dataset

* Self-created dataset
* ~400–500 query patterns
* Multiple intents like fees, courses, hostel, placements, and contact
* Covers multiple colleges

---

## ⚠️ Limitations

* Limited dataset size
* Cannot handle highly complex or unrelated queries

---

## 🚀 Future Scope

* Add more colleges
* Improve accuracy with advanced models
* Add voice-based interaction
* Deploy as a web application

---

## 👩‍💻 Author

**Maahi Thakur**
BCA (AI & ML)
Shoolini University

---

## 📌 Final Note

This project focuses on simplicity, usability, and efficiency, providing a practical solution for handling student enquiries using AI.
