# 📧 Spam Email Classifier

## 📌 Project Overview
This project is a **machine learning-based spam email classifier** that predicts whether an email message is spam or not. It uses **Naïve Bayes** for text classification and is implemented in **Google Colab** with a simple **Gradio UI** for easy interaction.

---

## 🚀 Features
- ✅ Classifies emails as **Spam** or **Not Spam**
- ✅ Uses **TF-IDF vectorization** for text processing
- ✅ Built with **Scikit-learn (Naïve Bayes Classifier)**
- ✅ Simple **Gradio UI** for testing
- ✅ Deployable via **Google Colab & GitHub**

---

## 🛠️ Technologies Used
- **Python** (Google Colab)
- **Pandas, Scikit-learn** (Machine Learning)
- **Gradio** (User Interface)

---

## 📂 Dataset
The dataset used is the **SMS Spam Collection** from UCI Machine Learning Repository. It contains:
- 5,574 messages labeled as **ham (not spam)** or **spam**.
- Preprocessed using **TF-IDF** before training.

**Dataset Source:** [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 📌 Installation & Running the Project
### **Step 1: Install Dependencies**
Run the following command in Google Colab:
```python
!pip install gradio pandas scikit-learn
```

### **Step 2: Load & Train the Model**
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms-spam-collection.csv"
df = pd.read_csv(url, encoding="latin-1")
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Train model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])
model.fit(X_train, y_train)
```

### **Step 3: Create the UI with Gradio**
```python
import gradio as gr

def predict_spam(text):
    prediction = model.predict([text])[0]
    return "Spam" if prediction == 1 else "Not Spam"

interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=3, placeholder="Enter your email text here..."),
    outputs="text",
    title="📧 Spam Email Classifier",
    description="Enter an email message to check if it's Spam or Not."
)
interface.launch()
```

---

## 🌍 Deployment
To deploy the project on **Google Colab**, simply:
1. Open the Colab notebook
2. Run all cells
3. Click the **Gradio public link** to access the UI

---

## 🔗 GitHub Integration
### **Push the Project to GitHub**
```python
!git remote set-url origin https://your-username:your-token@github.com/your-username/your-repository.git
!git add .
!git commit -m "Initial commit - Spam Classifier Project"
!git push origin main
```

---

## 📜 License
This project is **open-source** and available under the **MIT License**.

---

## 🙌 Contributors
- **Rajyasri V** 
- Contributions welcome! Feel free to submit a pull request. 🎉

---

## 📞 Contact
For questions or suggestions, feel free to reach out:
- 📧 Email: rajya4107@gmail.com
- 🔗 GitHub: https://github.com/Rajya-V

