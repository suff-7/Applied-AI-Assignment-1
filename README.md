# Amazon Bin Image Quantity Predictor  
### Applied AI for Industry – Team Submission

**Team Members**  
- Syed Sufyaan – SE22UECM046  
- Ayaz Ahmed Ansari – SE22UARI024  
- Aryan Pawar – SE22UARI195  
- Hamza Babukhan – SE22UARI208  

---

## 🚀 Deployed App

View and use the live web app here:  
[https://applied-ai-assignment-1-mtatsyuuydprtjjvte2ntg.streamlit.app/](https://applied-ai-assignment-1-mtatsyuuydprtjjvte2ntg.streamlit.app/)

---

## 📦 Project Structure

- `app.py` – The Streamlit UI and model inference code.
- `requirements.txt` – List of all dependencies for running the app.
- `quantity_model.pth` – Downloaded automatically at runtime from Google Drive (no need to add to repo).
- `notebook1d38817fa3-3.ipynb` – Complete training and analysis notebook.
- `training_history_gpu.json` – Training/validation logs for plotting and evaluation.

---

## 📝 How to Run the Code (Locally)

**1. Clone this repository**

git clone https://github.com/suff-7/Applied-AI-Assignment-1.git
cd Applied-AI-Assignment-1


**2. Make sure you have Python 3.8+ installed. We recommend using a virtual environment:**

python -m venv .venv
source .venv/bin/activate # on Linux/Mac
.venv\Scripts\activate # on Windows


**3. Install dependencies**

pip install -r requirements.txt


**4. Run the Streamlit app**

streamlit run app.py


- The app will automatically download the trained model weights from Google Drive on the first run.
- Upload one or more test bin images to get quantity predictions and model confidence.
- The UI is modern and supports batch uploads.

---

## 🧠 About This Project

This project demonstrates a deep learning solution to automatically count items in Amazon bin images.  
- Based on a custom ResNet-style regression architecture.
- Trained from scratch using a 10,000-sample subset of the Amazon Bin Image Dataset.
- Full development, training, and analysis in the included notebook.
- Deployable and reproducible via the included Streamlit app.

**Main features:**
- Batch and single-image predictions.
- Model “confidence” visualization.
- Clear, user-friendly interface for demo and grading.

---

## 📊 For the Instructor

- The full training notebook provides EDA, preprocessing, model selection/rationale, and evaluation metrics.
- Streamlit app allows live or local demo; no large files need to be uploaded to the repo.
- All core code files (including the model class) are included and can be reviewed in `app.py` and the notebook.

---

*See project documentation for design, code explanations, etc (will be submitted separately).*

---
