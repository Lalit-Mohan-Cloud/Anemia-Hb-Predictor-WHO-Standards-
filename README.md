# 🛡️ Anemia (Hb) Predictor (WHO Standards)

A **Streamlit-based application** that predicts Hemoglobin (Hb) levels from conjunctiva images using a **CNN + Formulaic Model**.
The app applies **WHO Hemoglobin cut-offs** to classify anemia severity by demographic group.
For **safety**, the final reported Hb is chosen as the **minimum of the CNN and Formula model outputs** to maximize sensitivity.

---

## 📌 Features

* Upload a conjunctiva image (`.jpg`, `.jpeg`, `.png`)
* Predict Hb levels using:

  * **CNN-based deep learning model** (`model.keras`)
  * **Formulaic Hb estimation model** (based on RGB values)
* Apply **WHO anemia classification standards** for:

  * Men (15+ years)
  * Non-pregnant Women (15+ years)
  * Pregnant Women
  * Children (various age groups)
* Safety-first logic: reports the **lower of CNN and formula Hb values**
* Generates interpretation with **severity classification (Normal, Mild, Moderate, Severe)**

---

## ⚙️ Project Structure

```
anemia-predictor/
│── app.py                        # Main Streamlit app
│── model.keras       # Trained CNN model
│── requirements.txt              # Dependencies
│── README.md                     # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/anemia-predictor.git
cd anemia-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will start at 👉 **[http://127.0.0.1:8501](http://127.0.0.1:8501)**

---

## 📡 Usage

1. Select your **demographic group** in the sidebar.
2. Upload a **conjunctiva image**.
3. Click **"Predict & Classify Hb Level"**.
4. View:

   * CNN Hb prediction
   * Formulaic Hb prediction
   * Final reported Hb (minimum value)
   * WHO classification (Normal, Mild, Moderate, Severe)

---

## 📊 WHO Hemoglobin Cut-offs (g/dL)

| Group                        | Normal | Mild      | Moderate | Severe |
| ---------------------------- | ------ | --------- | -------- | ------ |
| Men (15+ years)              | ≥13.0  | 11.0–12.9 | 8.0–10.9 | <8.0   |
| Non-pregnant Women (15+ yrs) | ≥12.0  | 11.0–11.9 | 8.0–10.9 | <8.0   |
| Pregnant Women               | ≥11.0  | 10.0–10.9 | 7.0–9.9  | <7.0   |
| Children 12–14 years         | ≥12.0  | 11.0–11.9 | 8.0–10.9 | <8.0   |
| Children 5–11 years          | ≥11.5  | 11.0–11.4 | 8.0–10.9 | <8.0   |
| Children 6–59 months         | ≥11.0  | 10.0–10.9 | 7.0–9.9  | <7.0   |

---

## 📦 Requirements

* Python 3.8+
* streamlit
* tensorflow
* numpy
* opencv-python
* pillow

Install with:

```bash
pip install streamlit tensorflow numpy opencv-python pillow
```

---

## ⚠️ Disclaimer

This tool is intended for **screening and educational purposes only**.
It should **not** be used as a substitute for professional medical diagnosis.
Always consult a **licensed healthcare provider** for accurate diagnosis and treatment.

---
