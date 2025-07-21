#  AI Inventory Prediction App

This Streamlit-based app predicts inventory levels using time-series forecasting with Prophet and fallback regression models. It also uses LLaMA 3.1 for contextual AI insights based on predictions.

---

##  Features

- Upload inventory data (`.csv`, `.xlsx`, `.json`)
- Automated cleaning and validation
- Time-series forecasting with Prophet or fallback ML
- Alert system for stockout, reorder, and critical levels
- AI insight generation via LLaMA
- Visual flowchart of the AI pipeline
- Downloadable prediction reports

---

## 🛠 Installation Guide

###  1. Clone the repo

```bash
git clone https://github.com/habi-babti/OCP-OPtiStock.git
cd ai-inventory-predictor
````
### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. (Optional) Install Prophet dependencies

``` bash
# Linux/Windows
pip install prophet
# MacOS (use cmdstanpy if needed)
pip install prophet --no-binary=prophet
```
If Prophet fails, the app will fallback to classic regression automatically.

## Running the App
```bash
streamlit run streamlit_app.py
or
python main.py
```
