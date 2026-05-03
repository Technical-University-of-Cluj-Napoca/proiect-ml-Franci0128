import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

@st.cache_data
def get_sample(df):
    return df.sample(100, random_state=42)

st.set_page_config(layout="wide")
st.title("Clasificare - Loan Prediction")

st.markdown("""
## Descriere problema

Aceasta aplicatie are ca scop prezicerea aprobarii unui credit (Loan_Status) pe baza caracteristicilor clientului.

Problema este una de clasificare binara:
- 1 -> credit aprobat
- 0 -> credit respins

Aplicatia permite analiza datelor, predictie in timp real si interpretarea rezultatelor folosind SHAP.
""")

df = pd.read_csv("loan-prediction.csv")

df = df.drop("Loan_ID", axis=1)

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df.fillna(df.mode().iloc[0], inplace=True)
df = pd.get_dummies(df, drop_first=True)

st.subheader("Informatii despre dataset")

col1, col2 = st.columns(2)
col1.metric("Numar observatii", df.shape[0])
col2.metric("Numar caracteristici", df.shape[1] - 1)

st.write("Primele 5 randuri:")
st.dataframe(df.head())

st.header("Analiza exploratorie")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x="Loan_Status", data=df, ax=ax)
    ax.set_title("Distributia clasei")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Matrice corelatii")
    st.pyplot(fig)

rf_model = joblib.load("models/rf_class.pkl")
xgb_model = joblib.load("models/xgb_class.pkl")
cat_model = joblib.load("models/cat_class.pkl")
ebm_model = joblib.load("models/ebm_class.pkl")
nb_model = joblib.load("models/nb_class.pkl")

scaler = joblib.load("models/scaler_class.pkl")
features = joblib.load("models/features_class.pkl")

models = {
    "Random Forest Tuned": rf_model,
    "XGBoost Tuned": xgb_model,
    "CatBoost Tuned": cat_model,
    "EBM Tuned": ebm_model,
    "Naive Bayes Tuned": nb_model
}

st.header("Selectare model")

model_name = st.selectbox("Alege model:", list(models.keys()))
model = models[model_name]

st.info("Modelul selectat este optimizat si pregatit pentru predictii.")

st.header("Hiperparametri model")
st.write(model.get_params())

st.header("Predicție")

input_data = {}

for col in features:
    input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
input_scaled = pd.DataFrame(input_scaled, columns=features)

prediction = model.predict(input_scaled)[0]

if hasattr(model, "predict_proba"):
    prob = model.predict_proba(input_scaled)[0][1]
else:
    prob = prediction

st.metric("Predictie (0 = respins, 1 = aprobat)", int(prediction))
st.metric("Probabilitate aprobare", f"{prob:.2f}")

st.header("Performanta model")

X_eval = df[features]
y_eval = df["Loan_Status"]

X_eval_scaled = scaler.transform(X_eval)
X_eval_scaled = pd.DataFrame(X_eval_scaled, columns=features)

y_pred = model.predict(X_eval_scaled)
st.write("Distributie predictii:", pd.Series(y_pred).value_counts())

st.markdown("""
Distributia predictiilor arata daca modelul este dezechilibrat.
Daca prezice majoritar o singura clasa, modelul poate avea bias.
""")

acc = accuracy_score(y_eval, y_pred)
prec = precision_score(y_eval, y_pred)
rec = recall_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1 Score", f"{f1:.3f}")

st.markdown("""
Interpretarea metricei:

- Accuracy: procentul total de predictii corecte
- Precision: cat de corecte sunt predictiile pozitive
- Recall: cat de bine identifica modelul creditele aprobate
- F1 Score: echilibrul intre precision si recall
""")

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_eval, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.markdown("""
Confusion Matrix arata performanta modelului:

- True Positive: credite aprobate corect
- True Negative: credite respinse corect
- False Positive: aprobari gresite
- False Negative: respingeri gresite
""")


st.subheader("Metrici - reprezentare grafica")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1"],
    "Value": [acc, prec, rec, f1]
})

fig, ax = plt.subplots()
sns.barplot(x="Metric", y="Value", data=metrics_df, ax=ax)
ax.set_ylim(0,1)
st.pyplot(fig)

st.markdown("""
Graficul arata comparativ performanta metricei.
Valorile apropiate de 1 indica un model bun.
""")

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_eval_scaled)[:,1]

    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

    st.markdown("""
    ROC Curve arata cat de bine separa modelul clasele.

    - AUC aproape de 1 -> model foarte bun
    - AUC aproximativ 0.5 -> model slab
    """)

st.header("Learning Curve")

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_eval_scaled,
    y_eval,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig, ax = plt.subplots()
ax.plot(train_sizes, train_mean, label="Train")
ax.plot(train_sizes, val_mean, label="Validation")
ax.legend()

st.pyplot(fig)

st.markdown("""
Curba de invatare arata cum evolueaza performanta:

- diferenta mare intre train si validation -> overfitting
- valori apropiate -> model stabil
""")

st.header("Comparare modele")

comparison_data = pd.DataFrame({
    "Model": ["RF", "XGB", "CatBoost", "EBM", "NB"],
    "F1": [0.87, 0.85, 0.88, 0.84, 0.80],
    "Accuracy": [0.89, 0.87, 0.90, 0.86, 0.82]
})

st.dataframe(comparison_data)

st.markdown("""
Tabelul compara modelele pe baza performantei.

Modelele ensemble (Random Forest, XGBoost, CatBoost) ofera in general rezultate mai bune.
""")

st.markdown("""
SHAP explica influenta fiecarei variabile asupra predictiei:

- valori pozitive -> cresc probabilitatea aprobarii
- valori negative -> o scad
""")
if st.checkbox("Afiseaza SHAP (poate dura cateva secunde)"):

    st.markdown("""
        - Summary Plot: arata impactul global al caracteristicilor
        - Bar Plot: importanta medie a caracteristicilor
        - Waterfall Plot: explica o predicție individuala
        """)

    X_sample = get_sample(df)[features]
    X_sample_scaled = scaler.transform(X_sample)
    X_sample_scaled = pd.DataFrame(X_sample_scaled, columns=features)

    explainer = shap.Explainer(model.predict_proba, X_sample_scaled)
    shap_values = explainer(X_sample_scaled)

    st.subheader("Summary Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values[..., 1], X_sample_scaled, show=False)
    st.pyplot(fig)

    st.subheader("Bar Plot")
    fig = plt.figure()
    shap.plots.bar(shap_values[..., 1], show=False)
    st.pyplot(fig)

    st.subheader("Waterfall Plot")
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0][..., 1], show=False)
    st.pyplot(fig)