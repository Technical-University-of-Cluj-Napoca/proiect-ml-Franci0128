import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def get_sample(df):
    return df.drop("quality", axis=1).sample(100, random_state=42)


@st.cache_resource
def get_explainer(model, X):
    return shap.Explainer(model, X)

st.set_page_config(layout="wide")
st.title("Regresie - Wine Quality")


st.markdown("""
## Descriere problema

Aceasta aplicatie are ca scop estimarea calitatii vinului pe baza unor caracteristici chimice masurate in laborator. 
Datasetul utilizat contine informatii despre compozitia vinului, precum aciditatea, continutul de alcool, nivelul de sulfuri si alti indicatori relevanti.

Problema abordata este una de regresie, deoarece variabila tinta (*quality*) este numerica si reprezinta un scor al calitatii vinului.

In cadrul aplicatiei sunt utilizate mai multe modele de machine learning antrenate anterior si optimizate prin ajustarea hiperparametrilor. 
Utilizatorul poate selecta modelul dorit, introduce valori pentru caracteristici si obtine o predictie in timp real.

De asemenea, aplicatia ofera functionalitati de analiza exploratorie a datelor, compararea performantei modelelor si interpretarea rezultatelor folosind metoda SHAP, care permite intelegerea contributiei fiecarei caracteristici la predictia finala.
""")


df = pd.read_csv("winequality-red.csv", sep=";")

st.subheader("Informatii despre dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Numar observatii", df.shape[0])

with col2:
    st.metric("Numar caracteristici", df.shape[1] - 1)

st.write("Primele 5 randuri din dataset:")
st.dataframe(df.head())

st.header("Analiza exploratorie")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["quality"], bins=10, ax=ax)
    ax.set_title("Distributia calitatii")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Matrice corelatii")
    st.pyplot(fig)

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
cat_model = joblib.load("models/catboost_model.pkl")
ebm_model = joblib.load("models/ebm_model.pkl")
svr_model = joblib.load("models/svr_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

models = {
    "Random Forest Tuned": rf_model,
    "XGBoost Tuned": xgb_model,
    "CatBoost Tuned": cat_model,
    "EBM Tuned": ebm_model,
    "SVR Tuned": svr_model
}

st.header("Selectare model")

model_name = st.selectbox("Alege model:", list(models.keys()))
model = models[model_name]

st.info("""
Modelul selectat este unul dintre cele mai performante modele obtinute dupa etapa de optimizare a hiperparametrilor.
Acesta a fost antrenat anterior si este utilizat aici pentru predictii rapide.
""")

st.header("Hiperparametri model")

st.markdown("""
Mai jos sunt valorile hiperparametrilor pentru modelul selectat.
Acestea au fost obtinute in urma procesului de optimizare.
""")

st.write(model.get_params())


st.header("Predicție")

st.markdown("""
Introduceti valorile caracteristicilor chimice ale vinului folosind slider-ele de mai jos.
Aceste valori vor fi folosite pentru a estima calitatea vinului.
""")

input_data = {}

for col in features:
    input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])

if model_name == "SVR Tuned":
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
else:
    prediction = model.predict(input_df)[0]

st.metric("Calitate estimată", f"{prediction:.2f}")

st.header("Performanță model selectat")

X_eval = df.drop("quality", axis=1)
y_eval = df["quality"]

if model_name == "SVR Tuned":
    X_eval = scaler.transform(X_eval)

y_pred = model.predict(X_eval)

rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
r2 = r2_score(y_eval, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric("RMSE", f"{rmse:.3f}")

with col2:
    st.metric("R²", f"{r2:.3f}")

st.subheader("Predicții vs valori reale")

fig, ax = plt.subplots()
ax.scatter(y_eval, y_pred, alpha=0.5)
ax.plot([y_eval.min(), y_eval.max()],
        [y_eval.min(), y_eval.max()],
        'r--')
ax.set_xlabel("Valori reale")
ax.set_ylabel("Predicții")
ax.set_title("Predicții vs Real")

st.pyplot(fig)

st.header("Learning Curve")

st.markdown("""
Curba de invatare arata cum evolueaza performanta modelului in functie de dimensiunea setului de antrenare.

- diferenta mare intre train si validation → overfitting
- valori apropiate -> model stabil
""")

X_full = df.drop("quality", axis=1)
y_full = df["quality"]

if model_name == "SVR Tuned":
    X_full_scaled = scaler.transform(X_full)
    X_used_lc = X_full_scaled
else:
    X_used_lc = X_full

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_used_lc,
    y_full,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)

fig, ax = plt.subplots()
ax.plot(train_sizes, train_mean, label="Train")
ax.plot(train_sizes, val_mean, label="Validation")
ax.set_title("Learning Curve")
ax.set_xlabel("Dimensiune set antrenare")
ax.set_ylabel("Eroare (MSE)")
ax.legend()

st.pyplot(fig)

st.info("""
Diferenta dintre curbe indica daca modelul sufera de overfitting sau generalizeaza bine.
""")

st.header("Comparare modele")

st.markdown("""
Tabelul de mai jos prezinta performanta celor mai bune modele dupa ajustarea hiperparametrilor.

- RMSE: eroarea medie (mai mic = mai bun)
- R2: cat de bine explica modelul variatia datelor (mai mare = mai bun)
""")

comparison_data = pd.DataFrame({
    "Model": [
        "Random Forest Tuned",
        "CatBoost Tuned",
        "SVR Tuned",
        "EBM Tuned",
        "XGBoost Tuned"
    ],
    "RMSE": [0.563, 0.565, 0.586, 0.586, 0.587],
    "R2": [0.487, 0.485, 0.445, 0.445, 0.443]
})

st.success("Modelul Random Forest Tuned ofera cea mai buna performanta pe acest dataset.")

st.dataframe(comparison_data)

st.markdown("""
SHAP este o metoda de interpretabilitate care arata contributia fiecarei caracteristici la predictia finala.

- valorile pozitive cresc predictia
- valorile negative o scad
- importanta caracteristicilor este determinata global si local
""")

if st.checkbox("Afiseaza SHAP (poate dura cateva secunde)"):

    use_full = st.checkbox("Foloseste tot datasetul (lent)")

    st.markdown("""
    - Summary Plot: arata impactul global al caracteristicilor
    - Bar Plot: importanta medie a caracteristicilor
    - Waterfall Plot: explica o predicție individuala
    """)

    if use_full:
        X_used = df.drop("quality", axis=1)[features]
    else:
        X_used = get_sample(df)[features]

    # 👇 AICI îl pui (important)
    if model_name == "SVR Tuned" and use_full:
        st.warning("Limitare automată la 200 exemple pentru performanță")
        X_used = X_used[:200]

    # 👇 apoi vine SHAP
    if model_name == "SVR Tuned":
        X_scaled = scaler.transform(X_used)

        explainer = shap.KernelExplainer(model.predict, X_scaled[:100])
        shap_values = explainer.shap_values(X_scaled[:50])

        X_shap = X_scaled[:50]
    else:
        explainer = shap.Explainer(model, X_used)
        shap_values = explainer(X_used)

        X_shap = X_used

    st.subheader("Summary Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    st.pyplot(fig)

    st.subheader("Bar Plot")
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

    st.subheader("Waterfall Plot")
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)