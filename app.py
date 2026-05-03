import streamlit as st

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide"
)

st.markdown("""
<style>
/* rotunjire imagini */
img {
    border-radius: 12px;
}

/* card style pentru coloane */
section[data-testid="stHorizontalBlock"] > div {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
}

/* spațiere text */
h1, h2, h3 {
    margin-top: 10px;
}

/* linie separator */
hr {
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

st.title("Machine Learning App")

st.markdown("""
## Despre proiect

Aceasta aplicatie interactiva prezinta doua probleme de Machine Learning:

- Regresie (Wine Quality) – prezicerea calitatii vinului  
- Clasificare (Loan Prediction) – aprobarea unui credit  


## Functionalitati

- Analiza exploratorie a datelor (EDA)
- Compararea modelelor de machine learning
- Introducerea valorilor pentru predictii
- Metrici de evaluare
- Curbe de invatare (learning curves)
- Explicabilitate folosind SHAP


Selecteaza o pagina din meniul din stanga pentru a incepe.
""")

st.markdown("## ")

st.header("Sectiuni aplicatie")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.image("assets/red-wine.jpg", use_container_width=True)
    st.markdown("### Regresie - Wine Quality")
    st.markdown("""
Prezice calitatea vinului pe baza caracteristicilor chimice.
""")


with col2:
    st.image("assets/loan.jpg", use_container_width=True)
    st.markdown("### Clasificare - Loan Prediction")
    st.markdown("""
Determina daca un credit va fi aprobat sau respins.
""")
