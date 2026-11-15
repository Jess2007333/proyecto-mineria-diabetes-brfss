import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Riesgo de diabetes – Proyecto académico", page_icon="🩺", layout="centered")

RUTA_MODELO = Path(__file__).parent / "modelo_diabetes.joblib"
UMBRAL = 0.5

EDAD = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
        8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80 o más"}
SALUD = {1: "1 – Excelente", 2: "2 – Muy buena", 3: "3 – Buena", 4: "4 – Regular", 5: "5 – Mala"}
EDUC = {1: "Nunca asistió o solo preescolar", 2: "Primaria", 3: "Secundaria incompleta",
        4: "Bachiller", 5: "Estudios superiores incompletos", 6: "Título universitario"}
INGRESO = {1: "Menos de 10 000 USD", 2: "10 000–15 000", 3: "15 000–20 000", 4: "20 000–25 000",
           5: "25 000–35 000", 6: "35 000–50 000", 7: "50 000–75 000", 8: "75 000 o más"}
RANGOS = {"bmi": (12.0, 98.0), "menthlth": (0, 30), "physhlth": (0, 30)}
BINARIAS = ["highbp", "highchol", "cholcheck", "smoker", "stroke", "heartdiseaseorattack", "physactivity",
            "fruits", "veggies", "hvyalcoholconsump", "anyhealthcare", "nodocbccost", "diffwalk", "sex"]
CATEGORICAS = {"genhlth": (1, 5), "age": (1, 13), "education": (1, 6), "income": (1, 8)}
CAMPOS = list(RANGOS) + BINARIAS + list(CATEGORICAS)


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


def validar(datos: dict) -> list:
    errores = []
    for c in CAMPOS:
        if c not in datos or datos[c] is None:
            errores.append(f"Falta el campo '{c}'.")
    if errores:
        return errores
    for c, (lo, hi) in {**RANGOS, **CATEGORICAS}.items():
        if not lo <= datos[c] <= hi:
            errores.append(f"'{c}' debe estar entre {lo} y {hi}.")
    for c in BINARIAS:
        if datos[c] not in (0, 1):
            errores.append(f"'{c}' debe ser 0 o 1.")
    return errores


def predecir(datos: dict):
    modelo = cargar_modelo()
    fila = pd.DataFrame([{c: float(datos[c]) for c in CAMPOS}])
    proba = float(modelo.predict_proba(fila)[0, 1])
    return proba, int(proba >= UMBRAL)


def si_no(etiqueta, clave, ayuda=None, defecto=0):
    return 1 if st.radio(etiqueta, ["No", "Sí"], index=defecto, horizontal=True, key=clave, help=ayuda) == "Sí" else 0


st.title("🩺 Estimación del riesgo de diabetes")
st.caption("Proyecto de aula – Minería de Datos, Universidad de la Costa (CUC). Modelo: Gradient Boosting entrenado con BRFSS 2015.")
st.warning("Herramienta **académica**. Sus resultados no constituyen un diagnóstico ni sustituyen una valoración profesional de salud.")

st.subheader("Datos del paciente")
c1, c2 = st.columns(2)
with c1:
    sex = 1 if st.radio("Sexo biológico", ["Femenino", "Masculino"], horizontal=True, key="sex") == "Masculino" else 0
    age = st.selectbox("Rango de edad (años)", list(EDAD), format_func=EDAD.get, index=7, key="age")
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=12.0, max_value=98.0, value=27.0, step=0.1, key="bmi")
    genhlth = st.selectbox("Salud general autopercibida", list(SALUD), format_func=SALUD.get, index=2, key="genhlth")
    menthlth = st.number_input("Días de mala salud mental (últimos 30)", 0, 30, 0, key="menthlth")
    physhlth = st.number_input("Días de mala salud física (últimos 30)", 0, 30, 0, key="physhlth")
    education = st.selectbox("Nivel educativo", list(EDUC), format_func=EDUC.get, index=4, key="education")
    income = st.selectbox("Ingreso anual del hogar (USD)", list(INGRESO), format_func=INGRESO.get, index=5, key="income")
with c2:
    highbp = si_no("Presión arterial alta", "highbp")
    highchol = si_no("Colesterol alto", "highchol")
    cholcheck = si_no("Control de colesterol en los últimos 5 años", "cholcheck", defecto=1)
    smoker = si_no("Ha fumado 100 cigarrillos o más en su vida", "smoker")
    stroke = si_no("Antecedente de accidente cerebrovascular", "stroke")
    heartdiseaseorattack = si_no("Enfermedad coronaria o infarto", "heartdiseaseorattack")
    physactivity = si_no("Actividad física en los últimos 30 días", "physactivity", defecto=1)
    fruits = si_no("Consume fruta a diario", "fruits", defecto=1)
    veggies = si_no("Consume verduras a diario", "veggies", defecto=1)
    hvyalcoholconsump = si_no("Consumo elevado de alcohol", "hvyalcoholconsump")
    anyhealthcare = si_no("Tiene cobertura de salud", "anyhealthcare", defecto=1)
    nodocbccost = si_no("No consultó al médico por costo (último año)", "nodocbccost")
    diffwalk = si_no("Dificultad seria para caminar o subir escaleras", "diffwalk")

if st.button("Calcular riesgo", type="primary", key="calcular"):
    datos = {c: globals()[c] for c in CAMPOS}
    errores = validar(datos)
    if errores:
        st.error("Revise los datos: " + " ".join(errores))
    else:
        proba, clase = predecir(datos)
        st.session_state["ultima_prob"], st.session_state["ultima_clase"] = proba, clase
        st.subheader("Resultado")
        st.metric("Probabilidad estimada de prediabetes/diabetes", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
        if clase == 1:
            st.error("El modelo clasifica el perfil como **riesgo alto** (probabilidad ≥ 50 %). Se recomienda valoración clínica.")
        else:
            st.success("El modelo clasifica el perfil como **riesgo bajo** (probabilidad < 50 %). Esto no descarta la enfermedad.")
        st.caption("El modelo se entrenó con datos de encuesta de EE. UU. (2015) balanceados 50/50; la probabilidad no equivale a la prevalencia real. "
                   "Su sensibilidad en prueba fue cercana al 80 %: aproximadamente 1 de cada 5 casos positivos no es detectado.")
