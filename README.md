# Predicción del riesgo de diabetes (BRFSS 2015) – Proyecto de aula, Minería de Datos (CUC)

Autores: Jesús Alberto Vargas España

Clasificación supervisada (Gradient Boosting y comparación con 4 modelos y una línea base) y segmentación con K-Means sobre *Diabetes Health Indicators* (versión 50/50). Herramienta **académica**: no sustituye una valoración profesional.

## Contenido
- `Proyecto_Mineria_Corte3_Final.ipynb`: notebook final ejecutado (EDA, pipeline, validación cruzada, evaluación, K-Means). Genera `modelo_diabetes.joblib`.
- `app.py`: aplicación Streamlit que carga el mismo pipeline.
- `modelo_diabetes.joblib`: pipeline (escalado + Gradient Boosting) entrenado con scikit-learn 1.9.1.
- `requirements.txt`, `tests/test_app.py` (pruebas con `streamlit.testing`).

## Ejecución
```bash
pip install -r requirements.txt
streamlit run app.py
```
Para reproducir el notebook, coloque `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (Kaggle) junto al notebook; en Colab se pide subirlo. Semilla global: 42.

## Enlaces
- Aplicación pública: _pendiente de completar tras el despliegue_
- Repositorio: _pendiente_

## Declaración de uso de IA
Ver el Anexo B del artículo final (herramientas, prompts, verificación y evaluación crítica).
