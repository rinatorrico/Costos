import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import numpy as np
from math import sqrt

st.title("🔎 Análisis Inteligente de Costos de Producción")

DATASET = "dataset_contabilidad_costos.csv"

# 1. Cargar y explorar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv(DATASET)

df = cargar_datos()
st.subheader("Vista previa del dataset")
st.dataframe(df.head())

st.markdown("*Tamaño del dataset:* {} filas, {} columnas".format(df.shape[0], df.shape[1]))

# Validar columnas esperadas
features = [
    'Unidades_Producidas',
    'Costo_Materia_Prima',
    'Costo_Mano_Obra',
    'Costo_Indirecto',
    'Horas_Maquina'
]
columnas_esperadas = features + ['Centro_Costo', 'Costo_Unitario']
faltantes = [col for col in columnas_esperadas if col not in df.columns]
if faltantes:
    st.error(f"❌ El archivo CSV no contiene las siguientes columnas requeridas: {faltantes}")
    st.stop()

# 2. Limpieza y codificación
df_clean = df.copy()
label_encoder = LabelEncoder()
df_clean['Centro_Costo_Cod'] = label_encoder.fit_transform(df_clean['Centro_Costo'])

# ----------------------------------------
# 3. MODELO DE REGRESIÓN – Costo Unitario
# ----------------------------------------
st.header("📊 Predicción del Costo Unitario")

X_reg = df_clean[features]
y_reg = df_clean['Costo_Unitario']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

rmse = sqrt(mean_squared_error(y_test_r, y_pred_r))
st.subheader(f"RMSE del modelo de regresión: {rmse:.2f}")

st.subheader("Importancia de características (regresión)")
importancias_r = pd.DataFrame({
    "Característica": features,
    "Importancia": reg_model.feature_importances_
})
st.bar_chart(importancias_r.set_index("Característica"))

# ----------------------------------------
# 4. MODELO DE CLASIFICACIÓN – Centro de Costo
# ----------------------------------------
st.header("🏷 Clasificación del Centro de Costo")

X_clf = df_clean[features]
y_clf = df_clean['Centro_Costo_Cod']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)

acc = clf_model.score(X_test_c, y_test_c)
st.subheader(f"Precisión del modelo de clasificación: {acc:.2%}")

# Matriz de confusión
cm = confusion_matrix(y_test_c, y_pred_c)
st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# Clasificación detallada
st.text("Reporte de Clasificación")
st.text(classification_report(y_test_c, y_pred_c, target_names=label_encoder.classes_))

# ----------------------------------------
# 5. Formulario interactivo de predicción
# ----------------------------------------
st.header("📝 Formulario de predicción personalizada")

with st.form("form_pred"):
    unidades = st.number_input("Unidades Producidas", min_value=0.0, max_value=100000.0, step=1.0)
    mp = st.number_input("Costo Materia Prima", min_value=0.0, max_value=100000.0, step=0.5)
    mo = st.number_input("Costo Mano de Obra", min_value=0.0, max_value=100000.0, step=0.5)
    ci = st.number_input("Costo Indirecto", min_value=0.0, max_value=100000.0, step=0.5)
    hm = st.number_input("Horas Máquina", min_value=0.0, max_value=5000.0, step=0.5)
    submit = st.form_submit_button("Predecir")

    if submit:
        entrada = pd.DataFrame([[unidades, mp, mo, ci, hm]], columns=features)
        pred_costo = reg_model.predict(entrada)[0]
        pred_centro_cod = clf_model.predict(entrada)[0]
        pred_centro = label_encoder.inverse_transform([pred_centro_cod])[0]

        st.success(f"🔹 Costo Unitario estimado: Bs {pred_costo:.2f}")
        st.info(f"🏷 Centro de Costo estimado: {pred_centro}")

# ----------------------------------------
# 6. Reflexión educativa
# ----------------------------------------
st.markdown("---")
st.markdown("### 🎓 Reflexión")
st.write(
    "Este proyecto demuestra cómo la Inteligencia Artificial puede aplicarse a la contabilidad de costos, "
    "mejorando la precisión y automatización de decisiones operativas y financieras."
)