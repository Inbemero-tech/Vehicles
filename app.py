# ============================================
# 🚗 Vehicles Analysis Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# CONFIGURACIÓN INICIAL
# ==========================
st.set_page_config(
    page_title="Vehicle Market Analysis Dashboard", layout="wide")
st.title("🚗 Vehicle Market Analysis Dashboard")

# ==========================
# CARGA DE DATOS
# ==========================


def load_data():
    try:
        df = pd.read_csv("vehicles_us_clean.csv")
        st.success("Datos cargados correctamente ✅")
        return df
    except FileNotFoundError:
        st.error(
            "❌ No se encontró el archivo 'vehicles_us_clean.csv'. Verifica la ruta.")
        return pd.DataFrame()  # Retorna un DataFrame vacío para evitar errores


df = load_data()

# ==========================
# FILTROS INTERACTIVOS
# ==========================
st.sidebar.header("🔍 Filtros globales")

type_filter = st.sidebar.multiselect("Tipo de vehículo:", options=df["type"].dropna(
).unique(), default=df["type"].dropna().unique())
condition_filter = st.sidebar.multiselect("Condición:", options=df["condition"].dropna(
).unique(), default=df["condition"].dropna().unique())
year_range = st.sidebar.slider("Rango de años:", int(df["model_year"].min()), int(
    df["model_year"].max()), (int(df["model_year"].min()), int(df["model_year"].max())))
price_range = st.sidebar.slider("Rango de precios:", int(df["price"].min()), int(
    df["price"].max()), (int(df["price"].min()), int(df["price"].max())))

# ==========================
# RESUMEN GENERAL
# ==========================
st.header("📊 Resumen general del mercado")

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["condition"].isin(condition_filter)) &
    (df["model_year"].between(year_range[0], year_range[1])) &
    (df["price"].between(price_range[0], price_range[1]))
]

# Métricas principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vehículos disponibles", len(filtered_df))
col2.metric("Precio promedio", f"${filtered_df['price'].mean():,.0f}")
col3.metric("Año promedio", int(filtered_df["model_year"].mean()))
col4.metric("Km promedio", f"{filtered_df['odometer'].mean():,.0f}")

# ==========================
# VISUALIZACIONES
# ==========================
st.subheader("📈 Visualizaciones")

tab1, tab2, tab3 = st.tabs(
    ["Distribución de precios", "Precio vs Año/Kilometraje", "Precio promedio por tipo"])

with tab1:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["price"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribución de precios")
    ax.set_xlabel("Precio (USD)")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="model_year", y="price",
                    hue="odometer", palette="viridis", ax=ax)
    ax.set_title("Precio vs Año del vehículo y Kilometraje")
    st.pyplot(fig)

with tab3:
    avg_price = filtered_df.groupby("type")["price"].mean().sort_values()
    fig, ax = plt.subplots()
    avg_price.plot(kind="bar", ax=ax)
    ax.set_title("Precio promedio por tipo de vehículo")
    ax.set_ylabel("Precio promedio (USD)")
    st.pyplot(fig)

# ==========================
# ANÁLISIS DE CORRELACIONES
# ==========================
st.subheader("🧠 Análisis de correlaciones")

# Seleccionar solo columnas numéricas relevantes
num_cols = ["price", "model_year", "odometer", "days_listed"]
corr_df = df[num_cols].corr()

col1, col2 = st.columns([2, 3])

with col1:
    st.write("Este análisis nos permite observar la relación entre las variables numéricas del conjunto de datos y el precio de los vehículos.")
    st.write("- Una correlación positiva indica que, al aumentar una variable, también aumenta el precio.")
    st.write("- Una correlación negativa indica que, al aumentar una variable, el precio tiende a disminuir.")
    st.write(
        "- Los valores más cercanos a **1 o -1** representan relaciones más fuertes.")
    st.dataframe(corr_df.style.background_gradient(
        cmap='coolwarm').format("{:.2f}"))

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm',
                fmt=".2f", linewidths=0.5)
    ax.set_title("Mapa de calor de correlaciones")
    st.pyplot(fig)

# Correlaciones categóricas con precio (calculadas por promedio de precio por categoría)
st.markdown("### 💬 Correlaciones categóricas aproximadas")
cat_cols = ["condition", "fuel", "transmission", "type"]
corr_cat = {}

for col in cat_cols:
    if col in df.columns:
        mean_prices = df.groupby(
            col)["price"].mean().sort_values(ascending=False)
        corr_cat[col] = mean_prices

for col, data in corr_cat.items():
    st.write(f"**{col.capitalize()}** - Precio promedio por categoría:")
    st.bar_chart(data)

# ==========================
# MODELO PREDICTIVO SIMPLE
# ==========================
st.subheader("🤖 Entrenamiento del modelo predictivo")

# Variables para el modelo
features = ["model_year", "odometer",
            "condition", "type", "fuel", "transmission"]
df_model = df.dropna(subset=features + ["price"])

# Separación de variables
X = df_model[features]
y = df_model["price"]

# Preprocesamiento: codificar variables categóricas
categorical = ["condition", "type", "fuel", "transmission"]
numeric = ["model_year", "odometer"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

# Pipeline de entrenamiento
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("MAE (Error absoluto medio)", f"${mae:,.0f}")
col2.metric("R² (Coeficiente de determinación)", f"{r2:.2f}")

# Visualización de predicciones
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
ax.set_xlabel("Precio real")
ax.set_ylabel("Precio predicho")
ax.set_title("Comparación de precios reales vs predichos")
st.pyplot(fig)

# ==========================
# PREDICCIÓN INTERACTIVA
# ==========================
st.subheader("💡 Simulador de predicción de precio")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    model_year = col1.number_input(
        "Año del modelo", min_value=1980, max_value=2025, value=2015)
    odometer = col2.number_input(
        "Kilometraje (en millas)", min_value=0, max_value=500000, value=60000)
    condition = col3.selectbox("Condición", df["condition"].dropna().unique())

    col4, col5, col6 = st.columns(3)
    type_v = col4.selectbox("Tipo de vehículo", df["type"].dropna().unique())
    fuel = col5.selectbox("Combustible", df["fuel"].dropna().unique())
    transmission = col6.selectbox(
        "Transmisión", df["transmission"].dropna().unique())

    submit = st.form_submit_button("🔮 Estimar precio")

if submit:
    new_data = pd.DataFrame({
        "model_year": [model_year],
        "odometer": [odometer],
        "condition": [condition],
        "type": [type_v],
        "fuel": [fuel],
        "transmission": [transmission]
    })

    predicted_price = model.predict(new_data)[0]
    st.success(
        f"💰 El precio estimado del vehículo es de **${predicted_price:,.0f} USD**")

# ==========================
# FIN
# ==========================
st.caption("Desarrollado por Inti — Análisis de datos vehiculares 🚗📊")
