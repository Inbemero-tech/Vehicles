# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(
    page_title="Análisis de Vehículos en Venta - EE. UU.",
    layout="wide",
    page_icon="🚗"
)

st.title("🚗 Análisis de Vehículos en Venta en EE. UU.")
st.markdown("""
Explora los datos de vehículos en venta, filtra por tipo o condición, 
y analiza las tendencias de precios, kilometraje y más.
""")

# ==============================
# CARGA DE DATOS
# ==============================


@st.cache_data
def load_data():
    df = pd.read_csv("vehicles_us.csv")
    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price", "model_year", "odometer", "type"])


df = load_data()

# ==============================
# SIDEBAR DE FILTROS
# ==============================
st.sidebar.header("🔍 Filtros de Exploración")

type_filter = st.sidebar.multiselect(
    "Tipo de vehículo:",
    options=df["type"].unique(),
    default=df["type"].unique()
)

condition_filter = st.sidebar.multiselect(
    "Condición:",
    options=df["condition"].unique(),
    default=df["condition"].unique()
)

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["condition"].isin(condition_filter))
]

# ==============================
# TABLA INTERACTIVA COMPLETA
# ==============================
st.markdown("### 📋 Conjunto de Datos Completo")
st.markdown("""
Puedes desplazarte horizontal y verticalmente para explorar todos los registros disponibles.
Usa los filtros en la barra lateral para actualizar la tabla en tiempo real.
""")

# Mostramos toda la tabla con scroll
st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)

# ==============================
# SECCIÓN: VISUALIZACIONES
# ==============================
st.markdown("## 📊 Visualizaciones")

# Histograma de precios
st.markdown("### 💰 Distribución de precios de vehículos")
st.markdown("""
El siguiente histograma muestra cómo se distribuyen los precios de los vehículos filtrados. 
Permite observar concentraciones de valores y detectar posibles outliers.
""")
fig_price = px.histogram(
    filtered_df, x="price", nbins=50, color_discrete_sequence=["#76b5c5"]
)
fig_price.update_layout(xaxis_title="Precio (USD)", yaxis_title="Frecuencia")
st.plotly_chart(fig_price, use_container_width=True)

# Gráfico de dispersión
st.markdown("### 🚘 Relación entre kilometraje y precio")
st.markdown("""
Cada punto representa un vehículo. 
Podrás notar cómo los precios tienden a disminuir a medida que el kilometraje aumenta.
""")
fig_scatter = px.scatter(
    filtered_df,
    x="odometer",
    y="price",
    color="type",
    hover_data=["model_year", "condition"],
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_scatter.update_layout(
    xaxis_title="Kilometraje (millas)", yaxis_title="Precio (USD)")
st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================
# SECCIÓN: ANÁLISIS CORRELACIONAL
# ==============================
st.markdown("## 🔍 Análisis Correlacional")

st.markdown("""
El mapa de calor muestra las relaciones entre las variables numéricas. 
Los valores cercanos a 1 o -1 indican relaciones fuertes (positivas o negativas).
""")

corr = filtered_df[["price", "model_year", "odometer", "days_listed"]].corr()
corr.columns = ["Precio", "Año Modelo", "Kilometraje", "Días Publicado"]
corr.index = ["Precio", "Año Modelo", "Kilometraje", "Días Publicado"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
st.pyplot(fig)

# ==============================
# SECCIÓN: SIMULADOR DE PRECIO
# ==============================
st.markdown("## 🎯 Simulador de Precio Estimado")

st.markdown("""
Usa los controles para simular un precio estimado de venta basado en el año, 
kilometraje y tipo de vehículo.
""")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.slider("Año del modelo", int(
        df["model_year"].min()), int(df["model_year"].max()), 2015)

with col2:
    odometer = st.slider("Kilometraje (millas)", 0, int(
        df["odometer"].max()), 60000, step=5000)

with col3:
    vehicle_type = st.selectbox("Tipo de vehículo", df["type"].unique())

# Modelo simple para simulación de precio
base_price = df["price"].mean()
price_est = base_price + \
    (year - df["model_year"].mean()) * 150 - (odometer / 1000) * 20

st.metric(
    label=f"💵 Precio estimado para un {vehicle_type}", value=f"${price_est:,.0f} USD")

st.markdown("---")
st.caption("Aplicación desarrollada para análisis exploratorio de datos de vehículos - Proyecto académico 🧠")
