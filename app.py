import streamlit as st

# Configuración de la página (debe ser el primer comando de Streamlit)
st.set_page_config(
    page_title="Data Analytics",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import Counter, defaultdict

from src.download import download_csv_from_url
from src.create_plot import create_plot
from src.advanced import create_advanced_plot, advanced_statistical_analysis
warnings.filterwarnings('ignore')

# Configurar tema oscuro y responsive
st.markdown("""
<style>
    #MainMenu, header, footer {
        visibility: hidden;
    }

    /* Estilos globales */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Contenedores y elementos principales */
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        border-right: 1px solid #333333;
    }
    
    /* Tablas y DataFrames */
    .stDataFrame {
        background-color: #1e1e1e;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stDataFrame th {
        background-color: #2e2e2e !important;
        color: #ffffff !important;
        padding: 1rem !important;
    }
    
    .stDataFrame td {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        padding: 0.75rem !important;
    }
    
    /* Media queries para responsividad */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .stDataFrame {
            font-size: 14px;
        }
        
        .stButton>button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Título principal con estilo personalizado
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='color: #ffffff; text-shadow: 2px 2px 4px #000000; font-size: 2.5rem;'>
        📊 Plataforma de Análisis de Datos
    </h1>
    <p style='color: #cccccc; font-size: 1.2rem; margin-top: 1rem;'>
        Explora, analiza y comprende tus datos en una sola plataforma.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar para navegación con estilo mejorado
st.sidebar.markdown("""
<div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #333333;'>
    <h2 style='color: #ffffff; font-size: 1.5rem;text-transform: uppercase;'>🎮 Panel de Control</h2>
</div>
""", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["🔗 Descargar CSV desde URL", "📈 Análisis de Datos", "🛠️ Manipulación de Datos", "📊 Análisis Avanzado"],
    help="Selecciona la funcionalidad que deseas utilizar",
    key="main_navigation"
)

# Inicializar session state
if 'df' not in st.session_state:
    st.session_state.df = None

# ========== SECCIÓN DE DESCARGA ==========
if option == "🔗 Descargar CSV desde URL":
    st.header("🔗 Descarga de Archivos CSV")
    st.markdown("""
        ⚽ Esta sección está diseñada exclusivamente para descargar y analizar archivos CSV provenientes de la plataforma **football-data.co.uk**, especializada en datos estadísticos de fútbol.
    """)
    st.markdown("""
        📁 Si deseas analizar archivos CSV de otro tipo (por ejemplo, datos de ventas, finanzas u otras áreas), por favor utiliza la sección para subir un archivo local.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url_input = st.text_input(
            "URL del archivo CSV:",
            placeholder="https://ejemplo.com/archivo.csv o mmz4281/2324/E0.csv",
            help="Puedes usar URLs completas o relativas"
        )
    
    with col2:
        base_url = st.text_input(
            "URL base (opcional):",
            placeholder="https://sitio.com/",
            help="Para URLs relativas"
        )
    
    if st.button("🔽 Descargar CSV", type="primary"):
        if url_input:
            with st.spinner("Descargando archivo..."):
                df, error = download_csv_from_url(url_input, base_url)
                
                if df is not None:
                    st.session_state.df = df
                    st.success("✅️ ¡Archivo descargado exitosamente!")
                    
                    # Mostrar información básica
                    st.subheader("Información del Dataset")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Filas", df.shape[0])
                    with col2:
                        st.metric("Columnas", df.shape[1])
                    with col3:
                        st.metric("Tamaño", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    
                    # Vista previa
                    st.subheader("Vista Previa de los Datos")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                else:
                    st.error(f"❌ Error al descargar el archivo: {error}")
        else:
            st.warning("⚠️ Por favor, ingresa una URL válida")
    
    # Opción para subir archivo local
    st.divider()
    st.subheader("💻 O sube un archivo CSV local")
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("✅️ ¡Archivo cargado exitosamente!")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")

# ========== SECCIÓN DE MANIPULACIÓN ==========
elif option == "🛠️ Manipulación de Datos":
    st.header("Manipulación de Datos")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Pestañas para diferentes operaciones
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🗑️ Eliminar", "➕ Agregar", "🔄 Ordenar", "🔍 Valores Nulos", "✏️ Renombrar y Formatear", "🔄 Buscar y Reemplazar"])
        
        with tab1:
            st.subheader("Eliminar Filas y Columnas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Eliminar Columnas**")
                cols_to_delete = st.multiselect(
                    "Selecciona columnas a eliminar:",
                    df.columns.tolist()
                )
                
                if cols_to_delete and st.button("Eliminar Columnas Seleccionadas"):
                    df = df.drop(columns=cols_to_delete)
                    st.session_state.df = df
                    st.success(f"Columnas eliminadas: {', '.join(cols_to_delete)}")
                    st.rerun()
            
            with col2:
                st.write("**Eliminar Filas**")
                
                # Eliminar por índice
                row_indices = st.text_input(
                    "Índices de filas a eliminar (separados por coma):",
                    placeholder="0,1,5,10"
                )
                
                if row_indices and st.button("Eliminar Filas por Índice"):
                    try:
                        indices = [int(x.strip()) for x in row_indices.split(',')]
                        df = df.drop(index=indices)
                        st.session_state.df = df
                        st.success(f"Filas eliminadas: {row_indices}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                # Eliminar duplicados
                if st.button("Eliminar Filas Duplicadas"):
                    initial_rows = len(df)
                    df = df.drop_duplicates()
                    final_rows = len(df)
                    st.session_state.df = df
                    st.success(f"Se eliminaron {initial_rows - final_rows} filas duplicadas")
                    st.rerun()
        
        with tab2:
            st.subheader("Agregar Columnas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_col_name = st.text_input("Nombre de la nueva columna:")
                new_col_type = st.selectbox(
                    "Tipo de columna:",
                    ["Valor constante", "Cálculo basado en otras columnas", "Secuencia numérica"]
                )
            
            with col2:
                if new_col_type == "Valor constante":
                    const_value = st.text_input("Valor constante:")
                    if st.button("Agregar Columna Constante") and new_col_name:
                        df[new_col_name] = const_value
                        st.session_state.df = df
                        st.success(f"Columna '{new_col_name}' agregada")
                        st.rerun()
                
                elif new_col_type == "Cálculo basado en otras columnas":
                    col_calc = st.text_input(
                        "Fórmula (usa nombres de columnas):",
                        placeholder="columna1 + columna2"
                    )
                    if st.button("Agregar Columna Calculada") and new_col_name and col_calc:
                        try:
                            df[new_col_name] = eval(col_calc, {"__builtins__": {}}, df.to_dict('series'))
                            st.session_state.df = df
                            st.success(f"Columna '{new_col_name}' agregada")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error en la fórmula: {str(e)}")
                
                elif new_col_type == "Secuencia numérica":
                    start_val = st.number_input("Valor inicial:", value=1)
                    if st.button("Agregar Secuencia") and new_col_name:
                        df[new_col_name] = range(start_val, start_val + len(df))
                        st.session_state.df = df
                        st.success(f"Columna '{new_col_name}' agregada")
                        st.rerun()
        
        with tab3:
            st.subheader("Ordenar Columnas y Datos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Reordenar Columnas**")
                new_order = st.multiselect(
                    "Selecciona el orden de las columnas:",
                    df.columns.tolist(),
                    default=df.columns.tolist()
                )
                
                if st.button("Aplicar Nuevo Orden") and len(new_order) == len(df.columns):
                    df = df[new_order]
                    st.session_state.df = df
                    st.success("Orden de columnas actualizado")
                    st.rerun()
            
            with col2:
                st.write("**Ordenar Datos**")
                sort_cols = st.multiselect(
                    "Columnas para ordenar:",
                    df.columns.tolist()
                )
                sort_ascending = st.checkbox("Orden ascendente", value=True)
                
                if sort_cols and st.button("Ordenar Datos"):
                    df = df.sort_values(sort_cols, ascending=sort_ascending)
                    st.session_state.df = df
                    st.success("Datos ordenados")
                    st.rerun()
        
        with tab4:
            st.subheader("Análisis de Valores Nulos")
            
            null_info = df.isnull().sum()
            null_percentage = (df.isnull().sum() / len(df)) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Resumen de Valores Nulos**")
                null_df = pd.DataFrame({
                    'Columna': null_info.index,
                    'Valores Nulos': null_info.values,
                    'Porcentaje': null_percentage.values
                }).round(2)
                st.dataframe(null_df, use_container_width=True)
            
            with col2:
                st.write("**Acciones sobre Valores Nulos**")
                
                null_cols = null_info[null_info > 0].index.tolist()
                
                if null_cols:
                    selected_col = st.selectbox("Columna a tratar:", null_cols, key="null_column_select")
                    action = st.selectbox(
                        "Acción:",
                        ["Eliminar filas con nulos", "Rellenar con valor", "Rellenar con media/moda"],
                        key="null_action_select"
                    )
                    
                    if action == "Rellenar con valor":
                        fill_value = st.text_input("Valor para rellenar:")
                        if st.button("Aplicar Relleno") and fill_value:
                            df[selected_col] = df[selected_col].fillna(fill_value)
                            st.session_state.df = df
                            st.success(f"Valores nulos rellenados en '{selected_col}'")
                            st.rerun()
                    
                    elif action == "Rellenar con media/moda":
                        if df[selected_col].dtype in ['int64', 'float64']:
                            fill_val = df[selected_col].mean()
                            method = "media"
                        else:
                            fill_val = df[selected_col].mode().iloc[0] if not df[selected_col].mode().empty else "N/A"
                            method = "moda"
                        
                        if st.button(f"Rellenar con {method}"):
                            df[selected_col] = df[selected_col].fillna(fill_val)
                            st.session_state.df = df
                            st.success(f"Valores nulos rellenados con {method} en '{selected_col}'")
                            st.rerun()
                    
                    elif action == "Eliminar filas con nulos":
                        if st.button("Eliminar Filas"):
                            initial_rows = len(df)
                            df = df.dropna(subset=[selected_col])
                            final_rows = len(df)
                            st.session_state.df = df
                            st.success(f"Se eliminaron {initial_rows - final_rows} filas")
                            st.rerun()
                else:
                    st.info("¡Excelente! No hay valores nulos en el dataset")
        
        with tab5:
            st.subheader("Renombrar Columnas y Formatear Fechas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Renombrar Columnas**")
                selected_col = st.selectbox("Selecciona una columna:", df.columns.tolist(), key="rename_column_select")
                new_name = st.text_input("Nuevo nombre para la columna:")
                
                if st.button("Renombrar Columna") and new_name:
                    df = df.rename(columns={selected_col: new_name})
                    st.session_state.df = df
                    st.success(f"Columna '{selected_col}' renombrada a '{new_name}'")
                    st.rerun()
            
            with col2:
                st.write("**Formatear Fechas**")
                date_cols = [col for col in df.columns if df[col].dtype == 'object']
                selected_date_col = st.selectbox("Selecciona columna de fecha:", date_cols, key="date_column_select")
                date_format = st.text_input("Formato de fecha (ejemplo: %Y-%m-%d):", "%Y-%m-%d")
                
                if st.button("Formatear Fecha") and selected_date_col:
                    try:
                        df[selected_date_col] = pd.to_datetime(
                            df[selected_date_col],
                            format=date_format,
                            dayfirst=True,
                            errors='coerce'
                        ).dt.strftime(date_format)
                        st.session_state.df = df
                        st.success(f"Formato de fecha actualizado en '{selected_date_col}'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al formatear fecha: {str(e)}")
        
        with tab6:
            st.subheader("Buscar y Reemplazar Valores")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_col = st.selectbox("Selecciona la columna donde buscar:", df.columns.tolist(), key="search_column_select")
                search_value = st.text_input("Valor a buscar:")
            
            with col2:
                replace_value = st.text_input("Valor de reemplazo:")
                case_sensitive = st.checkbox("Distinguir mayúsculas/minúsculas", value=False)
            
            if st.button("Buscar y Reemplazar") and search_value and replace_value:
                try:
                    # Contar ocurrencias antes del reemplazo
                    if case_sensitive:
                        matches = df[search_col].astype(str).str.contains(search_value, regex=False, na=False).sum()
                        df[search_col] = df[search_col].astype(str).str.replace(search_value, replace_value, regex=False)
                    else:
                        matches = df[search_col].astype(str).str.contains(search_value, case=False, regex=False, na=False).sum()
                        df[search_col] = df[search_col].astype(str).str.replace(search_value, replace_value, case=False, regex=False)
                    
                    st.session_state.df = df
                    st.toast(f"🔄 {matches} ocurrencias reemplazadas", icon="✅")
                    st.success(f"Se reemplazaron {matches} ocurrencias de '{search_value}' por '{replace_value}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al realizar el reemplazo: {str(e)}")
        
        # Mostrar dataset actual
        st.divider()
        st.subheader("Dataset Actual")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.info("📁 Primero carga un archivo CSV en la sección 'Descargar CSV desde URL'")

# ========== SECCIÓN DE ANÁLISIS ==========
elif option == "📈 Análisis de Datos":
    st.header("Análisis y Visualización de Datos")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Estadísticas básicas
        st.subheader("📊 Estadísticas Descriptivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Información General**")
            st.write(f"- **Filas:** {df.shape[0]:,}")
            st.write(f"- **Columnas:** {df.shape[1]}")
            st.write(f"- **Memoria:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            st.write(f"- **Duplicados:** {df.duplicated().sum()}")
        
        with col2:
            st.write("**Tipos de Datos**")
            type_counts = df.dtypes.value_counts()
            for dtype, count in type_counts.items():
                st.write(f"- **{dtype}:** {count} columnas")
        
        # Mostrar estadísticas descriptivas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Estadísticas de Columnas Numéricas")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Sección de visualización
        st.divider()
        st.subheader("🎨 Crear Visualizaciones")
        
        plot_type = st.selectbox(
            "Tipo de gráfico:",
            ["Gráfico de Barras", "Gráfico Circular", "Heatmap",
            "Violin Plot", "Box Plot", "Área", "Burbujas"],
            key="plot_type"
        )
        
        # Contenedor para filtros y selección de columnas
        with st.expander("🔍 Configuración de Datos y Filtros", expanded=True):
            col1, col2 = st.columns(2)
            
            # Separar columnas por tipo
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            with col1:
                if plot_type in ["Gráfico de Barras", "Gráfico Circular"]:
                    x_column = st.selectbox("Columna Categórica (X):", categorical_cols, key="x_column_cat")
                else:
                    x_column = st.selectbox("Columna X:", df.columns.tolist(), key="x_column_all")
                
                # Mostrar valores únicos de la columna X para filtrar si es categórica
                if df[x_column].dtype in ['object', 'category']:
                    x_values = st.multiselect(
                        f"Filtrar valores de {x_column}:",
                        df[x_column].unique(),
                        default=list(df[x_column].unique())[:5]
                    )
            
            with col2:
                if plot_type == "Heatmap":
                    y_column = None
                    st.write("*Automático para correlación*")
                elif plot_type in ["Scatter Plot", "Burbujas", "Box Plot", "Violin Plot"]:
                    y_column = st.selectbox("Columna Numérica (Y):", [None] + numeric_cols, key=f"y_column_{plot_type}")
                else:
                    # Opciones de agregación para otros tipos de gráficos
                    agg_method = st.selectbox(
                        "Método de agregación:",
                        ["Conteo", "Suma", "Promedio", "Porcentaje"],
                        key="agg_method"
                    )
                    
                    if agg_method == "Suma" or agg_method == "Promedio":
                        y_column = st.selectbox("Columna para agregar:", numeric_cols, key="y_column_agg")
                        # Filtrar datos según los valores seleccionados de X
                        if 'x_values' in locals() and x_values:
                            df_filtered = df[df[x_column].isin(x_values)]
                        else:
                            df_filtered = df
                        # Preparar datos agregados
                        if agg_method == "Suma":
                            plot_data = df_filtered.groupby(x_column)[y_column].sum().reset_index()
                        else:  # Promedio
                            plot_data = df_filtered.groupby(x_column)[y_column].mean().reset_index()
                    elif agg_method == "Conteo":
                        y_column = "count"
                        if 'x_values' in locals() and x_values:
                            df_filtered = df[df[x_column].isin(x_values)]
                        else:
                            df_filtered = df
                        plot_data = df_filtered.groupby(x_column).size().reset_index(name=y_column)
                    else:  # Porcentaje
                        y_column = "percentage"
                        if 'x_values' in locals() and x_values:
                            df_filtered = df[df[x_column].isin(x_values)]
                        else:
                            df_filtered = df
                        counts = df_filtered.groupby(x_column).size()
                        plot_data = (counts / len(df_filtered) * 100).reset_index(name=y_column)
                        
                    if y_column:
                        y_min = float(plot_data[y_column].min())
                        y_max = float(plot_data[y_column].max())
                        # Si los valores son iguales, ajustar el rango
                        if y_min == y_max:
                            y_min = y_min * 0.95 if y_min != 0 else -1
                            y_max = y_max * 1.05 if y_max != 0 else 1
                        y_range = st.slider(
                            f"Rango de valores para {y_column}:",
                            min_value=y_min,
                            max_value=y_max,
                            value=(y_min, y_max)
                        )
                if plot_type in ["Scatter Plot", "Burbujas", "Box Plot", "Violin Plot"]:
                    y_column = st.selectbox("Columna Numérica (Y):", [None] + numeric_cols, key="y_column_numeric")
                    if y_column:
                        y_min, y_max = float(df[y_column].min()), float(df[y_column].max())
                        # Si los valores son iguales, ajustar el rango
                        if y_min == y_max:
                            y_min = y_min * 0.95 if y_min != 0 else -1
                            y_max = y_max * 1.05 if y_max != 0 else 1
                        y_range = st.slider(
                            f"Rango de valores para {y_column}:",
                            min_value=y_min,
                            max_value=y_max,
                            value=(y_min, y_max)
                        )
                else:
                    y_column = st.selectbox("Columna Y (opcional):", [None] + df.columns.tolist(), key="y_column_optional")
        
        # Opciones adicionales de personalización
        with st.expander("🎨 Opciones de Personalización", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Columna para color (opcional)
                color_column = st.selectbox(
                    "Columna para color:",
                    [None] + df.columns.tolist(),
                    key="color_column"
                )
                
                # Tamaño de marcadores (para scatter y bubble)
                if plot_type in ["Scatter Plot", "Burbujas"]:
                    marker_size = st.slider("Tamaño de marcadores:", 5, 30, 10)
            
            with col2:
                # Orientación para gráficos de barras
                if plot_type == "Gráfico de Barras":
                    orientation = st.radio("Orientación:", ["Vertical", "Horizontal"])
                
                # Mostrar valores en el gráfico
                show_values = st.checkbox("Mostrar valores en el gráfico", value=False)
        
        # Crear gráfico
        if st.button("🎯 Generar Gráfico", type="primary"):
            # Preparar datos base
            if 'x_values' in locals() and x_values:
                base_data = df[df[x_column].isin(x_values)]
            else:
                base_data = df.copy()
            
            # Aplicar agregación si corresponde
            if plot_type in ["Gráfico de Barras", "Gráfico de Líneas", "Área"] and 'agg_method' in locals():
                if agg_method == "Suma":
                    plot_data = base_data.groupby(x_column)[y_column].sum().reset_index()
                elif agg_method == "Promedio":
                    plot_data = base_data.groupby(x_column)[y_column].mean().reset_index()
                elif agg_method == "Conteo":
                    plot_data = base_data.groupby(x_column).size().reset_index(name="count")
                    y_column = "count"
                else:  # Porcentaje
                    counts = base_data.groupby(x_column).size()
                    plot_data = (counts / len(base_data) * 100).reset_index(name="percentage")
                    y_column = "percentage"
                
                st.info(f"📊 Mostrando {agg_method.lower()} de {y_column if agg_method in ['Suma', 'Promedio'] else 'valores'} por {x_column}")
            else:
                plot_data = base_data
            
            # Aplicar filtro de rango Y si existe
            if 'y_range' in locals() and y_column:
                plot_data = plot_data[
                    (plot_data[y_column] >= y_range[0]) &
                    (plot_data[y_column] <= y_range[1])
                ]

            fig = create_plot(
                df=plot_data,
                plot_type=plot_type,
                x_col=x_column,
                y_col=y_column,
                color_col=color_column,
                marker_size=marker_size if 'marker_size' in locals() else 10,
                orientation='h' if 'orientation' in locals() and orientation == "Horizontal" else 'v',
                show_values='show_values' in locals() and show_values
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Análisis rápido adicional
        st.divider()
        st.subheader("🔍 Análisis Rápido")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Ver Valores Únicos por Columna"):
                unique_info = pd.DataFrame({
                    'Columna': df.columns,
                    'Valores Únicos': [df[col].nunique() for col in df.columns],
                    'Tipo': [str(df[col].dtype) for col in df.columns]
                })
                st.dataframe(unique_info, use_container_width=True)
        
        with col2:
            if st.button("Detectar Outliers (Columnas Numéricas)"):
                if len(numeric_cols) > 0:
                    outlier_info = []
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                        outlier_info.append({
                            'Columna': col,
                            'Outliers': len(outliers),
                            'Porcentaje': f"{len(outliers)/len(df)*100:.2f}%"
                        })
                    
                    outlier_df = pd.DataFrame(outlier_info)
                    st.dataframe(outlier_df, use_container_width=True)
                else:
                    st.info("❌ No hay columnas numéricas para análisis de outliers")
        
        # Opción para descargar datos procesados
        st.divider()
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV Procesado",
            data=csv,
            file_name="datos_procesados.csv",
            mime="text/csv"
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos')
            processed_data = output.getvalue()

        st.download_button(
            label="📥 Descargar Excel Procesado",
            data=processed_data,
            file_name="datos_procesados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    else:
        st.info("📁 Primero carga un archivo CSV en la sección 'Descargar CSV desde URL'")

# ========== SECCIÓN DE ANÁLISIS AVANZADO ==========
elif option == "📊 Análisis Avanzado":
    st.header("Análisis Avanzado de Datos")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Selección del tipo de análisis
        analysis_type = st.selectbox(
            "Selecciona el tipo de análisis:",
            ["📈 Análisis de Tendencias", "🔍 Detección de Anomalías", "⏰ Series Temporales", "📊 Análisis de Distribución", "🔗 Análisis de Correlaciones", "🎯 Segmentación de Datos", "🔮 Análisis Predictivo", "🔄 Análisis de Secuencias"],
            key="analysis_type_select"
        )
        
        if analysis_type == "📈 Análisis de Tendencias":
            st.subheader("Análisis de Tendencias")
            
            # Selección de columnas para tendencias
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            trend_col = st.selectbox("Selecciona la columna para analizar tendencias:", numeric_cols, key="trend_column_select")
            
            if st.button("Analizar Tendencias"):
                # Calcular estadísticas de tendencia
                trend_data = df[trend_col].rolling(window=5).mean()
                
                # Crear gráfico de tendencias
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df[trend_col], name="Datos Originales"))
                fig.add_trace(go.Scatter(y=trend_data, name="Tendencia (Media Móvil)"))
                fig.update_layout(title=f"Análisis de Tendencias - {trend_col}", xaxis_title="Índice", yaxis_title=trend_col)
                st.plotly_chart(fig)
                
                # Mostrar estadísticas adicionales
                st.write("**Estadísticas de Tendencia:**")
                st.write(f"- Tendencia general: {'Creciente' if df[trend_col].corr(pd.Series(range(len(df)))) > 0 else 'Decreciente'}")
                st.write(f"- Variación total: {df[trend_col].max() - df[trend_col].min():.2f}")
        
        elif analysis_type == "🔍 Detección de Anomalías":
            st.subheader("Detección de Anomalías")
            
            # Selección de columna para anomalías
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            anomaly_col = st.selectbox("Selecciona la columna para detectar anomalías:", numeric_cols, key="anomaly_column_select")
            
            if st.button("Detectar Anomalías"):
                # Calcular estadísticas para anomalías
                Q1 = df[anomaly_col].quantile(0.25)
                Q3 = df[anomaly_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identificar anomalías
                anomalies = df[(df[anomaly_col] < lower_bound) | (df[anomaly_col] > upper_bound)]
                
                # Crear gráfico de anomalías
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[anomaly_col], name="Distribución"))
                fig.add_trace(go.Scatter(y=anomalies[anomaly_col], mode='markers', 
                                       name="Anomalías", marker=dict(color='red', size=10)))
                fig.update_layout(title=f"Detección de Anomalías - {anomaly_col}")
                st.plotly_chart(fig)
                
                # Mostrar estadísticas de anomalías
                st.write("**Resumen de Anomalías:**")
                st.write(f"- Número de anomalías: {len(anomalies)}")
                st.write(f"- Porcentaje de anomalías: {(len(anomalies)/len(df))*100:.2f}%")
                st.write(f"- Rango normal: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        elif analysis_type == "⏰ Series Temporales":
            st.subheader("Análisis de Series Temporales")
            
            # Identificar columnas de fecha
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) == 0:
                st.warning("No se detectaron columnas de fecha. Por favor, asegúrate de que al menos una columna esté en formato fecha.")
            else:
                # Selección de columnas para análisis temporal
                date_col = st.selectbox("Selecciona la columna de fecha:", date_cols, key="time_series_date_select")
                value_col = st.selectbox("Selecciona la columna de valores:", df.select_dtypes(include=[np.number]).columns, key="time_series_value_select")
                
                if st.button("Analizar Serie Temporal"):
                    # Preparar datos temporales
                    temp_df = df.sort_values(date_col)
                    temp_df.set_index(date_col, inplace=True)
                    
                    # Análisis de componentes temporales
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df[value_col], name="Serie Original"))
                    
                    # Agregar media móvil
                    rolling_mean = temp_df[value_col].rolling(window=7).mean()
                    fig.add_trace(go.Scatter(x=temp_df.index, y=rolling_mean, name="Media Móvil (7 días)"))
                    
                    fig.update_layout(title=f"Análisis Temporal - {value_col}", xaxis_title="Fecha", yaxis_title=value_col)
                    st.plotly_chart(fig)
                    
                    # Estadísticas temporales
                    st.write("**Estadísticas Temporales:**")
                    st.write(f"- Periodo total: {temp_df.index.max() - temp_df.index.min()}")
                    st.write(f"- Frecuencia promedio: {(temp_df.index.max() - temp_df.index.min()) / len(temp_df)} entre registros")
        
        elif analysis_type == "📊 Análisis de Distribución":
            st.subheader("Análisis de Distribución")
            
            # Selección de columna para distribución
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            dist_col = st.selectbox("Selecciona la columna para analizar distribución:", numeric_cols, key="distribution_column_select")
            
            if st.button("Analizar Distribución"):
                # Crear gráficos de distribución
                fig = go.Figure()
                
                # Histograma
                fig.add_trace(go.Histogram(x=df[dist_col], name="Histograma", nbinsx=30))
                
                # Añadir curva de densidad
                kde = stats.gaussian_kde(df[dist_col].dropna())
                x_range = np.linspace(df[dist_col].min(), df[dist_col].max(), 100)
                fig.add_trace(go.Scatter(x=x_range, y=kde(x_range)*len(df[dist_col])*
                                       (df[dist_col].max()-df[dist_col].min())/30,
                                       name="Densidad", line=dict(color='red')))
                
                fig.update_layout(title=f"Distribución de {dist_col}", bargap=0.1)
                st.plotly_chart(fig)
                
                # Estadísticas de distribución
                st.write("**Estadísticas de Distribución:**")
                st.write(f"- Asimetría: {df[dist_col].skew():.2f}")
                st.write(f"- Curtosis: {df[dist_col].kurtosis():.2f}")
                st.write(f"- Test de Normalidad (p-valor): {stats.normaltest(df[dist_col].dropna())[1]:.4f}")
        
        elif analysis_type == "🔗 Análisis de Correlaciones":
            st.subheader("Análisis de Correlaciones")
            
            # Seleccionar columnas numéricas para correlación
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas para el análisis de correlaciones.")
            else:
                # Matriz de correlación
                corr_matrix = df[numeric_cols].corr()
                
                # Crear heatmap
                fig = px.imshow(corr_matrix,
                               labels=dict(color="Correlación"),
                               x=corr_matrix.columns,
                               y=corr_matrix.columns,
                               color_continuous_scale="RdBu")
                
                fig.update_layout(title="Matriz de Correlación")
                st.plotly_chart(fig)
                
                # Análisis detallado de correlaciones
                st.write("**Correlaciones más Fuertes:**")
                # Obtener correlaciones más fuertes (excluyendo la diagonal)
                correlations = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        correlations.append({
                            'Variables': f"{numeric_cols[i]} vs {numeric_cols[j]}",
                            'Correlación': corr_matrix.iloc[i,j]
                        })
                
                correlations_df = pd.DataFrame(correlations)
                correlations_df = correlations_df.sort_values('Correlación', key=abs, ascending=False)
                st.dataframe(correlations_df)

        elif analysis_type == "🎯 Segmentación de Datos":
            st.subheader("Segmentación de Datos (Clustering)")
            
            # Seleccionar columnas para segmentación
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas para la segmentación.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Primera variable:", numeric_cols, key="segmentation_var1_select")
                with col2:
                    var2 = st.selectbox("Segunda variable:", numeric_cols, key="segmentation_var2_select")
                
                n_clusters = st.slider("Número de segmentos:", 2, 10, 3)
                
                if st.button("Realizar Segmentación"):
                    # Preparar datos para clustering
                    X = df[[var1, var2]].dropna()
                    
                    # Normalizar datos
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Realizar clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Visualizar resultados
                    fig = px.scatter(x=X[var1], y=X[var2], color=clusters.astype(str),
                                    title=f"Segmentación de Datos - {var1} vs {var2}",
                                    labels={"x": var1, "y": var2, "color": "Segmento"})
                    st.plotly_chart(fig)
                    
                    # Análisis de clusters
                    st.write("**Análisis de Segmentos:**")
                    for i in range(n_clusters):
                        segment_data = X[clusters == i]
                        st.write(f"Segmento {i+1}:")
                        st.write(f"- Tamaño: {len(segment_data)} ({len(segment_data)/len(X)*100:.1f}%)")
                        st.write(f"- Centro: {var1}={segment_data[var1].mean():.2f}, {var2}={segment_data[var2].mean():.2f}")
                    
                    # Clustering Jerárquico
                    st.subheader("Análisis de Clustering Jerárquico")
                    if st.button("Realizar Clustering Jerárquico"):
                        # Crear dendrograma
                        Z = linkage(X_scaled, method='ward')
                        fig_dendrogram = go.Figure()
                        
                        # Convertir el dendrograma a formato plotly
                        dn = dendrogram(Z, no_plot=True)
                        
                        # Crear trazas para las líneas del dendrograma
                        for i, d in zip(dn['icoord'], dn['dcoord']):
                            fig_dendrogram.add_trace(go.Scatter(x=i, y=d, mode='lines', line=dict(color='blue')))
                        
                        fig_dendrogram.update_layout(title='Dendrograma del Clustering Jerárquico',
                                                    showlegend=False)
                        st.plotly_chart(fig_dendrogram)
                    
                    # Análisis de Componentes Principales (PCA)
                    st.subheader("Análisis de Componentes Principales (PCA)")
                    if st.button("Realizar PCA"):
                        # Aplicar PCA
                        pca = PCA()
                        pca_result = pca.fit_transform(X_scaled)
                        
                        # Visualizar varianza explicada
                        explained_variance = pca.explained_variance_ratio_ * 100
                        fig_pca = go.Figure(data=[
                            go.Bar(x=[f'PC{i+1}' for i in range(len(explained_variance))],
                                  y=explained_variance)
                        ])
                        fig_pca.update_layout(title='Varianza Explicada por Componente Principal',
                                             xaxis_title='Componente Principal',
                                             yaxis_title='Varianza Explicada (%)')
                        st.plotly_chart(fig_pca)
                        
                        # Mostrar contribuciones de variables
                        loadings = pd.DataFrame(
                            pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                            index=[var1, var2]
                        )
                        st.write("**Contribución de Variables a los Componentes Principales:**")
                        st.dataframe(loadings)
                    
                    # Análisis de Series Temporales con Descomposición
                    st.subheader("Descomposición de Series Temporales")
                    if len(df.select_dtypes(include=['datetime64']).columns) > 0:
                        date_col = st.selectbox("Selecciona columna temporal:", 
                                               df.select_dtypes(include=['datetime64']).columns,
                                               key="decomp_date_select")
                        value_col = st.selectbox("Selecciona columna de valores para descomposición:", 
                                                df.select_dtypes(include=[np.number]).columns,
                                                key="decomp_value_select")
                        
                        if st.button("Realizar Descomposición Temporal"):
                            # Preparar datos
                            ts_data = df.set_index(date_col)[value_col]
                            
                            # Realizar descomposición
                            decomposition = seasonal_decompose(ts_data, period=30)
                            
                            # Visualizar componentes
                            fig_decomp = go.Figure()
                            fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=ts_data,
                                                           name='Original'))
                            fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend,
                                                           name='Tendencia'))
                            fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal,
                                                           name='Estacional'))
                            fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid,
                                                           name='Residual'))
                            
                            fig_decomp.update_layout(title='Descomposición de Series Temporales',
                                                    xaxis_title='Fecha',
                                                    yaxis_title='Valor')
                            st.plotly_chart(fig_decomp)
                    
                    # Aplicar K-means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Visualizar resultados
                    fig = px.scatter(x=X[var1], y=X[var2], color=clusters,
                                    labels={'x': var1, 'y': var2, 'color': 'Segmento'},
                                    title=f'Segmentación de Datos: {var1} vs {var2}')
                    st.plotly_chart(fig)
                    
                    # Análisis de segmentos
                    st.write("**Análisis de Segmentos:**")
                    for i in range(n_clusters):
                        segment_data = X[clusters == i]
                        st.write(f"**Segmento {i+1}:**")
                        st.write(f"- Tamaño: {len(segment_data)} ({len(segment_data)/len(X)*100:.1f}%)")
                        st.write(f"- Centro: {var1}={segment_data[var1].mean():.2f}, {var2}={segment_data[var2].mean():.2f}")

        elif analysis_type == "🔮 Análisis Predictivo":
            st.subheader("Análisis Predictivo Simple")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas para el análisis predictivo.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    target = st.selectbox("Variable objetivo:", numeric_cols, key="predictive_target_select")
                with col2:
                    feature = st.selectbox("Variable predictora:", [col for col in numeric_cols if col != target], key="predictive_feature_select")
                
                if st.button("Realizar Predicción"):
                    # Preparar datos
                    X = df[feature].values.reshape(-1, 1)
                    y = df[target].values
                    mask = ~np.isnan(X.flatten()) & ~np.isnan(y)
                    X = X[mask]
                    y = y[mask]
                    
                    # Dividir datos
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Entrenar modelo
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Realizar predicciones
                    y_pred = model.predict(X_test)
                    
                    # Visualizar resultados
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers',
                                            name='Datos reales', marker=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred, mode='lines',
                                            name='Predicción', line=dict(color='red')))
                    fig.update_layout(title=f'Predicción de {target} usando {feature}',
                                    xaxis_title=feature, yaxis_title=target)
                    st.plotly_chart(fig)
                    
                    # Métricas de rendimiento
                    st.write("**Métricas de Rendimiento:**")
                    st.write(f"- R² Score: {r2_score(y_test, y_pred):.3f}")
                    st.write(f"- Error Cuadrático Medio: {mean_squared_error(y_test, y_pred):.3f}")
                    st.write(f"- Error Absoluto Medio: {mean_absolute_error(y_test, y_pred):.3f}")

        elif analysis_type == "🔄 Análisis de Secuencias":
            st.subheader("Análisis de Patrones Secuenciales")
            
            # Seleccionar columna para análisis de secuencias
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) == 0:
                st.warning("No se encontraron columnas categóricas para el análisis de secuencias.")
            else:
                sequence_col = st.selectbox("Selecciona la columna para analizar secuencias:", categorical_cols, key="sequence_column_select")
                window_size = st.slider("Tamaño de la ventana de secuencia:", 2, 5, 2)
                
                if st.button("Analizar Secuencias"):
                    # Crear secuencias
                    sequences = []
                    for i in range(len(df) - window_size + 1):
                        seq = df[sequence_col].iloc[i:i+window_size].tolist()
                        sequences.append(tuple(seq))
                    
                    # Contar frecuencias de secuencias
                    seq_counts = Counter(sequences)
                    
                    # Mostrar patrones más comunes
                    st.write("**Patrones Secuenciales más Frecuentes:**")
                    seq_df = pd.DataFrame({
                        'Secuencia': [' → '.join(seq) for seq in seq_counts.keys()],
                        'Frecuencia': seq_counts.values(),
                        'Porcentaje': [count/len(sequences)*100 for count in seq_counts.values()]
                    })
                    seq_df = seq_df.sort_values('Frecuencia', ascending=False).head(10)
                    st.dataframe(seq_df)
                    
                    # Visualizar top 5 secuencias
                    fig = px.bar(seq_df.head(), x='Secuencia', y='Frecuencia',
                                title='Top 5 Patrones Secuenciales más Frecuentes')
                    st.plotly_chart(fig)
                    
                    # Análisis de transiciones
                    st.write("**Matriz de Transición:**")
                    transitions = defaultdict(Counter)
                    for seq in sequences:
                        for i in range(len(seq)-1):
                            transitions[seq[i]][seq[i+1]] += 1
                    
                    # Convertir a DataFrame
                    trans_matrix = pd.DataFrame(transitions).fillna(0)
                    fig = px.imshow(trans_matrix, title='Matriz de Transición',
                                   labels=dict(x='Estado Siguiente', y='Estado Actual', color='Frecuencia'))
                    st.plotly_chart(fig)
    else:
        st.info("📁 Primero carga un archivo CSV en la sección 'Descargar CSV desde URL'")

elif option == "📊 Análisis Avanzado":
    st.header("Análisis Avanzado de Datos")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Pestañas para análisis avanzado
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Análisis por Columna", "📈 Visualizaciones Avanzadas", "🔍 Filtros y Segmentación", "📊 Correlaciones y Estadísticas"])
        
        with tab1:
            st.subheader("Análisis Detallado por Columna")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_column = st.selectbox("Selecciona una columna para análisis:", df.columns.tolist(), key="detailed_analysis_column_select")
                
                # Mostrar información básica
                st.write("**Información Básica**")
                st.write(f"- **Tipo**: {df[selected_column].dtype}")
                st.write(f"- **Valores únicos**: {df[selected_column].nunique():,}")
                st.write(f"- **Valores nulos**: {df[selected_column].isnull().sum():,}")
                st.write(f"- **% Nulos**: {df[selected_column].isnull().sum()/len(df)*100:.2f}%")
                
                # Análisis estadístico
                if st.button("📊 Análisis Estadístico Completo"):
                    stats_result = advanced_statistical_analysis(df, selected_column)
                    
                    st.write("**Estadísticas Avanzadas**")
                    for key, value in stats_result.items():
                        if isinstance(value, float):
                            st.write(f"- **{key}**: {value:.4f}")
                        else:
                            st.write(f"- **{key}**: {value}")
            
            with col2:
                if selected_column:
                    # Distribución de valores
                    st.write("**Distribución de Valores**")
                    
                    if df[selected_column].dtype in ['int64', 'float64']:
                        # Para columnas numéricas
                        fig_hist = px.histogram(df, x=selected_column, nbins=30, 
                                              title=f"Distribución de {selected_column}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Box plot
                        fig_box = px.box(df, y=selected_column, 
                                        title=f"Box Plot de {selected_column}")
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                    else:
                        # Para columnas categóricas
                        value_counts = df[selected_column].value_counts().head(20)
                        fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                        title=f"Top 20 valores más frecuentes en {selected_column}")
                        fig_bar.update_xaxes(title=selected_column)
                        fig_bar.update_yaxes(title="Frecuencia")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Tabla de valores únicos
                    st.write("**Valores Únicos (Top 10)**")
                    if df[selected_column].dtype in ['int64', 'float64']:
                        unique_stats = df[selected_column].describe()
                        st.dataframe(unique_stats.to_frame().T, use_container_width=True)
                    else:
                        value_counts_df = df[selected_column].value_counts().head(10).reset_index()
                        value_counts_df.columns = [selected_column, 'Frecuencia']
                        value_counts_df['Porcentaje'] = (value_counts_df['Frecuencia'] / len(df) * 100).round(2)
                        st.dataframe(value_counts_df, use_container_width=True)
        
        with tab2:
            st.subheader("Visualizaciones Avanzadas con Filtros")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                plot_type = st.selectbox(
                    "Tipo de gráfico:",
                    ["Gráfico de Barras", "Gráfico de Líneas", "Gráfico Circular", 
                     "Scatter Plot", "Heatmap", "Box Plot", "Violin Plot", 
                     "Histograma", "Gráfico de Área"],
                    key="advanced_plot_type_select"
                )
            
            with col2:
                x_column = st.selectbox("Columna X:", df.columns.tolist(), key="advanced_x_column_select")
            
            with col3:
                y_columns = df.columns.tolist()
                if plot_type in ["Gráfico de Líneas", "Scatter Plot", "Box Plot", "Violin Plot", "Gráfico de Área"]:
                    y_column = st.selectbox("Columna Y:", [None] + y_columns, key="advanced_y_column_select")
                elif plot_type == "Heatmap":
                    y_column = None
                    st.write("*Automático*")
                else:
                    y_column = st.selectbox("Columna Y (opcional):", [None] + y_columns, key="advanced_y_column_optional_select")
            
            with col4:
                color_column = st.selectbox(
                    "Color por:",
                    [None] + df.columns.tolist(),
                    key="advanced_color_column_select"
                )
            
            # Filtros de valores
            st.write("**🎯 Filtros de Valores**")
            filter_values = {}
            
            # Filtro para columna X
            if x_column:
                unique_x_values = df[x_column].unique()
                if len(unique_x_values) <= 50:  # Solo mostrar filtro si hay pocos valores únicos
                    selected_x_values = st.multiselect(
                        f"Filtrar valores de {x_column} (opcional):",
                        unique_x_values,
                        default=unique_x_values.tolist()
                    )
                    if len(selected_x_values) < len(unique_x_values):
                        filter_values[x_column] = selected_x_values
                elif df[x_column].dtype in ['int64', 'float64']:
                    # Para columnas numéricas, usar slider
                    min_val, max_val = float(df[x_column].min()), float(df[x_column].max())
                    if min_val != max_val:
                        range_values = st.slider(
                            f"Rango de {x_column}:",
                            min_val, max_val, (min_val, max_val)
                        )
                        if range_values != (min_val, max_val):
                            filtered_df = df[(df[x_column] >= range_values[0]) & (df[x_column] <= range_values[1])]
                            filter_values = {'range_filter': filtered_df}
            
            # Crear gráfico
            if st.button("🎯 Generar Visualización Avanzada", type="primary"):
                if 'range_filter' in filter_values:
                    fig = create_advanced_plot(filter_values['range_filter'], plot_type, x_column, y_column, color_column)
                else:
                    fig = create_advanced_plot(df, plot_type, x_column, y_column, color_column, filter_values)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Información adicional del gráfico
                    st.write("**📈 Información del Gráfico**")
                    if 'range_filter' in filter_values:
                        data_used = filter_values['range_filter']
                    elif filter_values and x_column in filter_values:
                        data_used = df[df[x_column].isin(filter_values[x_column])]
                    else:
                        data_used = df
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Registros mostrados", len(data_used))
                    with col2:
                        st.metric("Total de registros", len(df))
                    with col3:
                        st.metric("% de datos mostrados", f"{len(data_used)/len(df)*100:.1f}%")
        
        with tab3:
            st.subheader("Filtros y Segmentación de Datos")
            
            st.write("**🔧 Constructor de Filtros**")
            
            # Filtros múltiples
            filter_columns = st.multiselect("Selecciona columnas para filtrar:", df.columns.tolist())
            
            filtered_df = df.copy()
            active_filters = {}
            
            if filter_columns:
                for col in filter_columns:
                    st.write(f"**Filtro para {col}:**")
                    
                    if df[col].dtype in ['int64', 'float64']:
                        # Filtro numérico
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        if min_val != max_val:
                            range_filter = st.slider(
                                f"Rango de {col}:",
                                min_val, max_val, (min_val, max_val),
                                key=f"range_{col}"
                            )
                            if range_filter != (min_val, max_val):
                                filtered_df = filtered_df[
                                    (filtered_df[col] >= range_filter[0]) & 
                                    (filtered_df[col] <= range_filter[1])
                                ]
                                active_filters[col] = f"{range_filter[0]} - {range_filter[1]}"
                    else:
                        # Filtro categórico
                        unique_values = df[col].unique()
                        if len(unique_values) <= 100:  # Límite para evitar sobrecarga
                            selected_values = st.multiselect(
                                f"Valores de {col}:",
                                unique_values,
                                default=unique_values.tolist(),
                                key=f"multi_{col}"
                            )
                            if len(selected_values) < len(unique_values):
                                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                                active_filters[col] = f"{len(selected_values)} valores seleccionados"
                        else:
                            st.info(f"Demasiados valores únicos en {col} ({len(unique_values)}). Usa búsqueda de texto.")
                            search_term = st.text_input(f"Buscar en {col}:", key=f"search_{col}")
                            if search_term:
                                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)]
                                active_filters[col] = f"Contiene: '{search_term}'"
                
                # Mostrar filtros activos
                if active_filters:
                    st.write("**🎯 Filtros Activos:**")
                    for col, filter_desc in active_filters.items():
                        st.write(f"- **{col}**: {filter_desc}")
                
                # Mostrar estadísticas de filtrado
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Registros originales", len(df))
                with col2:
                    st.metric("Registros filtrados", len(filtered_df))
                with col3:
                    reduction = (1 - len(filtered_df)/len(df)) * 100 if len(df) > 0 else 0
                    st.metric("Reducción", f"{reduction:.1f}%")
                
                # Mostrar datos filtrados
                st.divider()
                st.write("**📊 Datos Filtrados**")
                st.dataframe(filtered_df, use_container_width=True)
                
                # Opción para guardar datos filtrados
                if len(filtered_df) > 0:
                    csv_filtered = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Datos Filtrados",
                        data=csv_filtered,
                        file_name="datos_filtrados.csv",
                        mime="text/csv"
                    )
        
        with tab4:
            st.subheader("Análisis de Correlaciones y Estadísticas")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Matriz de correlación
                st.write("**🔗 Matriz de Correlación**")
                
                corr_method = st.selectbox(
                    "Método de correlación:",
                    ["pearson", "spearman", "kendall"],
                    key="correlation_method_select"
                )
                
                correlation_matrix = df[numeric_cols].corr(method=corr_method)
                
                # Heatmap de correlación
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    title=f"Matriz de Correlación ({corr_method.title()})",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlaciones más fuertes
                st.write("**💪 Correlaciones Más Fuertes**")
                
                # Obtener correlaciones sin la diagonal
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        corr_value = correlation_matrix.iloc[i, j]
                        corr_pairs.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Correlación': corr_value,
                            'Correlación Abs': abs(corr_value)
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlación Abs', ascending=False)
                st.dataframe(corr_df.head(10), use_container_width=True)
                
                # Análisis de regresión simple
                st.divider()
                st.write("**📈 Análisis de Regresión**")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_reg = st.selectbox("Variable independiente (X):", numeric_cols, key="regression_x_select")
                with col2:
                    y_reg = st.selectbox("Variable dependiente (Y):", numeric_cols, key="regression_y_select")
                
                if x_reg != y_reg and st.button("🧮 Calcular Regresión"):
                    # Eliminar valores nulos
                    reg_data = df[[x_reg, y_reg]].dropna()
                    
                    import streamlit as st

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    📊 Sistema de Análisis de Datos - Creado con Streamlit
</div>
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    © 2025 - @nicolee.palomino
</div>
""", unsafe_allow_html=True)