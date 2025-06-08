import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Función para análisis estadístico avanzado
def advanced_statistical_analysis(df, column):
    """Realiza análisis estadístico avanzado de una columna"""
    if df[column].dtype in ['int64', 'float64']:
        # Estadísticas básicas
        stats_dict = {
            'Media': df[column].mean(),
            'Mediana': df[column].median(),
            'Moda': df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A',
            'Desviación Estándar': df[column].std(),
            'Varianza': df[column].var(),
            'Asimetría (Skewness)': df[column].skew(),
            'Curtosis': df[column].kurtosis(),
            'Rango': df[column].max() - df[column].min(),
            'Q1 (25%)': df[column].quantile(0.25),
            'Q3 (75%)': df[column].quantile(0.75),
            'IQR': df[column].quantile(0.75) - df[column].quantile(0.25)
        }
        
        return stats_dict
    else:
        # Para columnas categóricas
        stats_dict = {
            'Valores únicos': df[column].nunique(),
            'Valor más frecuente': df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A',
            'Frecuencia del más común': df[column].value_counts().iloc[0] if not df[column].empty else 0,
            'Valores nulos': df[column].isnull().sum(),
            'Porcentaje nulos': f"{df[column].isnull().sum() / len(df) * 100:.2f}%"
        }
        return stats_dict

# Función para crear gráficos avanzados
def create_advanced_plot(df, plot_type, x_col, y_col=None, color_col=None, filter_values=None):
    try:
        # Aplicar filtros si se proporcionan
        filtered_df = df.copy()
        if filter_values and x_col in filter_values:
            filtered_df = filtered_df[filtered_df[x_col].isin(filter_values[x_col])]
        
        if plot_type == "Gráfico de Barras":
            if y_col:
                fig = px.bar(filtered_df, x=x_col, y=y_col, color=color_col, 
                           title=f"Gráfico de Barras: {x_col} vs {y_col}")
            else:
                value_counts = filtered_df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Distribución de {x_col}")
                fig.update_xaxis(title=x_col)
                fig.update_yaxis(title="Frecuencia")
        
        elif plot_type == "Gráfico de Líneas":
            fig = px.line(filtered_df, x=x_col, y=y_col, color=color_col, 
                         title=f"Gráfico de Líneas: {x_col} vs {y_col}")
        
        elif plot_type == "Gráfico Circular":
            if filter_values and x_col in filter_values:
                value_counts = filtered_df[x_col].value_counts()
            else:
                value_counts = df[x_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, 
                        title=f"Distribución de {x_col}")
        
        elif plot_type == "Scatter Plot":
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col, 
                           title=f"Scatter Plot: {x_col} vs {y_col}")
        
        elif plot_type == "Heatmap":
            numeric_df = filtered_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                correlation_matrix = numeric_df.corr()
                fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", 
                               title="Matriz de Correlación (Heatmap)")
            else:
                st.error("Se necesitan al menos 2 columnas numéricas para crear un heatmap")
                return None
        
        elif plot_type == "Box Plot":
            fig = px.box(filtered_df, x=x_col, y=y_col, color=color_col,
                        title=f"Box Plot: {x_col} vs {y_col}")
        
        elif plot_type == "Violin Plot":
            fig = px.violin(filtered_df, x=x_col, y=y_col, color=color_col,
                           title=f"Violin Plot: {x_col} vs {y_col}")
        
        elif plot_type == "Histograma":
            fig = px.histogram(filtered_df, x=x_col, color=color_col, nbins=30,
                              title=f"Histograma de {x_col}")
        
        elif plot_type == "Gráfico de Área":
            fig = px.area(filtered_df, x=x_col, y=y_col, color=color_col,
                         title=f"Gráfico de Área: {x_col} vs {y_col}")
        
        return fig
    except Exception as e:
        st.error(f"Error al crear el gráfico: {str(e)}")
        return None