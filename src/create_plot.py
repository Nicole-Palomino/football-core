import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Función para crear gráficos
def create_plot(df, plot_type, x_col, y_col=None, color_col=None, marker_size=10, orientation='v', show_values=False):
    try:
        # Configuración del tema oscuro para los gráficos
        dark_template = go.layout.Template()
        dark_template.layout.plot_bgcolor = '#1e1e1e'
        dark_template.layout.paper_bgcolor = '#1e1e1e'
        dark_template.layout.font.color = '#ffffff'
        dark_template.layout.xaxis.gridcolor = '#333333'
        dark_template.layout.yaxis.gridcolor = '#333333'
        dark_template.layout.xaxis.linecolor = '#444444'
        dark_template.layout.yaxis.linecolor = '#444444'
        
        # Paleta de colores para tema oscuro
        default_colors = ['#00ff00', '#00ccff', '#ff3366', '#ffcc00', '#9933ff', '#ff9933', '#33ff33', '#ff99cc']
        if plot_type == "Gráfico de Barras":
            st.info("🔹 El gráfico de barras muestra la comparación de valores entre diferentes categorías. "
            "Si no se indica la columna Y, se genera un gráfico de frecuencias de la columna categórica.")
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                            orientation=orientation,
                            color_discrete_sequence=default_colors,
                            title=f"Gráfico de Barras: {x_col} vs {y_col}")
            else:
                value_counts = df[x_col].value_counts()
                if orientation == 'h':
                    fig = px.bar(x=value_counts.values, y=value_counts.index, orientation='h',
                                color_discrete_sequence=default_colors,
                                title=f"Distribución de {x_col}")
                else:
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                color_discrete_sequence=default_colors,
                                title=f"Distribución de {x_col}")
                fig.update_xaxis(title=x_col)
                fig.update_yaxis(title="Frecuencia")
            
            if show_values:
                fig.update_traces(texttemplate='%{value}', textposition='outside')
        
        elif plot_type == "Gráfico Circular":
            st.info("🔸 El gráfico circular (pie chart) muestra la proporción de cada categoría en relación al total. "
            "Utiliza los valores de la Columna Categórica (X) para agrupar y contar frecuencias.")
            value_counts = df[x_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        color_discrete_sequence=default_colors,
                        title=f"Distribución de {x_col}")
            if show_values:
                fig.update_traces(textinfo='percent+label+value')
        
        elif plot_type == "Heatmap":
            st.info("🟠 El heatmap muestra una matriz de correlación entre columnas numéricas. "
            "Se requiere al menos 2 columnas numéricas en el DataFrame para generar este gráfico.")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                correlation_matrix = numeric_df.corr()
                color_palette = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
                fig = px.imshow(correlation_matrix,
                        color_continuous_scale=color_palette,
                        aspect="auto",
                        title="Matriz de Correlación (Heatmap)")
                if show_values:
                    for i in range(len(correlation_matrix)):
                        for j in range(len(correlation_matrix.columns)):
                            fig.add_annotation(
                                x=correlation_matrix.columns[j],
                                y=correlation_matrix.index[i],
                                text=str(round(correlation_matrix.iloc[i, j], 2)),
                                showarrow=False,
                                font=dict(color='white')
                            )
            else:
                st.error("Se necesitan al menos 2 columnas numéricas para crear un heatmap")
                return None
        
        elif plot_type == "Violin Plot":
            st.info("🟣 El gráfico violin combina un box plot y una distribución de datos. "
            "Columna Categórica (X) define las categorías, Columna Numérica (Y) representa los valores numéricos.")
            fig = px.violin(df, x=x_col, y=y_col, color=color_col,
                          color_discrete_sequence=default_colors,
                          box=True, points="all",
                          title=f"Violin Plot: {x_col} vs {y_col}")
        
        elif plot_type == "Box Plot":
            st.info("🟢 El box plot (diagrama de caja) muestra la distribución de los datos numéricos de Columna Numérica (Y) "
            "agrupados por categorías de Columna Categórica (X). Incluye mediana, cuartiles y posibles valores atípicos.")
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                        color_discrete_sequence=default_colors,
                        title=f"Box Plot: {x_col} vs {y_col}",
                        points="all",
                        notched=True)
            fig.update_traces(marker=dict(size=4, opacity=0.7),
                            line=dict(width=2),
                            fillcolor='rgba(255,255,255,0.1)',
                            boxmean=True)
        
        elif plot_type == "Área":
            st.info("🔵 El gráfico de área es útil para visualizar tendencias acumulativas en el tiempo o por categorías. "
            "Se requiere Columna Categórica (X) como eje base y Columna Numérica (Y) como valores numéricos.")
            fig = px.area(df, x=x_col, y=y_col, color=color_col,
                         color_discrete_sequence=default_colors,
                         title=f"Gráfico de Área: {x_col} vs {y_col}")
            if show_values:
                fig.update_traces(texttemplate='%{y}', textposition='top')
        
        elif plot_type == "Burbujas":
            st.info("🟡 El gráfico de burbujas es una variante del scatter plot. "
            "Columna Categórica (X) y Columna Numérica (Y) definen la posición, y se define el color de las burbujas.")
            size_col = color_col if color_col else [marker_size]*len(df)
            color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            fig = px.scatter(df, x=x_col, y=y_col,
                            size=size_col,
                            color=color_col,
                            color_discrete_sequence=color_palette,
                            size_max=marker_size,
                            title=f"Gráfico de Burbujas: {x_col} vs {y_col}")
            fig.update_traces(marker=dict(line=dict(width=1, color='white'),
                                        opacity=0.7),
                            selector=dict(mode='markers'))
            if show_values:
                fig.update_traces(texttemplate='(%{x}, %{y})', textposition='top center')
        
        # Actualizar diseño general con tema oscuro
        fig.update_layout(
            showlegend=True if color_col else False,
            template=dark_template,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='#333333',
                linecolor='#444444',
                zerolinecolor='#444444',
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff')
            ),
            yaxis=dict(
                gridcolor='#333333',
                linecolor='#444444',
                zerolinecolor='#444444',
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff')
            ),
            margin=dict(t=50, l=50, r=50, b=50),
            hoverlabel=dict(
                bgcolor='#2b2b2b',
                font_color='#ffffff'
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error al crear el gráfico: {str(e)}")
        return None