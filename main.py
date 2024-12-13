import streamlit as st
import plotly.graph_objects as go
import numpy as np

def generate_star_field(num_stars, bounds):
    """Genera un campo de estrellas con posiciones, tamaños y colores aleatorios."""
    x_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    y_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    star_sizes = np.random.uniform(2, 10, num_stars)
    star_colors = np.random.choice(["white", "lightblue", "yellow"], num_stars, p=[0.6, 0.3, 0.1])
    return x_stars, y_stars, star_sizes, star_colors

# Configuración de la aplicación de Streamlit
st.title("Simulador de Campo de Estrellas (Plotly)")

# Deslizador para el número de estrellas
num_stars = st.sidebar.slider(
    "Número de estrellas", min_value=100, max_value=2000, value=500, step=100
)

# Generar los datos del campo de estrellas
bounds = (-2.5, 2.5)
x_stars, y_stars, star_sizes, star_colors = generate_star_field(num_stars, bounds)

# Crear la figura de Plotly
fig = go.Figure()

# Añadir las estrellas
for x, y, size, color in zip(x_stars, y_stars, star_sizes, star_colors):
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.8),
        showlegend=False
    ))

# Configurar el fondo negro y los ejes
fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="black",
    paper_bgcolor="black",
)

# Ajustar los límites del gráfico
fig.update_xaxes(range=[bounds[0], bounds[1]])
fig.update_yaxes(range=[bounds[0], bounds[1]])

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)
