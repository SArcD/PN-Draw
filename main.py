import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def draw_star_field(ax, num_stars, bounds):
    """Dibuja un campo de estrellas en el fondo negro."""
    # Generar posiciones aleatorias para las estrellas
    x_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    y_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    
    # Generar tamaños aleatorios para las estrellas
    star_sizes = np.random.uniform(0.5, 3, num_stars)
    
    # Generar colores aleatorios para las estrellas
    star_colors = np.random.choice(["white", "lightblue", "yellow"], num_stars, p=[0.6, 0.3, 0.1])
    
    # Dibujar cada estrella
    for x, y, size, color in zip(x_stars, y_stars, star_sizes, star_colors):
        ax.scatter(x, y, s=size, color=color, alpha=0.8)

# Configuración de la aplicación de Streamlit
st.title("Simulador de Campo de Estrellas")

# Parámetros para el campo de estrellas
bounds = (-2.5, 2.5)  # Límites del área donde se generarán las estrellas

# Crear un deslizador en la barra lateral para ajustar el número de estrellas
num_stars = st.sidebar.slider(
    "Número de estrellas", min_value=100, max_value=2000, value=500, step=100
)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_facecolor("black")  # Fondo negro
ax.axis("off")  # Ocultar los ejes

# Dibujar el campo de estrellas
draw_star_field(ax, num_stars=num_stars, bounds=bounds)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
