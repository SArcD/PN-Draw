import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def draw_star(ax, center, size, num_points=5, color="white", alpha=1.0):
    """Dibuja una estrella con un patrón de picos."""
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    radii = np.empty_like(angles)
    radii[::2] = size  # Picos largos
    radii[1::2] = size * 0.5  # Picos cortos
    
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    ax.fill(x, y, color=color, alpha=alpha)

def draw_star_field(ax, num_stars, bounds):
    """Dibuja un campo de estrellas en el fondo negro con forma de estrella."""
    x_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    y_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    star_sizes = np.random.uniform(0.005, 0.01, num_stars)
    star_colors = np.random.choice(
        ["white", "lightblue", "yellow", "red"], 
        num_stars, 
        p=[0.5, 0.25, 0.15, 0.1]  # Probabilidades ajustadas
    )
    
    for x, y, size, color in zip(x_stars, y_stars, star_sizes, star_colors):
        num_points = np.random.choice([5, 6, 7])  # Número de picos aleatorio
        draw_star(ax, center=(x, y), size=size, num_points=num_points, color=color, alpha=0.8)

# Configuración de Streamlit
st.title("Simulador de Campo de Estrellas")

# Parámetros interactivos
num_stars = st.sidebar.slider("Número de estrellas", min_value=100, max_value=2000, value=500, step=100)

# Configuración del gráfico
fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")  # Fondo negro global
ax.set_aspect("equal")
ax.set_facecolor("black")  # Fondo negro para los ejes
ax.axis("off")  # Ocultar los ejes

# Dibujar el campo de estrellas
bounds = (-2.5, 2.5)  # Límites del área donde se generarán las estrellas
draw_star_field(ax, num_stars=num_stars, bounds=bounds)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
