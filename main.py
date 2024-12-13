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
    for _ in range(num_stars):
        # Posición aleatoria de la estrella
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        
        # Tamaño aleatorio de la estrella
        size = np.random.uniform(0.05, 0.2)
        
        # Número de picos aleatorio
        num_points = np.random.choice([5, 6, 7])
        
        # Color aleatorio
        color = np.random.choice(["white", "lightblue", "yellow"], p=[0.6, 0.3, 0.1])
        
        # Dibujar la estrella
        draw_star(ax, center=(x, y), size=size, num_points=num_points, color=color, alpha=0.8)

# Configuración de la aplicación de Streamlit
st.title("Simulador de Campo de Estrellas")

# Deslizador para el número de estrellas
num_stars = st.sidebar.slider(
    "Número de estrellas", min_value=100, max_value=2000, value=500, step=100
)

# Configuración de Matplotlib
fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")  # Fondo negro global
ax.set_aspect("equal")
ax.set_facecolor("black")  # Fondo negro para los ejes
ax.axis("off")  # Ocultar los ejes

# Dibujar el campo de estrellas
bounds = (-2.5, 2.5)
draw_star_field(ax, num_stars=num_stars, bounds=bounds)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
