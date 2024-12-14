import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def draw_star_with_halo(ax, center, size, halo_size, num_points=5, color="white", halo_color="white", alpha=1.0):
    """Dibuja una estrella con un patrón de picos y un halo difuso."""
    # Dibujar el halo
    halo_alpha = alpha * 0.3  # Halo más transparente
    ax.scatter(center[0], center[1], s=halo_size, color=halo_color, alpha=halo_alpha, zorder=1)
    
    # Dibujar la estrella central
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    radii = np.empty_like(angles)
    radii[::2] = size  # Picos largos
    radii[1::2] = size * 0.5  # Picos cortos
    
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    ax.fill(x, y, color=color, alpha=alpha, zorder=2)

def draw_star_field_with_halos(ax, num_stars, bounds):
    """Dibuja un campo de estrellas con halos difusos."""
    x_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    y_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    star_sizes = np.random.uniform(0.001, 0.05, num_stars)
    halo_sizes = star_sizes * 100  # Escalar el halo según el tamaño de la estrella
    star_colors = np.random.choice(
        ["white", "lightblue", "yellow", "red"], 
        num_stars, 
        p=[0.5, 0.25, 0.15, 0.1]  # Probabilidades ajustadas
    )
    
    for x, y, size, halo_size, color in zip(x_stars, y_stars, star_sizes, halo_sizes, star_colors):
        num_points = np.random.choice([5, 6, 7])  # Número de picos aleatorio
        halo_color = color  # Usar el mismo color para el halo
        draw_star_with_halo(ax, center=(x, y), size=size, halo_size=halo_size, num_points=num_points, color=color, halo_color=halo_color, alpha=0.8)

# Configuración de Streamlit
st.title("Simulador de Campo de Estrellas con Halos")

# Parámetros interactivos
num_stars = st.sidebar.slider("Número de estrellas", min_value=100, max_value=2000, value=500, step=100)

# Configuración del gráfico
fig, ax = plt.subplots(figsize=(50, 50), facecolor="black")  # Fondo negro global
ax.set_aspect("equal")
ax.set_facecolor("black")  # Fondo negro para los ejes
ax.axis("off")  # Ocultar los ejes

# Dibujar el campo de estrellas con halos
bounds = (-2.5, 2.5)  # Límites del área donde se generarán las estrellas
draw_star_field_with_halos(ax, num_stars=num_stars, bounds=bounds)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
