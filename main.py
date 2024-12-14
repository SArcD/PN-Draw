import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Función para dibujar el campo de estrellas
def draw_star_field(ax, num_stars, bounds):
    """Dibuja un campo de estrellas en el fondo negro."""
    x_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    y_stars = np.random.uniform(bounds[0], bounds[1], num_stars)
    star_sizes = np.random.uniform(0.5, 3, num_stars)
    star_colors = np.random.choice(
        ["white", "lightblue", "yellow", "red"], 
        num_stars, 
        p=[0.6, 0.2, 0.1, 0.1]
    )
    for x, y, size, color in zip(x_stars, y_stars, star_sizes, star_colors):
        ax.scatter(x, y, s=size, color=color, alpha=0.8)

# Configuración de Streamlit
st.title("Generador de Nebulosa")

# Parámetros para el campo de estrellas
num_stars = st.sidebar.slider("Número de estrellas", min_value=100, max_value=2000, value=500, step=100)
bounds = (-2.5, 2.5)

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_facecolor("black")
ax.axis("off")

# Dibujar el campo de estrellas
draw_star_field(ax, num_stars, bounds)

# Mostrar el campo de estrellas en Streamlit
st.pyplot(fig)

# Sección para la estrella central
st.sidebar.subheader("Configuración de la Estrella Central")
star_x = st.sidebar.slider("Posición X", min_value=float(bounds[0]), max_value=float(bounds[1]), value=0.0, step=0.1)
star_y = st.sidebar.slider("Posición Y", min_value=float(bounds[0]), max_value=float(bounds[1]), value=0.0, step=0.1)
star_size = st.sidebar.slider("Tamaño de la estrella", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
star_color = st.sidebar.color_picker("Color de la estrella", "#ffffff")

# Función para dibujar la estrella central con halo y destellos
def draw_central_star(ax, center, size, color):
    """Dibuja una estrella central con halo difuso y destellos."""
    # Dibujar el halo difuso
    halo_size = size * 1000
    ax.scatter(center[0], center[1], s=halo_size, color=color, alpha=0.2, zorder=1)
    
    # Dibujar la estrella central
    ax.scatter(center[0], center[1], s=size * 100, color=color, alpha=1.0, zorder=2)
    
    # Dibujar los destellos en forma de cruz
    for angle in [0, 45]:
        x_vals = [center[0] - size, center[0], center[0] + size]
        y_vals = [center[1], center[1], center[1]]
        coords = np.array([x_vals, y_vals])
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
        rotated_coords = rotation_matrix @ coords
        ax.plot(rotated_coords[0], rotated_coords[1], color=color, alpha=0.5, lw=2, zorder=1)

# Dibujar la estrella central
draw_central_star(ax, (star_x, star_y), star_size, star_color)

# Actualizar la visualización en Streamlit
st.pyplot(fig)
