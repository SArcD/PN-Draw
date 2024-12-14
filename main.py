import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def draw_star_with_halo(ax, center, size, halo_size, num_points=5, color="white", halo_color="white", alpha=1.0):
    """Dibuja una estrella con un patrón de picos y un halo difuso."""
    # Dibujar el halo
    halo_alpha = alpha * 0.5  # Halo más transparente
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
    star_sizes = np.random.uniform(0.001, 0.005, num_stars)
    halo_sizes = star_sizes * 100  # Escalar el halo según el tamaño de la estrella
    star_colors = np.random.choice(
        ["white", "lightblue", "yellow", "red"], 
        num_stars, 
        p=[0.7, 0.15, 0.05, 0.1]  # Probabilidades ajustadas
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


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Función para dibujar una estrella con núcleo brillante y bordes difusos
def draw_central_star(ax, position, size, color):
    # Dibujar el núcleo brillante
    star_core = plt.Circle(position, size, color=color, alpha=1.0, zorder=5)
    ax.add_artist(star_core)
    
    # Dibujar el halo difuso
    halo_sizes = np.linspace(size * 1.2, size * 3, 5)
    alpha_values = np.linspace(0.4, 0.1, len(halo_sizes))
    for halo_size, alpha in zip(halo_sizes, alpha_values):
        halo = plt.Circle(position, halo_size, color=color, alpha=alpha, zorder=4)
        ax.add_artist(halo)
    
    # Dibujar los rayos de brillo en forma de "X"
    for angle in [0, 45]:
        x_vals = [position[0] - halo_sizes[-1] * np.cos(np.radians(angle)),
                  position[0] + halo_sizes[-1] * np.cos(np.radians(angle))]
        y_vals = [position[1] - halo_sizes[-1] * np.sin(np.radians(angle)),
                  position[1] + halo_sizes[-1] * np.sin(np.radians(angle))]
        ax.plot(x_vals, y_vals, color=color, alpha=0.2, lw=2, zorder=3)

# Configuración inicial del gráfico
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_facecolor("black")
ax.axis("off")

# Dibujar el campo de estrellas una vez
# (Aquí deberías incluir tu función para dibujar el campo de estrellas)
# draw_star_field(ax, num_stars=500, bounds=(-2.5, 2.5))

# Controles en la barra lateral
st.sidebar.header("Controles de la Estrella Central")
x_pos = st.sidebar.slider("Posición X", -2.5, 2.5, 0.0, step=0.1)
y_pos = st.sidebar.slider("Posición Y", -2.5, 2.5, 0.0, step=0.1)
size = st.sidebar.slider("Tamaño", 0.1, 1.0, 0.2, step=0.1)
color = st.sidebar.selectbox("Color", ["white", "yellow", "red", "blue"])

# Dibujar la estrella central según los controles del usuario
draw_central_star(ax, (x_pos, y_pos), size, color)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

