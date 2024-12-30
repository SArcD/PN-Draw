import numpy as np
import streamlit as st
import plotly.graph_objects as go
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# Parámetros iniciales
nx, ny = 200, 200  # Aumentar la resolución de la malla
lx, ly = 1e4 * 1.496e+11, 1e4 * 1.496e+11  # Dimensiones físicas de la malla en metros (10,000 AU)
dx, dy = lx / nx, ly / ny  # Tamaño de celda
dt_default = 2.0  # Paso de tiempo por defecto
c = 0.1  # Velocidad de advección constante
R_gas = 8.314  # Constante de gas ideal en J/(mol·K)
M_mol = 0.02896  # Masa molar del gas (kg/mol, aire)
k_B = 1.38e-23  # Constante de Boltzmann (J/K)
m_H = 1.67e-27  # Masa del átomo de hidrógeno (kg)
G_default = 6.674e-11 * 100  # Constante gravitacional inicial
mu = 2.33  # Peso molecular medio para gas molecular
gamma = 5 / 3  # Índice adiabático para un gas monoatómico
M_solar = 1.989e30  # Masa solar en kg



# Reemplazar el uso de noise2d por noise2 en la generación de condiciones iniciales
def create_initial_conditions(nx, ny, lx, ly):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)

    # Generar densidad inicial basada en ruido Simplex
    noise = OpenSimplex(seed=42)
    rho0 = np.zeros((nx, ny))
    scale = 10.0  # Escala del ruido

    for i in range(nx):
        for j in range(ny):
            rho0[i, j] = noise.noise2(i / scale, j / scale)  # Usar noise2

    # Normalizar la densidad para alcanzar una masa total de ~10 masas solares
    total_volume = lx * ly * dx  # Volumen total de la nube
    target_mass = 10 * M_solar  # Masa objetivo: 10 masas solares
    mean_density = target_mass / total_volume

    rho_min, rho_max = 0.1 * mean_density, 10 * mean_density
    rho0 = rho_min + (rho0 - rho0.min()) / (rho0.max() - rho0.min()) * (rho_max - rho_min)

    # Generar un campo de temperatura inicial más representativo
    temp_min, temp_max = 10, 20  # Temperatura mínima y máxima en K
    temperature = temp_max - (temp_max - temp_min) * (rho0 - rho_min) / (rho_max - rho_min)

    return rho0, temperature


# Calcular el potencial gravitacional
def calculate_gravitational_potential(rho, dx, dy, G):
    from scipy.fft import fft2, ifft2, fftfreq

    kx = 2 * np.pi * fftfreq(rho.shape[0], dx)
    ky = 2 * np.pi * fftfreq(rho.shape[1], dy)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1  # Evitar división por cero

    rho_ft = fft2(rho)
    phi_ft = -4 * np.pi * G * rho_ft / k_squared
    phi_ft[0, 0] = 0  # Normalizar el modo cero

    phi = np.real(ifft2(phi_ft))
    return phi

# Actualizar el campo de densidad y temperatura
def update_density_and_temperature(rho, temperature, phi, dx, dy, dt, radiation_enabled):
    grad_phi_x, grad_phi_y = np.gradient(phi, dx, dy)
    v_x = -grad_phi_x
    v_y = -grad_phi_y

    rho_new = rho.copy()
    temperature_new = temperature.copy()
    for i in range(1, rho.shape[0] - 1):
        for j in range(1, rho.shape[1] - 1):
            rho_new[i, j] -= dt * (
                (v_x[i + 1, j] * rho[i + 1, j] - v_x[i - 1, j] * rho[i - 1, j]) / (2 * dx)
                + (v_y[i, j + 1] * rho[i, j + 1] - v_y[i, j - 1] * rho[i, j - 1]) / (2 * dy)
            )
            # Calentamiento adiabático: T ~ rho^(gamma - 1)
            if rho_new[i, j] > 1e-15:  # Establecer límite inferior para densidad
                temperature_new[i, j] = max(
                    temperature[i, j] * (rho_new[i, j] / rho[i, j])**(gamma - 1), 10  # Temperatura mínima de 10 K
                )

            # Efecto de radiación opcional
            if radiation_enabled and temperature_new[i, j] > 50:  # Umbral de radiación
                temperature_new[i, j] -= 1e-5 * (temperature_new[i, j]**4) * dt

    return np.maximum(rho_new, 1e-15), np.maximum(temperature_new, 10)  # Evitar valores negativos o muy bajos

# Inicializar los campos
rho, temperature = create_initial_conditions(nx, ny, lx, ly)
pressure = (rho * R_gas * temperature) / M_mol  # Calcular presión inicial

# Calcular la masa total inicial de la nube
total_mass = np.sum(rho) * dx * dy  # Masa total en kilogramos
st.sidebar.write(f"Masa total inicial de la nube: {total_mass:.2e} kg")
# Calcular la región de interés
x_idx, y_idx = np.unravel_index(np.argmax(rho), rho.shape)
region_size = 20  # Reducir el tamaño de la región de interés
rho_region = rho[  
    max(0, x_idx - region_size):min(nx, x_idx + region_size),
    max(0, y_idx - region_size):min(ny, y_idx + region_size)
]
temp_region = temperature[  
    max(0, x_idx - region_size):min(nx, x_idx + region_size),
    max(0, y_idx - region_size):min(ny, y_idx + region_size)
]

# Configurar los inputs en Streamlit
st.sidebar.title("Simulación de colapso gravitacional")
dt = st.sidebar.slider("Escalar paso de tiempo inicial (dt)", min_value=0.1, max_value=1000.0, value=dt_default, step=0.1)
G_multiplier = st.sidebar.number_input("Multiplicador de la constante gravitacional (G)", min_value=1, max_value=10000000, value=1000000, step=10)
G = G_default * G_multiplier
steps = st.sidebar.slider("Número de pasos de simulación", min_value=10, max_value=2000, value=500, step=10)
radiation_enabled = st.sidebar.checkbox("Habilitar efecto de radiación", value=True)

# Generar el GIF y obtener las historias de evolución
density_history, temperature_history, pressure_history, dt_history = create_density_evolution_gif(
    rho, temperature, dx, dy, steps, dt, G, radiation_enabled, output_path="density_collapse.gif"
)
st.image("density_collapse.gif")

# Crear gráficas iniciales
fig_density = go.Figure(data=go.Heatmap(
    z=rho,
    x=np.linspace(0, lx, nx),
    y=np.linspace(0, ly, ny),
    colorscale="Viridis",
    colorbar=dict(title="Densidad (kg/m³)")
))
fig_density.add_trace(go.Scatter(
    x=[y_idx * dx],
    y=[x_idx * dy],
    mode="markers",
    marker=dict(size=15, color="red"),
    name="Región de interés"
))
fig_density.update_layout(
    title="Densidad inicial de la nube",
    xaxis_title="x (m)",
    yaxis_title="y (m)"
)
st.plotly_chart(fig_density)

fig_temperature = go.Figure(data=go.Heatmap(
    z=temperature,
    x=np.linspace(0, lx, nx),
    y=np.linspace(0, ly, ny),
    colorscale="Plasma",
    colorbar=dict(title="Temperatura (K)")
))
fig_temperature.add_trace(go.Scatter(
    x=[y_idx * dx],
    y=[x_idx * dy],
    mode="markers",
    marker=dict(size=15, color="red"),
    name="Región de interés"
))
fig_temperature.update_layout(
    title="Temperatura inicial de la nube",
    xaxis_title="x (m)",
    yaxis_title="y (m)"
)
st.plotly_chart(fig_temperature)

fig_pressure = go.Figure(data=go.Heatmap(
    z=pressure,
    x=np.linspace(0, lx, nx),
    y=np.linspace(0, ly, ny),
    colorscale="Inferno",
    colorbar=dict(title="Presión (Pa)")
))
fig_pressure.add_trace(go.Scatter(
    x=[y_idx * dx],
    y=[x_idx * dy],
    mode="markers",
    marker=dict(size=15, color="red"),
    name="Región de interés"
))
fig_pressure.update_layout(
    title="Presión inicial de la nube",
    xaxis_title="x (m)",
    yaxis_title="y (m)"
)
st.plotly_chart(fig_pressure)

# Gráficas de evolución temporal
fig_evolution_density = go.Figure()
fig_evolution_density.add_trace(go.Scatter(y=density_history, mode="lines", name="Densidad máxima"))
fig_evolution_density.update_layout(
    title="Evolución temporal de la densidad máxima",
    xaxis_title="Iteración",
    yaxis_title="Densidad máxima (kg/m³)"
)
st.plotly_chart(fig_evolution_density)

fig_evolution_temperature = go.Figure()
fig_evolution_temperature.add_trace(go.Scatter(y=temperature_history, mode="lines", name="Temperatura máxima"))
fig_evolution_temperature.update_layout(
    title="Evolución temporal de la temperatura máxima",
    xaxis_title="Iteración",
    yaxis_title="Temperatura máxima (K)"
)
st.plotly_chart(fig_evolution_temperature)

fig_evolution_pressure = go.Figure()
fig_evolution_pressure.add_trace(go.Scatter(y=pressure_history, mode="lines", name="Presión máxima"))
fig_evolution_pressure.update_layout(
    title="Evolución temporal de la presión máxima",
    xaxis_title="Iteración",
    yaxis_title="Presión máxima (Pa)"
)
st.plotly_chart(fig_evolution_pressure)

fig_evolution_dt = go.Figure()
fig_evolution_dt.add_trace(go.Scatter(y=dt_history, mode="lines", name="Paso de tiempo adaptativo"))
fig_evolution_dt.update_layout(
    title="Evolución temporal del paso de tiempo adaptativo",
    xaxis_title="Iteración",
    yaxis_title="Paso de tiempo (dt)"
)
st.plotly_chart(fig_evolution_dt)



##############



import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageColor


def hex_to_rgb(hex_color):
    """Convert hexadecimal color to an RGB tuple."""
    return tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

#def generate_star_field(image_size, num_stars):
#    """
#    Generate a star field as a PIL image with white stars.

#    Parameters:
#        image_size (tuple): Size of the image (width, height).
#        num_stars (int): Number of stars to generate.

#    Returns:
#        PIL.Image: Image with generated stars.
#    """
#    width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    for _ in range(num_stars):
#        x = np.random.randint(0, width)
#        y = np.random.randint(0, height)
#        size = np.random.randint(1, 4)
#        brightness = np.random.randint(200, 255)  # Brightness for white stars
#        draw.ellipse(
#            [x - size, y - size, x + size, y + size],
#            fill=(255, 255, 255, brightness)
#        )

#    return img


from PIL import Image, ImageDraw, ImageFilter
import numpy as np

#from PIL import Image, ImageDraw, ImageFilter
#import numpy as np

#def generate_star_field(image_size, num_stars, diffuse_effect=True):
#    """
#    Generate a star field with stars that have a diffuse effect for the background.

#    Parameters:
#        image_size (tuple): Size of the image (width, height).
#        num_stars (int): Number of stars to generate.
#        diffuse_effect (bool): If True, applies a diffuse blur to the stars.

#    Returns:
#        PIL.Image: Image with generated stars.
 #   """
 #   width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
#    draw = ImageDraw.Draw(img)

#    for _ in range(num_stars):
#        # Randomize position, size, and brightness
#        x = np.random.randint(0, width)
#        y = np.random.randint(0, height)
#        size = np.random.randint(0.01, 2)  # Small size for background stars
#        brightness = np.random.randint(150, 555)  # Brightness range

#        # Generate color variation (white, yellowish, bluish)
#        color = (
#            np.random.randint(brightness - 50, brightness),
#            np.random.randint(brightness - 50, brightness),
#            brightness
#        )

#        # Draw the star core
#        draw.ellipse(
#            [x - size, y - size, x + size, y + size],
#            fill=color + (255,)
#        )

#        # Add glow (halo effect) for diffuse stars
#        if diffuse_effect:
#            for r in range(size + 1, size * 3):
#                alpha = int(255 * (1 - (r - size) / (size * 2)))  # Gradually decrease opacity
#                draw.ellipse(
#                    [x - r, y - r, x + r, y + r],
#                    outline=color + (alpha,)
#                )

    # Apply blur only to the background stars (diffuse effect)
    #if diffuse_effect:
    #    img = img.filter(ImageFilter.GaussianBlur(radius=2))  # Small blur for diffusion

#    return img


from PIL import Image, ImageDraw
import numpy as np

def generate_star_field(image_size, num_stars, min_points=5, max_points=8,
                        min_size=0.001, max_size=0.02, diffuse_effect=True):
    """
    Generate a star field with astronomical-looking stars.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        num_stars (int): Number of stars to generate.
        min_points (int): Minimum number of points for a star's shape (default: 5).
        max_points (int): Maximum number of points for a star's shape (default: 8).
        min_size (float): Minimum size of a star (default: 0.1).
        max_size (float): Maximum size of a star (default: 1.0).
        diffuse_effect (bool): If True, applies a diffuse blur to the stars for a soft effect (default: True).

    Returns:
        PIL.Image: Image with generated astronomical-like stars.
    """

    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    for _ in range(num_stars):
        # Randomize position, size, and brightness
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        size = np.random.uniform(min_size, max_size) * min(width, height) / 5.0  # Scale based on image size
        brightness = np.random.randint(150, 555)  # Brightness range
        # Generate color variation (white, yellowish, bluish)
        #color = (
        #    np.random.randint(brightness - 50, brightness),
        #    np.random.randint(brightness - 50, brightness),
        #    brightness
        #)

        # Variación de color con tonos azules y rojos
        r_offset = np.random.randint(-20, 20)  # Variación roja
        b_offset = np.random.randint(-20, 20)  # Variación azul
        r = np.clip(brightness + r_offset, 0, 255)
        g = np.clip(brightness, 0, 255)
        b = np.clip(brightness + b_offset, 0, 255)
        color = (int(r), int(g), int(b))
        
        # Create astronomical star shape (random number of points within range)
        num_points = np.random.randint(min_points, max_points + 1)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)  # Evenly spaced points

        # Calculate star coordinates based on size and angles
        star_coords = [(x + size * np.cos(angle), y + size * np.sin(angle))
                       for angle in angles]

        # Draw the filled star shape
        draw.polygon(star_coords, fill=color + (255,))

        # Optionally add diffuse glow (halo effect)
        if diffuse_effect:
            for r in range(int(size * 0.5), int(size * 1.5)):  # Adjust blur radius based on size
                alpha = int(255 * (1 - (r - size * 0.5) / size))  # Gradually decrease opacity
                draw.polygon([(x + r * np.cos(angle), y + r * np.sin(angle))
                              for angle in angles],
                             outline=color + (alpha,))

    return img



#def generate_filaments(image_size, center, num_filaments, radius, filament_length, start_color, end_color, blur_radius, elliptical):
#    width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    for _ in range(num_filaments):
#        angle = np.random.uniform(0, 2 * np.pi)
#        if elliptical:
#            semi_minor_axis = radius * 0.6
#            start_x = int(center[0] + radius * np.cos(angle))
#            start_y = int(center[1] + semi_minor_axis * np.sin(angle))
#        else:
#            start_x = int(center[0] + radius * np.cos(angle))
#            start_y = int(center[1] + radius * np.sin(angle))

#        end_x = int(start_x + filament_length * np.cos(angle))
#        end_y = int(start_y + filament_length * np.sin(angle))

#        for i in range(filament_length):
#            t = i / filament_length
#            x = int(start_x + t * (end_x - start_x))
#            y = int(start_y + t * (end_y - start_y))
#            thickness = max(1, int(5 * (1 - t**2)))
#            alpha = int(255 * (1 - t))

#            r = int(start_color[0] + t * (end_color[0] - start_color[0]))
#            g = int(start_color[1] + t * (end_color[1] - start_color[1]))
#            b = int(start_color[2] + t * (end_color[2] - start_color[2]))

#            draw.ellipse(
#                [
#                    x - thickness // 2,
#                    y - thickness // 2,
#                    x + thickness // 2,
#                    y + thickness // 2,
#                ],
#                fill=(r, g, b, alpha),
#            )

#    return img.filter(ImageFilter.GaussianBlur(blur_radius))

#def generate_diffuse_gas(image_size, center, inner_radius, outer_radius, start_color, end_color, blur_radius, elliptical):
#    width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    for r in range(inner_radius, outer_radius):
#        t = (r - inner_radius) / (outer_radius - inner_radius)
#        alpha = int(255 * (1 - t))

#        r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
#        g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
#        b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

#        if elliptical:
#            semi_minor_axis = r * 0.6
#            draw.ellipse(
#                [
#                    center[0] - r, center[1] - semi_minor_axis,
#                    center[0] + r, center[1] + semi_minor_axis
#                ],
#                outline=(r_color, g_color, b_color, alpha), width=1
#            )
#        else:
#            draw.ellipse(
#                [
#                    center[0] - r, center[1] - r, center[0] + r, center[1] + r
#                ],
#                outline=(r_color, g_color, b_color, alpha), width=1
#            )

#    return img.filter(ImageFilter.GaussianBlur(blur_radius))

#def generate_bubble(image_size, center, inner_radius, outer_radius, start_color, end_color, turbulence, blur_radius, elliptical):
#    width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    for r in range(inner_radius, outer_radius):
#        t = (r - inner_radius) / (outer_radius - inner_radius)
#        alpha = int(255 * (1 - t))

#        r_ = int(start_[0] + t * (end_[0] - start_[0]))
#        g_ = int(start_[1] + t * (end_[1] - start_[1]))
#        b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

#        offset_x = int(turbulence * np.random.uniform(-1, 1))
#        offset_y = int(turbulence * np.random.uniform(-1, 1))

#        if elliptical:
#            semi_minor_axis = r * 0.6
#            draw.ellipse(
#                [
#                    center[0] - r + offset_x, center[1] - semi_minor_axis + offset_y,
#                    center[0] + r + offset_x, center[1] + semi_minor_axis + offset_y
#                ],
#                outline=(r_color, g_color, b_color, alpha), width=1
#            )
#        else:
#            draw.ellipse(
#                [
#                    center[0] - r + offset_x, center[1] - r + offset_y,
#                    center[0] + r + offset_x, center[1] + r + offset_y
#                ],
#                outline=(r_color, g_color, b_color, alpha), width=1
#            )

#    return img.filter(ImageFilter.GaussianBlur(blur_radius))

#def generate_gas_arcs(image_size, center, radius, thickness, start_angle, end_angle, start_color, end_color, turbulence, blur_radius, elliptical):
#    width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    for t in range(thickness):
#        for angle in np.linspace(np.radians(start_angle), np.radians(end_angle), 500):
#            if elliptical:
#                semi_minor_axis = radius * 0.6
#                current_radius = radius + t + int(turbulence * np.random.uniform(-1, 1))
#                x = int(center[0] + current_radius * np.cos(angle))
#                y = int(center[1] + semi_minor_axis * np.sin(angle))
#            else:
#                current_radius = radius + t + int(turbulence * np.random.uniform(-1, 1))
#                x = int(center[0] + current_radius * np.cos(angle))
#                y = int(center[1] + current_radius * np.sin(angle))

#            t_angle = (angle - np.radians(start_angle)) / (np.radians(end_angle) - np.radians(start_angle))
#            r_color = int(start_color[0] + t_angle * (end_color[0] - start_color[0]))
#            g_color = int(start_color[1] + t_angle * (end_color[1] - start_color[1]))
#            b_color = int(start_color[2] + t_angle * (end_color[2] - start_color[2]))

#            draw.point((x, y), fill=(r_color, g_color, b_color, 255))

#    return img.filter(ImageFilter.GaussianBlur(blur_radius))

#def draw_central_star_with_filaments(image_size, position, star_size, halo_size, color, num_filaments, dispersion, blur_radius):
#    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    # Draw halo
#    halo = Image.new("RGBA", (halo_size * 2, halo_size * 2), (0, 0, 0, 0))
#    halo_draw = ImageDraw.Draw(halo)
#    halo_draw.ellipse((0, 0, halo_size * 2, halo_size * 2), fill=color + (50,))
#    halo = halo.filter(ImageFilter.GaussianBlur(radius=halo_size / 2))
#    img.paste(halo, (position[0] - halo_size, position[1] - halo_size), halo)

#    # Draw star
#    draw.ellipse(
#        (position[0] - star_size, position[1] - star_size, position[0] + star_size, position[1] + star_size),
#        fill=color + (255,),
#    )

#    # Draw radial filaments
#    filament_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
#    filament_draw = ImageDraw.Draw(filament_layer)
#    for i in range(num_filaments):
#        angle = 2 * np.pi * i / num_filaments
#        end_x = position[0] + (halo_size + np.random.uniform(-dispersion, dispersion)) * np.cos(angle)
#        end_y = position[1] + (halo_size + np.random.uniform(-dispersion, dispersion)) * np.sin(angle)

#        filament_draw.line(
#            [(position[0], position[1]), (end_x, end_y)],
#            fill=color + (100,),  # Semi-transparent
#            width=2,
#        )

#    # Apply Gaussian blur to the filaments for a diffuse effect
#    filament_layer = filament_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

#    # Combine the filaments with the star image
#    img = Image.alpha_composite(img, filament_layer)

#    return img

#def draw_central_star(image_size, position, star_size, halo_size, color):
#    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    # Draw halo
#    halo = Image.new("RGBA", (halo_size * 2, halo_size * 2), (0, 0, 0, 0))
#    halo_draw = ImageDraw.Draw(halo)
#    halo_draw.ellipse((0, 0, halo_size * 2, halo_size * 2), fill=color + (50,))
#    halo = halo.filter(ImageFilter.GaussianBlur(radius=halo_size / 2))
#    img.paste(halo, (position[0] - halo_size, position[1] - halo_size), halo)

#    # Draw star
#    draw.ellipse(
#        (position[0] - star_size, position[1] - star_size, position[0] + star_size, position[1] + star_size),
#        fill=color + (255,),
#    )

#    return img
########################################################

def generate_filaments(image_size, centers, num_filaments, radius, filament_length, start_color, end_color, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for center in centers:
        for _ in range(num_filaments):
            angle = np.random.uniform(0, 2 * np.pi)
            if elliptical:
                semi_minor_axis = radius * 0.6
                start_x = int(center[0] + radius * np.cos(angle))
                start_y = int(center[1] + semi_minor_axis * np.sin(angle))
            else:
                start_x = int(center[0] + radius * np.cos(angle))
                start_y = int(center[1] + radius * np.sin(angle))

            end_x = int(start_x + filament_length * np.cos(angle))
            end_y = int(start_y + filament_length * np.sin(angle))

            for i in range(filament_length):
                t = i / filament_length
                x = int(start_x + t * (end_x - start_x))
                y = int(start_y + t * (end_y - start_y))
                thickness = max(1, int(5 * (1 - t**2)))
                alpha = int(255 * (1 - t))

                r = int(start_color[0] + t * (end_color[0] - start_color[0]))
                g = int(start_color[1] + t * (end_color[1] - start_color[1]))
                b = int(start_color[2] + t * (end_color[2] - start_color[2]))

                draw.ellipse(
                    [
                        x - thickness // 2,
                        y - thickness // 2,
                        x + thickness // 2,
                        y + thickness // 2,
                    ],
                    fill=(r, g, b, alpha),
                )

    return img.filter(ImageFilter.GaussianBlur(blur_radius))


def generate_diffuse_gas(image_size, centers, inner_radius, outer_radius, start_color, end_color, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for center in centers:
        for r in range(inner_radius, outer_radius):
            t = (r - inner_radius) / (outer_radius - inner_radius)
            alpha = int(255 * (1 - t))

            r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
            g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
            b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

            if elliptical:
                semi_minor_axis = r * 0.6
                draw.ellipse(
                    [
                        center[0] - r, center[1] - semi_minor_axis,
                        center[0] + r, center[1] + semi_minor_axis
                    ],
                    outline=(r_color, g_color, b_color, alpha), width=1
                )
            else:
                draw.ellipse(
                    [
                        center[0] - r, center[1] - r, center[0] + r, center[1] + r
                    ],
                    outline=(r_color, g_color, b_color, alpha), width=1
                )

    return img.filter(ImageFilter.GaussianBlur(blur_radius))


def generate_bubble(image_size, centers, inner_radius, outer_radius, start_color, end_color, turbulence, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for center in centers:
        for r in range(inner_radius, outer_radius):
            t = (r - inner_radius) / (outer_radius - inner_radius)
            alpha = int(255 * (1 - t))

            r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
            g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
            b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

            offset_x = int(turbulence * np.random.uniform(-1, 1))
            offset_y = int(turbulence * np.random.uniform(-1, 1))

            if elliptical:
                semi_minor_axis = r * 0.6
                draw.ellipse(
                    [
                        center[0] - r + offset_x, center[1] - semi_minor_axis + offset_y,
                        center[0] + r + offset_x, center[1] + semi_minor_axis + offset_y
                    ],
                    outline=(r_color, g_color, b_color, alpha), width=1
                )
            else:
                draw.ellipse(
                    [
                        center[0] - r + offset_x, center[1] - r + offset_y,
                        center[0] + r + offset_x, center[1] + r + offset_y
                    ],
                    outline=(r_color, g_color, b_color, alpha), width=1
                )

    return img.filter(ImageFilter.GaussianBlur(blur_radius))


def generate_gas_arcs(image_size, centers, radius, thickness, start_angle, end_angle, start_color, end_color, turbulence, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for center in centers:
        for t in range(thickness):
            for angle in np.linspace(np.radians(start_angle), np.radians(end_angle), 500):
                if elliptical:
                    semi_minor_axis = radius * 0.6
                    current_radius = radius + t + int(turbulence * np.random.uniform(-1, 1))
                    x = int(center[0] + current_radius * np.cos(angle))
                    y = int(center[1] + semi_minor_axis * np.sin(angle))
                else:
                    current_radius = radius + t + int(turbulence * np.random.uniform(-1, 1))
                    x = int(center[0] + current_radius * np.cos(angle))
                    y = int(center[1] + current_radius * np.sin(angle))

                t_angle = (angle - np.radians(start_angle)) / (np.radians(end_angle) - np.radians(start_angle))
                r_color = int(start_color[0] + t_angle * (end_color[0] - start_color[0]))
                g_color = int(start_color[1] + t_angle * (end_color[1] - start_color[1]))
                b_color = int(start_color[2] + t_angle * (end_color[2] - start_color[2]))

                draw.point((x, y), fill=(r_color, g_color, b_color, 255))

    return img.filter(ImageFilter.GaussianBlur(blur_radius))


##########################################################
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def draw_star_with_filaments(img, position, star_size, halo_size, color, num_filaments, dispersion, blur_radius):
    """
    Draw a star with adjustable filaments, granular structure, and sunspots.

    Parameters:
        img (PIL.Image): The base image.
        position (tuple): The position of the star (x, y).
        star_size (int): The size of the star's core.
        halo_size (int): The size of the star's halo.
        color (tuple): RGB color of the star and filaments.
        num_filaments (int): Number of filaments.
        dispersion (int): Filament dispersion.
        blur_radius (float): Blur radius for the filaments.
    """
    # Ensure minimum star size to avoid errors
    if star_size < 6:
        star_size = 6

    # Draw halo with smooth gradient effect
    halo = Image.new("RGBA", (halo_size * 2, halo_size * 2), (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    for r in range(halo_size, 0, -1):
        alpha = int(100 * (r / halo_size))  # Gradually decrease opacity
        halo_draw.ellipse(
            (halo_size - r, halo_size - r, halo_size + r, halo_size + r),
            fill=color + (alpha,)
        )
    halo = halo.filter(ImageFilter.GaussianBlur(radius=halo_size / 2))
    img.paste(halo, (position[0] - halo_size, position[1] - halo_size), halo)

    # Draw star core with granular structure and sunspots
    star_layer = Image.new("RGBA", (star_size * 2, star_size * 2), (0, 0, 0, 0))
    star_draw = ImageDraw.Draw(star_layer)
    for r in range(star_size, 0, -1):
        star_color = (
            int(color[0] * (0.8 + 0.2 * (r / star_size))),
            int(color[1] * (0.8 + 0.2 * (r / star_size))),
            int(color[2] * (0.8 + 0.2 * (r / star_size))),
            255  # Fully opaque
        )
        star_draw.ellipse(
            (star_size - r, star_size - r, star_size + r, star_size + r),
            fill=star_color
        )

    # Add solar granulation texture
    granulation_density = max(50, star_size * 10)  # Adjust density for smaller stars
    for _ in range(granulation_density):
        x = np.random.randint(0, star_size * 2)
        y = np.random.randint(0, star_size * 2)
        intensity = np.random.randint(150, 255)
        size = np.random.randint(1, max(2, star_size // 20))  # Smaller granulation size for smaller stars
        star_draw.ellipse(
            (x - size, y - size, x + size, y + size),
            fill=(intensity, intensity, 0, 255)  # Fully opaque
        )

    # Add sunspots (dark areas)
    sunspot_count = max(1, star_size // 10)  # Fewer sunspots for smaller stars
    for _ in range(sunspot_count):
        spot_x = np.random.randint(0, star_size * 2)
        spot_y = np.random.randint(0, star_size * 2)
        spot_size = np.random.randint(max(1, star_size // 50), max(2, star_size // 20))  # Adjust sunspot size for smaller stars
        star_draw.ellipse(
            (spot_x - spot_size, spot_y - spot_size, spot_x + spot_size, spot_y + spot_size),
            fill=(0, 0, 0, 255)  # Fully opaque
        )

    # Mask to ensure circular profile
    mask = Image.new("L", (star_size * 2, star_size * 2), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, star_size * 2, star_size * 2), fill=255)

    star_circle = Image.new("RGBA", img.size, (0, 0, 0, 0))
    star_circle.paste(star_layer, (position[0] - star_size, position[1] - star_size), mask=mask)
    img = Image.alpha_composite(img, star_circle)

    # Draw radial filaments with glow effect
    filament_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    filament_draw = ImageDraw.Draw(filament_layer)
    for i in range(num_filaments):
        angle = 2 * np.pi * i / num_filaments + np.random.uniform(-0.1, 0.1)
        end_x = position[0] + (halo_size + np.random.uniform(-dispersion, dispersion)) * np.cos(angle)
        end_y = position[1] + (halo_size + np.random.uniform(-dispersion, dispersion)) * np.sin(angle)

        # Randomize filament opacity, width, and brightness for realism
        opacity = np.random.randint(200, 255)
        width = np.random.randint(2, 6)

        # Add a central filament
        filament_draw.line(
            [(position[0], position[1]), (end_x, end_y)],
            fill=color + (opacity,),
            width=width,
        )

        # Add radial glow around the filament
        for offset in range(1, 4):
            glow_opacity = max(10, opacity - offset * 50)  # Reduce opacity for glow layers
            filament_draw.line(
                [(position[0], position[1]), (end_x, end_y)],
                fill=color + (glow_opacity,),
                width=width + offset,
            )

    # Apply Gaussian blur only to the filament layer
    filament_layer = filament_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Combine the filaments with the star image
    img = Image.alpha_composite(img, filament_layer)

    return img


def draw_multiple_stars(image_size, star_configs):
    """
    Draw multiple stars with individual configurations.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        star_configs (list): List of dictionaries, each containing the parameters for a star.
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))

    for config in star_configs:
        img = draw_star_with_filaments(
            img,
            position=config["position"],
            star_size=config["star_size"],
            halo_size=config["halo_size"],
            color=config["color"],
            num_filaments=config["num_filaments"],
            dispersion=config["dispersion"],
            blur_radius=config["blur_radius"],
        )

    return img




def generate_bubble_texture(image_size, center, radius, line_color, line_opacity, num_lines, blur_radius):
    """
    Generate a textured bubble with intersecting semi-transparent lines.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        center (tuple): Center of the bubble (x, y).
        radius (int): Radius of the bubble.
        line_color (tuple): RGB color of the lines.
        line_opacity (int): Opacity of the lines.
        num_lines (int): Number of intersecting lines.
        blur_radius (int): Gaussian blur radius for smoothing lines.

    Returns:
        PIL.Image: Image with textured bubble.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for _ in range(num_lines):
        start_angle = np.random.uniform(0, 2 * np.pi)
        end_angle = start_angle + np.random.uniform(np.pi / 6, np.pi / 3)

        start_x = int(center[0] + radius * np.cos(start_angle))
        start_y = int(center[1] + radius * np.sin(start_angle))
        end_x = int(center[0] + radius * np.cos(end_angle))
        end_y = int(center[1] + radius * np.sin(end_angle))

        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=line_color + (line_opacity,),
            width=2
        )

    return img.filter(ImageFilter.GaussianBlur(blur_radius))


# Streamlit UI
st.title("Nebula Simulation with Circular and Elliptical Layers")

st.sidebar.header("Controls")
with st.sidebar.expander("Image size"):
    image_width = st.slider("Image Width", 400, 1600, 800)
    image_height = st.slider("Image Height", 400, 1600, 800)
    center_x = st.slider("Center X", 0, image_width, image_width // 2)
    center_y = st.slider("Center Y", 0, image_height, image_height // 2)
    center = (center_x, center_y)

# Filament parameters
#num_filaments = st.sidebar.slider("Number of Filaments", 10, 500, 100)
#filament_radius = st.sidebar.slider("Filament Radius", 10, 400, 200)
#filament_length = st.sidebar.slider("Filament Length", 10, 300, 100)
#filament_start_color = st.sidebar.color_picker("Filament Start Color", "#FFA500")
#filament_end_color = st.sidebar.color_picker("Filament End Color", "#FF4500")
#filament_blur = st.sidebar.slider("Filament Blur", 0, 30, 5)
#filament_elliptical = st.sidebar.checkbox("Elliptical Filaments", False)

# Diffuse gas parameters
#gas_inner_radius = st.sidebar.slider("Gas Inner Radius", 50, 400, 150)
#gas_outer_radius = st.sidebar.slider("Gas Outer Radius", 100, 500, 300)
#gas_start_color = st.sidebar.color_picker("Gas Start Color", "#FF4500")
#gas_end_color = st.sidebar.color_picker("Gas End Color", "#0000FF")
#gas_blur = st.sidebar.slider("Gas Blur", 0, 50, 20)
#gas_elliptical = st.sidebar.checkbox("Elliptical Gas", False)

# Bubble parameters (continuación)
#bubble_inner_radius = st.sidebar.slider("Bubble Inner Radius", 10, 200, 50)
#bubble_outer_radius = st.sidebar.slider("Bubble Outer Radius", 50, 300, 150)
#bubble_start_color = st.sidebar.color_picker("Bubble Start Color", "#FF00FF")
#bubble_end_color = st.sidebar.color_picker("Bubble End Color", "#000000")
#bubble_turbulence = st.sidebar.slider("Bubble Turbulence", 0.0, 10.0, 2.0)
#bubble_blur = st.sidebar.slider("Bubble Blur", 0, 30, 10)
#bubble_elliptical = st.sidebar.checkbox("Elliptical Bubble", False)

# Arc parameters
#arc_radius = st.sidebar.slider("Arc Radius", 50, 300, 150)
#arc_thickness = st.sidebar.slider("Arc Thickness", 1, 20, 5)
#arc_start_angle = st.sidebar.slider("Arc Start Angle", 0, 360, 0)
#arc_end_angle = st.sidebar.slider("Arc End Angle", 0, 360, 180)
#arc_start_color = st.sidebar.color_picker("Arc Start Color", "#FFFFFF")
#arc_end_color = st.sidebar.color_picker("Arc End Color", "#CCCCCC")
#arc_turbulence = st.sidebar.slider("Arc Turbulence", 0.0, 10.0, 2.0)
#arc_blur = st.sidebar.slider("Arc Blur", 0, 30, 5)
#arc_elliptical = st.sidebar.checkbox("Elliptical Arcs", False)

# Centers for layers
num_centers = st.sidebar.slider("Number of Centers", 1, 10, 1)
centers = [
    (
        st.sidebar.slider(f"Center X {i+1}", 0, image_width, image_width // 2, key=f"center_x_{i}"),
        st.sidebar.slider(f"Center Y {i+1}", 0, image_height, image_height // 2, key=f"center_y_{i}")
    )
    for i in range(num_centers)
]

# Sidebar for filament parameters
#st.sidebar.header("Filament Parameters")
#num_filaments = st.sidebar.slider("Number of Filaments", 10, 500, 100)
#filament_radius = st.sidebar.slider("Filament Radius", 10, 400, 200)
#filament_length = st.sidebar.slider("Filament Length", 10, 300, 100)
#filament_start_color = st.sidebar.color_picker("Filament Start Color", "#FFA500")
#filament_end_color = st.sidebar.color_picker("Filament End Color", "#FF4500")
#filament_blur = st.sidebar.slider("Filament Blur", 0, 30, 5)
#filament_elliptical = st.sidebar.checkbox("Elliptical Filaments", False)

# Filament configurations
num_filament_layers = st.sidebar.slider("Number of Filament Layers", 1, 5, 1)
filament_configs = []
for i in range(num_filament_layers):
    with st.sidebar.expander(f"Filament Layer {i + 1}"):
        num_filaments = st.slider(f"Number of Filaments (Layer {i + 1})", 10, 500, 100, key=f"num_filaments_{i}")
        radius = st.slider(f"Filament Radius (Layer {i + 1})", 10, 400, 200, key=f"radius_{i}")
        length = st.slider(f"Filament Length (Layer {i + 1})", 10, 300, 100, key=f"length_{i}")
        start_color = st.color_picker(f"Start Color (Layer {i + 1})", "#FFA500", key=f"start_color_{i}")
        end_color = st.color_picker(f"End Color (Layer {i + 1})", "#FF4500", key=f"end_color_{i}")
        blur = st.slider(f"Filament Blur (Layer {i + 1})", 0, 30, 5, key=f"blur_{i}")
        elliptical = st.checkbox(f"Elliptical Filaments (Layer {i + 1})", False, key=f"elliptical_{i}")
        filament_configs.append({
            "num_filaments": num_filaments,
            "radius": radius,
            "length": length,
            "start_color": hex_to_rgb(start_color),
            "end_color": hex_to_rgb(end_color),
            "blur": blur,
            "elliptical": elliptical
        })

# Diffuse gas configurations
num_gas_layers = st.sidebar.slider("Number of Gas Layers", 1, 5, 1)
gas_configs = []
for i in range(num_gas_layers):
    with st.sidebar.expander(f"Gas Layer {i + 1}"):
        inner_radius = st.slider(f"Inner Radius (Layer {i + 1})", 50, 400, 150, key=f"inner_radius_{i}")
        outer_radius = st.slider(f"Outer Radius (Layer {i + 1})", 100, 500, 300, key=f"outer_radius_{i}")
        start_color = st.color_picker(f"Start Color (Layer {i + 1})", "#FF4500", key=f"gas_start_color_{i}")
        end_color = st.color_picker(f"End Color (Layer {i + 1})", "#0000FF", key=f"gas_end_color_{i}")
        blur = st.slider(f"Gas Blur (Layer {i + 1})", 0, 50, 20, key=f"gas_blur_{i}")
        elliptical = st.checkbox(f"Elliptical Gas (Layer {i + 1})", False, key=f"gas_elliptical_{i}")
        gas_configs.append({
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "start_color": hex_to_rgb(start_color),
            "end_color": hex_to_rgb(end_color),
            "blur": blur,
            "elliptical": elliptical
        })


# Bubble configurations
num_bubble_layers = st.sidebar.slider("Number of Bubble Layers", 1, 5, 1)
bubble_configs = []
for i in range(num_bubble_layers):
    with st.sidebar.expander(f"Bubble Layer {i + 1}"):
        inner_radius = st.slider(f"Inner Radius (Layer {i + 1})", 10, 200, 50, key=f"bubble_inner_radius_{i}")
        outer_radius = st.slider(f"Outer Radius (Layer {i + 1})", 50, 300, 150, key=f"bubble_outer_radius_{i}")
        start_color = st.color_picker(f"Start Color (Layer {i + 1})", "#FF00FF", key=f"bubble_start_color_{i}")
        end_color = st.color_picker(f"End Color (Layer {i + 1})", "#000000", key=f"bubble_end_color_{i}")
        turbulence = st.slider(f"Turbulence (Layer {i + 1})", 0.0, 10.0, 2.0, key=f"bubble_turbulence_{i}")
        blur = st.slider(f"Bubble Blur (Layer {i + 1})", 0, 30, 10, key=f"bubble_blur_{i}")
        elliptical = st.checkbox(f"Elliptical Bubbles (Layer {i + 1})", False, key=f"bubble_elliptical_{i}")
        bubble_configs.append({
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "start_color": hex_to_rgb(start_color),
            "end_color": hex_to_rgb(end_color),
            "turbulence": turbulence,
            "blur": blur,
            "elliptical": elliptical
        })


# Sidebar for diffuse gas parameters
#st.sidebar.header("Diffuse Gas Parameters")
#gas_inner_radius = st.sidebar.slider("Gas Inner Radius", 50, 400, 150)
#gas_outer_radius = st.sidebar.slider("Gas Outer Radius", 100, 500, 300)
#gas_start_color = st.sidebar.color_picker("Gas Start Color", "#FF4500")
#gas_end_color = st.sidebar.color_picker("Gas End Color", "#0000FF")
#gas_blur = st.sidebar.slider("Gas Blur", 0, 50, 20)
#gas_elliptical = st.sidebar.checkbox("Elliptical Gas", False)

# Sidebar for bubble parameters
#st.sidebar.header("Bubble Parameters")
#bubble_inner_radius = st.sidebar.slider("Bubble Inner Radius", 10, 200, 50)
#bubble_outer_radius = st.sidebar.slider("Bubble Outer Radius", 50, 300, 150)
#bubble_start_color = st.sidebar.color_picker("Bubble Start Color", "#FF00FF")
#bubble_end_color = st.sidebar.color_picker("Bubble End Color", "#000000")
#bubble_turbulence = st.sidebar.slider("Bubble Turbulence", 0.0, 10.0, 2.0)
#bubble_blur = st.sidebar.slider("Bubble Blur", 0, 30, 10)
#bubble_elliptical = st.sidebar.checkbox("Elliptical Bubble", False)

# Sidebar for arc parameters
#st.sidebar.header("Arc Parameters")
#arc_radius = st.sidebar.slider("Arc Radius", 50, 300, 150)
#arc_thickness = st.sidebar.slider("Arc Thickness", 1, 20, 5)
#arc_start_angle = st.sidebar.slider("Arc Start Angle", 0, 360, 0)
#arc_end_angle = st.sidebar.slider("Arc End Angle", 0, 360, 180)
#arc_start_color = st.sidebar.color_picker("Arc Start Color", "#FFFFFF")
#arc_end_color = st.sidebar.color_picker("Arc End Color", "#CCCCCC")
#arc_turbulence = st.sidebar.slider("Arc Turbulence", 0.0, 10.0, 2.0)
#arc_blur = st.sidebar.slider("Arc Blur", 0, 30, 5)
#arc_elliptical = st.sidebar.checkbox("Elliptical Arcs", False)




# Star field parameters
num_stars = st.sidebar.slider("Number of Stars", 50, 1000, 200)
star_colors = ["#FFFFFF", "#FFD700", "#87CEEB"]

# Central star parameters
#st.sidebar.header("Central Star")
#star_size = st.sidebar.slider("Star Size", 5, 50, 20)
#halo_size = st.sidebar.slider("Halo Size", 10, 100, 50)
#star_color_hex = st.sidebar.color_picker("Star Color", "#FFFF00")
#star_color = tuple(int(star_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
#num_star_filaments = st.sidebar.slider("Number of Star Filaments", 5, 100, 30)
#filament_dispersion = st.sidebar.slider("Filament Dispersion", 1, 50, 10)
#star_blur_radius = st.sidebar.slider("Star Blur Radius", 0, 20, 5)

# Central star parameters
st.sidebar.header("Star Configuration")

# Number of stars
num_cstars = st.sidebar.slider("Number of Stars", 1, 10, 1)

# Create a list of star configurations
star_configs = []
for i in range(num_cstars):
    st.sidebar.subheader(f"Star {i + 1} Parameters")
    position_x = st.sidebar.slider(f"Position X (Star {i + 1})", 0, 800, 400)
    position_y = st.sidebar.slider(f"Position Y (Star {i + 1})", 0, 800, 400)
    star_size = st.sidebar.slider(f"Star Size (Star {i + 1})", 5, 50, 20)
    halo_size = st.sidebar.slider(f"Halo Size (Star {i + 1})", 10, 100, 50)
    star_color_hex = st.sidebar.color_picker(f"Star Color (Star {i + 1})", "#FFFF00")
    star_color = tuple(int(star_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    num_star_filaments = st.sidebar.slider(f"Number of Star Filaments (Star {i + 1})", 5, 100, 30)
    filament_dispersion = st.sidebar.slider(f"Filament Dispersion (Star {i + 1})", 1, 50, 10)
    star_blur_radius = st.sidebar.slider(f"Star Blur Radius (Star {i + 1})", 0, 20, 5)

    # Add the star configuration to the list
    star_configs.append({
        "position": (position_x, position_y),
        "star_size": star_size,
        "halo_size": halo_size,
        "color": star_color,
        "num_filaments": num_star_filaments,
        "dispersion": filament_dispersion,
        "blur_radius": star_blur_radius,
    })


# Generate layers
#image_size = (image_width, image_height)
#filaments_image = generate_filaments(
#    image_size, center, num_filaments, filament_radius, filament_length,
#    tuple(int(filament_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    tuple(int(filament_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    filament_blur, filament_elliptical
#)
#diffuse_gas_image = generate_diffuse_gas(
#    image_size, center, gas_inner_radius, gas_outer_radius,
#    tuple(int(gas_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    tuple(int(gas_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    gas_blur, gas_elliptical
#)
#bubble_image = generate_bubble(
#    image_size, center, bubble_inner_radius, bubble_outer_radius,
#    tuple(int(bubble_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    tuple(int(bubble_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    bubble_turbulence, bubble_blur, bubble_elliptical
#)
#gas_arcs_image = generate_gas_arcs(
#    image_size, center, arc_radius, arc_thickness, arc_start_angle, arc_end_angle,
#    tuple(int(arc_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    tuple(int(arc_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
#    arc_turbulence, arc_blur, arc_elliptical
#)

image_size=(image_width, image_height)


# Generate filament layers
filaments_image = Image.new("RGBA", image_size, (0, 0, 0, 0))
for config in filament_configs:
    filaments_image = Image.alpha_composite(
        filaments_image,
        generate_filaments(
            image_size, centers, 
            config["num_filaments"], config["radius"], config["length"],
            config["start_color"], config["end_color"], 
            config["blur"], config["elliptical"]
        )
    )

# Generate gas layers
diffuse_gas_image = Image.new("RGBA", image_size, (0, 0, 0, 0))
for config in gas_configs:
    diffuse_gas_image = Image.alpha_composite(
        diffuse_gas_image,
        generate_diffuse_gas(
            image_size, centers, 
            config["inner_radius"], config["outer_radius"],
            config["start_color"], config["end_color"], 
            config["blur"], config["elliptical"]
        )
    )

# Generate bubble layers
bubble_image = Image.new("RGBA", image_size, (0, 0, 0, 0))
for config in bubble_configs:
    bubble_image = Image.alpha_composite(
        bubble_image,
        generate_bubble(
            image_size, centers, 
            config["inner_radius"], config["outer_radius"],
            config["start_color"], config["end_color"], 
            config["turbulence"], config["blur"], config["elliptical"]
        )
    )


# Generate layers
#filaments_image = generate_filaments(image_size, centers, num_filaments, filament_radius, filament_length, hex_to_rgb(filament_start_color), hex_to_rgb(filament_end_color), filament_blur, filament_elliptical)
#diffuse_gas_image = generate_diffuse_gas(image_size, centers, gas_inner_radius, gas_outer_radius, hex_to_rgb(gas_start_color), hex_to_rgb(gas_end_color), gas_blur, gas_elliptical)
#bubble_image = generate_bubble(image_size, centers, bubble_inner_radius, bubble_outer_radius, hex_to_rgb(bubble_start_color), hex_to_rgb(bubble_end_color), bubble_turbulence, bubble_blur, bubble_elliptical)
#gas_arcs_image = generate_gas_arcs(image_size, centers, arc_radius, arc_thickness, arc_start_angle, arc_end_angle, hex_to_rgb(arc_start_color), hex_to_rgb(arc_end_color), arc_turbulence, arc_blur, arc_elliptical)
star_field_image = generate_star_field(image_size, num_stars)
central_star_image = draw_multiple_stars(image_size, star_configs)


final_image = Image.alpha_composite(star_field_image, filaments_image)
final_image = Image.alpha_composite(final_image, diffuse_gas_image)
final_image = Image.alpha_composite(final_image, bubble_image)  # Añadimos las burbujas aquí
#final_image = Image.alpha_composite(final_image, gas_arcs_image)
final_image = Image.alpha_composite(final_image, central_star_image)

st.image(final_image, caption="Nebula Simulation", use_column_width=True)


# Combine layers
#final_image = Image.alpha_composite(star_field_image, filaments_image)
#final_image = Image.alpha_composite(final_image, diffuse_gas_image)
#final_image = Image.alpha_composite(final_image, bubble_image)
#final_image = Image.alpha_composite(final_image, gas_arcs_image)
#final_image = Image.alpha_composite(final_image, central_star_image)

# Display the final image
#st.image(final_image, caption="Nebula Simulation", use_column_width=True)




#star_field_image = generate_star_field(image_size, num_stars)

#central_star_image = draw_central_star_with_filaments(
#    image_size, center, star_size, halo_size, star_color, num_star_filaments,
#    filament_dispersion, star_blur_radius
#)

#central_star_image = draw_multiple_stars(image_size, star_configs)
#st.image(central_star_image, caption="Multiple Stars", use_column_width=True)



# Combine images
#final_image = Image.alpha_composite(star_field_image, filaments_image)
#final_image = Image.alpha_composite(final_image, diffuse_gas_image)
#final_image = Image.alpha_composite(final_image, bubble_image)
#final_image = Image.alpha_composite(final_image, gas_arcs_image)
#final_image = Image.alpha_composite(final_image, central_star_image)

# Display the final image
#st.image(final_image, use_column_width=True)




# Bubble texture parameters
texture_radius = st.sidebar.slider("Bubble Texture Radius", 50, 300, 150)
texture_color = st.sidebar.color_picker("Texture Line Color", "#FFFFFF")
texture_opacity = st.sidebar.slider("Texture Line Opacity", 50, 255, 100)
texture_num_lines = st.sidebar.slider("Number of Texture Lines", 10, 100, 50)
texture_blur = st.sidebar.slider("Texture Blur Radius", 0, 20, 5)

# Generate bubble texture
bubble_texture = generate_bubble_texture(
    image_size=(image_width, image_height),
    center=center,
    radius=texture_radius,
    line_color=tuple(int(texture_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    line_opacity=texture_opacity,
    num_lines=texture_num_lines,
    blur_radius=texture_blur
)
# Combine bubble texture with bubble layer
bubble_combined = Image.alpha_composite(bubble_image.convert("RGBA"), bubble_texture.convert("RGBA"))

# Combine images sequentially
final_image = Image.alpha_composite(star_field_image, filaments_image)
final_image = Image.alpha_composite(final_image, diffuse_gas_image)
final_image = Image.alpha_composite(final_image, bubble_combined)  # Use the combined bubble and texture
#final_image = Image.alpha_composite(final_image, gas_arcs_image)
final_image = Image.alpha_composite(final_image, central_star_image)

# Display the final image
st.image(final_image, caption="Nebula Simulation", use_column_width=True)


############################################################
#def generate_gaseous_shells(image_size, center, semi_major, semi_minor, angle, inner_radius, outer_radius, start_color, end_color, deformity, blur_radius):
#    """
#    Generate gaseous shells with elliptical profiles, deformities, and sinusoidal variations.

#    Parameters:
#        image_size: Tuple[int, int] - Size of the image (width, height).
#        center: Tuple[int, int] - Center of the shell (x, y).
#        semi_major: int - Semi-major axis for the elliptical profile.
#        semi_minor: int - Semi-minor axis for the elliptical profile.
#        angle: float - Rotation angle of the ellipse in degrees.
#        inner_radius: int - Inner radius of the shell.
#        outer_radius: int - Outer radius of the shell.
#        start_color: Tuple[int, int, int] - RGB color at the inner edge.
#        end_color: Tuple[int, int, int] - RGB color at the outer edge.
#        deformity: float - Degree of irregularity in the shell shape.
#        blur_radius: int - Gaussian blur radius for smoothing.

#    Returns:
#        PIL.Image: Image with the generated shell.
#    """
#    width, height = image_size
#    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#    draw = ImageDraw.Draw(img)

#    # Precompute rotation matrix for the angle
#    angle_rad = np.radians(angle)
#    cos_theta = np.cos(angle_rad)
#    sin_theta = np.sin(angle_rad)

#    for r in range(inner_radius, outer_radius):
#        t = (r - inner_radius) / (outer_radius - inner_radius)  # Normalized radius
#        alpha = int(255 * (1 - t))  # Fade as we move outward

#        # Gradual color transition
#        r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
#        g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
#        b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

#        # Generate points for the elliptical profile with sinusoidal deformity
#        points = []
#        for theta in np.linspace(0, 2 * np.pi, 200):  # 200 points for smooth shell
#            noise = deformity * np.sin(3 * theta + t * np.pi)  # Sinusoidal deformity
#            radius = r + noise
#            x_ellipse = radius * semi_major * np.cos(theta) / outer_radius
#            y_ellipse = radius * semi_minor * np.sin(theta) / outer_radius

            # Rotate the points by the specified angle
#            x_rotated = cos_theta * x_ellipse - sin_theta * y_ellipse
#            y_rotated = sin_theta * x_ellipse + cos_theta * y_ellipse

#            # Translate to the center
#            x = int(center[0] + x_rotated)
#            y = int(center[1] + y_rotated)
#            points.append((x, y))

#        draw.polygon(points, outline=(r_color, g_color, b_color, alpha))

#    # Apply Gaussian blur to smooth out edges
#    return img.filter(ImageFilter.GaussianBlur(blur_radius))

from noise import pnoise2
import random

def generate_gaseous_shells(
    image_size, center, semi_major, semi_minor, angle, inner_radius, outer_radius, start_color, end_color, deformity, turbulence_intensity, blur_radius
):
    """
    Generate gaseous shells with extreme turbulence and gas-like deformities.

    Parameters:
        image_size: Tuple[int, int] - Size of the image (width, height).
        center: Tuple[int, int] - Center of the shell (x, y).
        semi_major: int - Semi-major axis for the elliptical profile.
        semi_minor: int - Semi-minor axis for the elliptical profile.
        angle: float - Rotation angle of the ellipse in degrees.
        inner_radius: int - Inner radius of the shell.
        outer_radius: int - Outer radius of the shell.
        start_color: Tuple[int, int, int] - RGB color at the inner edge.
        end_color: Tuple[int, int, int] - RGB color at the outer edge.
        deformity: float - Degree of sinusoidal irregularity in the shell shape.
        turbulence_intensity: float - Scale of random noise for extreme deformities.
        blur_radius: int - Gaussian blur radius for smoothing.

    Returns:
        PIL.Image: Image with the generated shell.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Precompute rotation matrix for the angle
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    for r in range(inner_radius, outer_radius):
        t = (r - inner_radius) / (outer_radius - inner_radius)  # Normalized radius
        alpha = int(255 * (1 - t))  # Fade as we move outward

        # Gradual color transition
        r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
        g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
        b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

        # Generate points for the elliptical profile with extreme turbulence
        points = []
        for theta in np.linspace(0, 2 * np.pi, 300):  # 300 points for smooth shell

            # Sinusoidal deformity
            #sinusoidal_deformity = deformity * np.sin(3 * theta + t * np.pi)
            #sinusoidal_deformity = deformity * np.sin((1 + t) * theta + t * np.pi)
            random_frequency = np.random.uniform(2, 5)
            sinusoidal_deformity = deformity * np.sin(random_frequency * theta + t * np.pi)

            
            # Randomized noise-based deformity
            random_deformity = turbulence_intensity * (random.uniform(-1, 1))

            # Perlin noise for smooth variations (optional, requires `noise` library)
            perlin_deformity = turbulence_intensity * pnoise2(
                center[0] + r * np.cos(theta) * 0.01,
                center[1] + r * np.sin(theta) * 0.01,
                octaves=3,
            )

            # Combine all deformities
            noise = sinusoidal_deformity + random_deformity + perlin_deformity
            radius = r + noise

            # Elliptical deformation
            x_ellipse = radius * semi_major * np.cos(theta) / outer_radius
            y_ellipse = radius * semi_minor * np.sin(theta) / outer_radius

            # Rotate the points by the specified angle
            x_rotated = cos_theta * x_ellipse - sin_theta * y_ellipse
            y_rotated = sin_theta * x_ellipse + cos_theta * y_ellipse

            # Translate to the center
            x = int(center[0] + x_rotated)
            y = int(center[1] + y_rotated)
            points.append((x, y))

        # Draw polygonal shells
        draw.polygon(points, outline=(r_color, g_color, b_color, alpha))

    # Apply Gaussian blur to smooth out edges
    return img.filter(ImageFilter.GaussianBlur(blur_radius))




# Streamlit: Gaseous Shell Parameters
st.sidebar.header("Gaseous Shells")
num_shells = st.sidebar.slider("Number of Shells", 1, 5, 1)

shells = []
for i in range(num_shells):
    st.sidebar.subheader(f"Shell {i + 1}")
    semi_major = st.sidebar.slider(f"Semi-Major Axis (Shell {i + 1})", 10, 800, 200)
    semi_minor = st.sidebar.slider(f"Semi-Minor Axis (Shell {i + 1})", 10, 800, 150)
    angle = st.sidebar.slider(f"Inclination Angle (Shell {i + 1})", 0, 360, 45)
    inner_radius = st.sidebar.slider(f"Inner Radius (Shell {i + 1})", 10, 400, 100)
    outer_radius = st.sidebar.slider(f"Outer Radius (Shell {i + 1})", inner_radius, 500, inner_radius + 50)
    deformity = st.sidebar.slider(f"Deformity (Shell {i + 1})", 0.0, 20.0, 5.0)
    turbulence_intensity = st.sidebar.slider(f"Turbulence intensity (Shell {i + 1})", 0.0, 20.0, 5.0)
    blur_radius = st.sidebar.slider(f"Blur Radius (Shell {i + 1})", 1, 50, 10)
    start_color = st.sidebar.color_picker(f"Start Color (Shell {i + 1})", "#FF4500")
    end_color = st.sidebar.color_picker(f"End Color (Shell {i + 1})", "#0000FF")

    shells.append({
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "angle": angle,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "deformity": deformity,
        "turbulence_intensity": turbulence_intensity,
        "blur_radius": blur_radius,
        "start_color": tuple(int(start_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)),
        "end_color": tuple(int(end_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    })

# Generate gaseous shells and combine with the final image
gaseous_shells = Image.new("RGBA", image_size, (0, 0, 0, 0))
for shell in shells:
    shell_image = generate_gaseous_shells(
        image_size, center, shell["semi_major"], shell["semi_minor"], shell["angle"],
        shell["inner_radius"], shell["outer_radius"],
        shell["start_color"], shell["end_color"], shell["deformity"],shell["turbulence_intensity"], shell["blur_radius"]
    )
    gaseous_shells = Image.alpha_composite(gaseous_shells, shell_image)


#def generate_gaseous_shells(
#    image_size, center, semi_major, semi_minor, angle, inner_radius, outer_radius, start_color, end_color, deformity, turbulence_intensity, blur_radius
#)

# Combine with other layers
final_image = Image.alpha_composite(final_image, gaseous_shells)

# Visualizar la imagen generada inicialmente
st.image(final_image, caption="Nebula Simulation with Gaseous Elliptical Shells", use_column_width=True)

# Crear dos copias independientes de la imagen final
static_image = final_image.copy()  # Para la imagen fija
animation_image = final_image.copy()  # Para la animación



##############################################################################3

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import streamlit as st
from moviepy.editor import ImageSequenceClip

# Functions for gravitational lensing effects
def apply_weak_lensing(image, black_hole_center, schwarzschild_radius, lens_type="point"):
    img_array = np.array(image)
    height, width, channels = img_array.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center
    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)
    deflection = (
        schwarzschild_radius / r**2 if lens_type == "point" else schwarzschild_radius / (r + schwarzschild_radius)**2
    )
    new_x = x + deflection * dx / r
    new_y = y + deflection * dy / r
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    deformed_img_array = np.zeros_like(img_array)
    for channel in range(channels):
        deformed_img_array[..., channel] = map_coordinates(
            img_array[..., channel], [new_y.ravel(), new_x.ravel()], order=1, mode="constant", cval=0
        ).reshape((height, width))
    return Image.fromarray(deformed_img_array)

def apply_strong_lensing(image, black_hole_center, schwarzschild_radius, lens_type="point"):
    img_array = np.array(image)
    height, width, channels = img_array.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center
    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)
    deflection = (
        schwarzschild_radius**2 / r if lens_type == "point" else (schwarzschild_radius / (r + schwarzschild_radius))**2
    )
    new_x = x + deflection * dx / r
    new_y = y + deflection * dy / r
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    deformed_img_array = np.zeros_like(img_array)
    for channel in range(channels):
        deformed_img_array[..., channel] = map_coordinates(
            img_array[..., channel], [new_y.ravel(), new_x.ravel()], order=1, mode="constant", cval=0
        ).reshape((height, width))
    return Image.fromarray(deformed_img_array)

def apply_microlensing(image, lens_center, einstein_radius, source_type="point", source_radius=1):
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dx = x - lens_center[0]
    dy = y - lens_center[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)
    u = r / einstein_radius
    amplification = (
        (u**2 + 2) / (u * np.sqrt(u**2 + 4))
        if source_type == "point"
        else ((u + source_radius)**2 + 2) / ((u + source_radius) * np.sqrt((u + source_radius)**2 + 4))
    )
    amplification = np.clip(amplification, 1, 10)
    for channel in range(channels):
        img_array[..., channel] *= amplification
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def apply_kerr_lensing(image, black_hole_center, schwarzschild_radius, spin_parameter):
    img_array = np.array(image)
    height, width, channels = img_array.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center
    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)
    phi = np.arctan2(dy, dx) + spin_parameter * schwarzschild_radius / r
    deflection = schwarzschild_radius**2 / r
    new_x = x_center + r * np.cos(phi) + deflection * dx / r
    new_y = y_center + r * np.sin(phi) + deflection * dy / r
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    deformed_img_array = np.zeros_like(img_array)
    for channel in range(channels):
        deformed_img_array[..., channel] = map_coordinates(
            img_array[..., channel], [new_y.ravel(), new_x.ravel()], order=1, mode="constant", cval=0
        ).reshape((height, width))
    return Image.fromarray(deformed_img_array)

# Modify brightness and apply red/blue shift
def adjust_brightness(img_array, magnification):
    return np.clip(img_array * magnification[..., None], 0, 255)

#def apply_red_blue_shift(img_array, schwarzschild_radius, r):
#    shift_factor = np.sqrt(1 - 2 * schwarzschild_radius / r)
#    shift_factor = np.clip(shift_factor, 0.5, 1.5)
#    mask = np.any(img_array > 0, axis=-1)
#    img_array[mask, 0] *= shift_factor[mask]
#    img_array[mask, 2] /= shift_factor[mask]
#    return np.clip(img_array, 0, 255)

#def apply_red_blue_shift(img_array, schwarzschild_radius, r):
#    """
#    Aplica un corrimiento al rojo y azul a los píxeles según la distancia al centro.

#    Parameters:
#        img_array (numpy.ndarray): Imagen en formato array (alto, ancho, canales).
#        schwarzschild_radius (float): Radio de Schwarzschild.
#        r (numpy.ndarray): Distancia radial de cada píxel al centro del agujero negro.

#    Returns:
#        numpy.ndarray: Imagen modificada con corrimiento al rojo y azul.
#    """
    # Calcular el factor de corrimiento
    #shift_factor = np.sqrt(1 - 2 * schwarzschild_radius / r)
    #shift_factor = np.clip(shift_factor, 0.8, 1.2)  # Limitar rangos para evitar extremos

    # Crear una máscara para identificar píxeles válidos (que no sean completamente negros)
    #mask = np.any(img_array > 0, axis=-1)  # Excluir píxeles completamente negros

    # Aplicar corrimientos de color
    #img_array[mask, 0] = img_array[mask, 0] * shift_factor[mask]  # Corrimiento al rojo
    #img_array[mask, 2] = img_array[mask, 2] / shift_factor[mask]  # Corrimiento al azul

    # Ajustar el canal verde para mantener balance cromático
    #img_array[mask, 1] = img_array[mask, 1] * (0.5 + 0.5 * shift_factor[mask])

    # Limitar los valores dentro del rango válido [0, 255]
    #img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    #return img_array

def apply_red_blue_shift(img_array, schwarzschild_radius, r):
    """
    Aplica un corrimiento al rojo y azul a los píxeles según la distancia al centro.

    Parameters:
        img_array (numpy.ndarray): Imagen en formato array (alto, ancho, canales).
        schwarzschild_radius (float): Radio de Schwarzschild.
        r (numpy.ndarray): Distancia radial de cada píxel al centro del agujero negro.

    Returns:
        numpy.ndarray: Imagen modificada con corrimiento al rojo y azul.
    """
    # Calcular el factor de corrimiento
    shift_factor = 1 + (schwarzschild_radius / r**2)
    #shift_factor = np.sqrt(1 - 2 * schwarzschild_radius / r)
    shift_factor = np.clip(shift_factor, 0.9, 1.1)  # Reducir la intensidad del cambio

    # Crear una máscara para identificar píxeles válidos (que no sean completamente negros)
    mask = np.any(img_array > 0, axis=-1)  # Excluir píxeles completamente negros

    # Convertir img_array a float temporalmente para evitar problemas de casting
    img_array = img_array.astype(np.float32)

    # Aplicar corrimientos de color
    # Corrimiento al rojo: Intensificar ligeramente el canal rojo
    img_array[mask, 0] *= shift_factor[mask]  # Incrementar rojo proporcionalmente
    # Corrimiento al azul: Reducir proporcionalmente el azul
    img_array[mask, 2] /= shift_factor[mask]

    # Balancear el canal verde para mantener una transición visual suave
    img_array[mask, 1] *= 0.5 + 0.5 * shift_factor[mask]

    # Normalizar los canales para evitar dominancia de un solo color
    #total_intensity = img_array.sum(axis=-1, keepdims=True)
    #img_array = (img_array / total_intensity) * np.clip(total_intensity, 0, 255)

    # Limitar los valores dentro del rango válido [0, 255] y convertir de vuelta a uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    return img_array



# Video generation with MoviePy
def save_video_with_moviepy(frames, fps, output_path="animation.mp4"):
    frames_array = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(frames_array, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)
    return output_path

# Streamlit interface
st.title("Gravitational Lensing Simulation")

# Sidebar for lensing parameters
lensing_type = st.sidebar.selectbox(
    "Select Lensing Type",
    ["Weak Lensing", "Strong Lensing", "Microlensing", "Kerr Lensing"]
)
black_hole_x_fixed = st.sidebar.slider("Black Hole X Position (Static Image)", 0, 1600, 400)
black_hole_y_fixed = st.sidebar.slider("Black Hole Y Position (Static Image)", 0, 1600, 400)
schwarzschild_radius = st.sidebar.slider("Schwarzschild Radius (pixels)", 1, 300, 50)

lens_type = st.sidebar.selectbox("Lens Type (Weak/Strong Lensing)", ["point", "extended"])


# Aplicar efectos de lensing a la imagen fija
r_static = np.sqrt((np.arange(static_image.size[0]) - black_hole_x_fixed)**2 +
                    (np.arange(static_image.size[1])[:, None] - black_hole_y_fixed)**2)
r_static = np.maximum(r_static, 1e-5)
magnification_static = 1 + (schwarzschild_radius / r_static)
magnification_static = np.clip(magnification_static, 1, 10)

if lensing_type == "Weak Lensing":
    processed_image = apply_weak_lensing(static_image, (black_hole_x_fixed, black_hole_y_fixed), schwarzschild_radius, lens_type=lens_type)
elif lensing_type == "Strong Lensing":
    processed_image = apply_strong_lensing(static_image, (black_hole_x_fixed, black_hole_y_fixed), schwarzschild_radius, lens_type=lens_type)
elif lensing_type == "Microlensing":
    processed_image = apply_microlensing(static_image, (black_hole_x_fixed, black_hole_y_fixed), einstein_radius, source_type=source_type, source_radius=source_radius)
elif lensing_type == "Kerr Lensing":
    # Parámetros para Kerr Lensing
    spin_parameter = st.sidebar.slider("Black Hole Spin Parameter (a)", 0.0, 2.0, 0.5)
    processed_image = apply_kerr_lensing(static_image, (black_hole_x_fixed, black_hole_y_fixed), schwarzschild_radius, spin_parameter)

processed_image_array = np.array(processed_image)
processed_image_array = adjust_brightness(processed_image_array, magnification_static)
processed_image_array = apply_red_blue_shift(processed_image_array, schwarzschild_radius, r_static)
processed_image = Image.fromarray(processed_image_array.astype(np.uint8))

# Mostrar la imagen fija procesada
st.image(processed_image, caption=f"{lensing_type} Applied (Static Image)", use_column_width=True)


# Definir controles para animación en la barra lateral
num_frames = st.sidebar.slider("Number of Frames", 10, 100, 30)
fps = st.sidebar.slider("Frames Per Second", 1, 30, 10)
x_start = st.sidebar.slider("Animation Start X Position", 0, 800, 200)
y_start = st.sidebar.slider("Animation Start Y Position", 0, 800, 200)
x_end = st.sidebar.slider("Animation End X Position", 0, 800, 600)
y_end = st.sidebar.slider("Animation End Y Position", 0, 800, 600)


# Botón para generar animación
generate_animation = st.sidebar.button("Generate Animation")

if generate_animation:
    # Parámetros de la animación
    frames = []
    x_positions = np.linspace(x_start, x_end, num_frames)
    y_positions = np.linspace(y_start, y_end, num_frames)

    for i in range(num_frames):
        # Posición actual del agujero negro
        current_position = (x_positions[i], y_positions[i])

        # Calcular r dinámicamente para cada frame
        r_dynamic = np.sqrt(
            (np.arange(animation_image.size[0]) - current_position[0])**2 +
            (np.arange(animation_image.size[1])[:, None] - current_position[1])**2
        )
        r_dynamic = np.maximum(r_dynamic, 1e-5)  # Evitar divisiones por cero

        # Aplicar efecto de lensing correspondiente
        frame_image = np.array(
            apply_kerr_lensing(animation_image, current_position, schwarzschild_radius, spin_parameter)
            if lensing_type == "Kerr Lensing"
            else apply_weak_lensing(animation_image, current_position, schwarzschild_radius, lens_type=lens_type)
            if lensing_type == "Weak Lensing"
            else apply_strong_lensing(animation_image, current_position, schwarzschild_radius, lens_type=lens_type)
            if lensing_type == "Strong Lensing"
            else apply_microlensing(animation_image, current_position, einstein_radius, source_type=source_type, source_radius=source_radius)
        )

        # Aplicar brillo y corrimiento dinámicos
        magnification_dynamic = 1 + (schwarzschild_radius / r_dynamic)
        magnification_dynamic = np.clip(magnification_dynamic, 1, 10)

        frame_image = adjust_brightness(frame_image, magnification_dynamic)
        frame_image = apply_red_blue_shift(frame_image, schwarzschild_radius, r_dynamic)

        # Agregar frame procesado a la lista
        frames.append(Image.fromarray(frame_image.astype(np.uint8)))

    # Guardar y mostrar la animación
    video_path = save_video_with_moviepy(frames, fps)
    st.video(video_path)

    # Botón para descargar el video
    with open(video_path, "rb") as video_file:
        st.download_button(
            label="Download Video",
            data=video_file,
            file_name="black_hole_animation.mp4",
            mime="video/mp4"
        )
else:
    st.write("Adjust the parameters and press 'Generate Animation' to create the video.")


################################









#import numpy as np
#from PIL import Image, ImageDraw
#import streamlit as st


#def create_photon_ring(image_size, shadow_radius, ring_width):
#    """
#    Create a photon ring image using Pillow.
#    """
#    img = Image.new("RGBA", image_size, (0, 0, 0, 0))  # Transparent background
#    draw = ImageDraw.Draw(img)
#    center = (image_size[0] // 2, image_size[1] // 2)

#    for r in range(shadow_radius, shadow_radius + ring_width):
#        intensity = int(255 * (1 - (r - shadow_radius) / ring_width))  # Gradient intensity
#        color = (255, 140, 0, intensity)  # Orange-like color with gradient alpha
#        draw.ellipse(
#            [center[0] - r, center[1] - r, center[0] + r, center[1] + r],
#            outline=color,
#            width=1
#        )

#    return img


# Streamlit UI
#st.title("Black Hole Shadow and Photon Ring")

# Parameters for the photon ring
#image_size = (800, 800)  # Size must match the nebulosa and field of stars
#shadow_radius = st.sidebar.slider("Shadow Radius", 50, 300, 150)
#ring_width = st.sidebar.slider("Ring Width", 10, 100, 30)

# Generate the photon ring (shadow and ring)
#photon_ring = create_photon_ring(image_size, shadow_radius, ring_width)

# Simulate loading the previously generated image of nebulosa and stars (final_image)
# Replace this with your actual `final_image`
#final_image = Image.new("RGBA", image_size, (0, 0, 0, 0))  # Placeholder for nebulosa image
# Load or generate the nebulosa image here (e.g., from your previous process)
# Example:
# final_image = Image.open("nebula_stars.png").convert("RGBA")

# Combine the photon ring with the nebulosa and star field
#combined_image = Image.alpha_composite(final_image, photon_ring)

# Display the final combined image
#st.image(combined_image, caption="Black Hole Shadow with Nebulosa and Photon Ring", use_column_width=True)

