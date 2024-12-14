import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageColor


# Function to generate a star field as a PIL image
def generate_star_field(num_stars, image_size):
    img = Image.new("RGBA", image_size, "black")
    draw = ImageDraw.Draw(img)
    for _ in range(num_stars):
        x = np.random.randint(0, image_size[0])
        y = np.random.randint(0, image_size[1])
        size = np.random.randint(1, 4)
        color = np.random.choice(["white", "lightblue", "yellow", "red"])
        draw.ellipse((x, y, x + size, y + size), fill=color)
    return img

# Function to draw a central star with a diffuse halo
def draw_central_star(image_size, position, star_size, halo_size, color):
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw halo
    halo = Image.new("RGBA", (halo_size * 2, halo_size * 2), (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.ellipse((0, 0, halo_size * 2, halo_size * 2), fill=color + (50,))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=halo_size / 2))
    img.paste(halo, (position[0] - halo_size, position[1] - halo_size), halo)
    
    # Draw star
    draw.ellipse(
        (position[0] - star_size, position[1] - star_size, position[0] + star_size, position[1] + star_size),
        fill=color + (255,),
    )
    
    return img
#########################################################


# Función para dibujar una estrella central con un halo difuso
def draw_central_star(image_size, position, star_size, halo_size, color):
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Dibujar halo
    halo = Image.new("RGBA", (halo_size * 2, halo_size * 2), (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.ellipse((0, 0, halo_size * 2, halo_size * 2), fill=color + (50,))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=halo_size / 2))
    img.paste(halo, (position[0] - halo_size, position[1] - halo_size), halo)
    
    # Dibujar estrella
    draw.ellipse(
        (position[0] - star_size, position[1] - star_size, position[0] + star_size, position[1] + star_size),
        fill=color + (255,),
    )
    
    return img

# Función para añadir un efecto de brillo alrededor de la estrella central
def add_glow_effect(image, position, glow_radius, glow_color):
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    for i in range(glow_radius, 0, -1):
        alpha = int(255 * (i / glow_radius) ** 2)
        draw.ellipse(
            (
                position[0] - i,
                position[1] - i,
                position[0] + i,
                position[1] + i,
            ),
            fill=glow_color + (alpha,),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=glow_radius / 2))
    return Image.alpha_composite(image, glow)

# Interfaz de Streamlit
st.title("Simulación de Nebulosa")

# Controles en la barra lateral
num_stars = st.sidebar.slider("Número de Estrellas", min_value=100, max_value=2000, value=500, step=100)
star_x = st.sidebar.slider("Posición X de la Estrella Central", min_value=0, max_value=800, value=400)
star_y = st.sidebar.slider("Posición Y de la Estrella Central", min_value=0, max_value=800, value=400)
star_size = st.sidebar.slider("Tamaño de la Estrella Central", min_value=5, max_value=50, value=20)
halo_size = st.sidebar.slider("Tamaño del Halo", min_value=10, max_value=100, value=50)
star_color = st.sidebar.color_picker("Color de la Estrella Central", "#FFFFFF")
glow_radius = st.sidebar.slider("Radio del Brillo", min_value=10, max_value=200, value=100)
glow_color = st.sidebar.color_picker("Color del Brillo", "#FFFF00")

# Generar imágenes
image_size = (800, 800)
star_field = generate_star_field(num_stars, image_size)
central_star = draw_central_star(image_size, (star_x, star_y), star_size, halo_size, ImageColor.getrgb(star_color))

# Combinar imágenes
combined_image = Image.alpha_composite(star_field, central_star)

# Añadir efecto de brillo
final_image = add_glow_effect(combined_image, (star_x, star_y), glow_radius, ImageColor.getrgb(glow_color))

# Mostrar imagen
#st.image(final_image, use_column_width=True)



############################################################

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def draw_gaseous_shells(image_size, shells, noise_intensity=3, blur_radius=5):
    """
    Draws diffuse gaseous shells on an image with irregular edges and blur effects.
    
    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        shells: List[Dict] - List of dictionaries defining shell properties:
            - "center": Tuple[int, int] - Center of the shell.
            - "semimajor_axis": int - Semimajor axis (or radius for circular shells).
            - "semiminor_axis": int - Semiminor axis.
            - "angle": float - Rotation angle of the shell in degrees.
            - "color": str - Shell color (e.g., "#FF0000").
            - "thickness": int - Edge thickness.
        noise_intensity: int - Level of edge irregularity.
        blur_radius: int - Radius for Gaussian blur.
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    for shell in shells:
        center = shell["center"]
        a = shell["semimajor_axis"]
        b = shell["semiminor_axis"]
        angle = shell["angle"]
        color = ImageColor.getrgb(shell["color"])
        thickness = shell["thickness"]

        # Create a separate layer for the shell
        shell_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
        shell_draw = ImageDraw.Draw(shell_layer)

        # Draw concentric noisy ellipses for a diffuse effect
        for t in range(thickness, 0, -1):
            alpha = int(255 * (t / thickness) ** 2)  # Decreasing alpha for diffuse edges
            
            # Generate random perturbation for irregular edges
            offset_x = np.random.uniform(-noise_intensity, noise_intensity)
            offset_y = np.random.uniform(-noise_intensity, noise_intensity)
            
            ellipse_bbox = (
                center[0] - a - t + offset_x,
                center[1] - b - t + offset_y,
                center[0] + a + t + offset_x,
                center[1] + b + t + offset_y,
            )
            
            shell_draw.ellipse(ellipse_bbox, outline=color + (alpha,), width=1)

        # Rotate the shell layer
        rotated_shell = shell_layer.rotate(angle, center=center)

        # Apply Gaussian blur to the shell
        blurred_shell = rotated_shell.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Composite the blurred shell onto the main image
        img = Image.alpha_composite(img, blurred_shell)
    
    return img

#########################################################
# New section for gaseous shells in Streamlit

# Sidebar inputs for gaseous shells
#st.sidebar.markdown("### Gaseous Shells")
#num_shells = st.sidebar.slider("Number of Shells", min_value=1, max_value=5, value=2)
#shells = []
#for i in range(num_shells):
#    st.sidebar.markdown(f"#### Shell {i+1}")
#    center_x = st.sidebar.slider(f"Shell {i+1} Center X", 0, image_size[0], 400)
#    center_y = st.sidebar.slider(f"Shell {i+1} Center Y", 0, image_size[1], 400)
#    semimajor_axis = st.sidebar.slider(f"Shell {i+1} Semimajor Axis", 50, 400, 200)
#    semiminor_axis = st.sidebar.slider(f"Shell {i+1} Semiminor Axis", 50, 400, 200)
#    angle = st.sidebar.slider(f"Shell {i+1} Angle", 0, 360, 0)  # Angle in degrees
#    shell_color = st.sidebar.color_picker(f"Shell {i+1} Color", "#00FFFF")
#    thickness = st.sidebar.slider(f"Shell {i+1} Thickness", 1, 150, 20)
    
#    shells.append({
#        "center": (center_x, center_y),
#        "semimajor_axis": semimajor_axis,
#        "semiminor_axis": semiminor_axis,
#        "angle": angle,
#        "color": shell_color,
#        "thickness": thickness,
#    })

# Generate gaseous shells
#gaseous_shells = draw_gaseous_shells(image_size, shells)

# Combine gaseous shells with the existing image
#final_image_with_shells = Image.alpha_composite(final_image, gaseous_shells)

# Display the updated image
#st.image(final_image_with_shells, use_column_width=True)

# Sidebar inputs for gaseous shells
st.sidebar.markdown("### Gaseous Shells")
num_shells = st.sidebar.slider("Number of Shells", min_value=1, max_value=5, value=2)

# Sliders for noise intensity and Gaussian blur
noise_intensity = st.sidebar.slider("Noise Intensity", min_value=1, max_value=10, value=3)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", min_value=1, max_value=20, value=5)

# Gather shell parameters
shells = []
for i in range(num_shells):
    st.sidebar.markdown(f"#### Shell {i+1}")
    center_x = st.sidebar.slider(f"Shell {i+1} Center X", 0, image_size[0], 400)
    center_y = st.sidebar.slider(f"Shell {i+1} Center Y", 0, image_size[1], 400)
    semimajor_axis = st.sidebar.slider(f"Shell {i+1} Semimajor Axis", 50, 400, 200)
    semiminor_axis = st.sidebar.slider(f"Shell {i+1} Semiminor Axis", 50, 400, 200)
    angle = st.sidebar.slider(f"Shell {i+1} Angle", 0, 360, 0)  # Angle in degrees
    shell_color = st.sidebar.color_picker(f"Shell {i+1} Color", "#00FFFF")
    thickness = st.sidebar.slider(f"Shell {i+1} Thickness", 1, 150, 20)
    
    shells.append({
        "center": (center_x, center_y),
        "semimajor_axis": semimajor_axis,
        "semiminor_axis": semiminor_axis,
        "angle": angle,
        "color": shell_color,
        "thickness": thickness,
    })

# Generate gaseous shells with noise and blur
gaseous_shells = draw_gaseous_shells(image_size, shells, noise_intensity=noise_intensity, blur_radius=blur_radius)

# Combine gaseous shells with the existing image
final_image_with_shells = Image.alpha_composite(final_image, gaseous_shells)

# Display the updated image
st.image(final_image_with_shells, use_column_width=True)



#######################################

import numpy as np
from PIL import ImageDraw, ImageFilter
import streamlit as st

# Sidebar controls for noise and blur
noise_intensity = st.sidebar.slider("Noise Intensity", min_value=1, max_value=10, value=3, step=1)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", min_value=1, max_value=20, value=5, step=1)

# Updated function to generate a noise layer with adjustable noise intensity and blur
def generate_noise_layer(image_size, semi_major, semi_minor, center, angle, shell_color, thickness, noise_level, blur_level):
    """
    Generate a noise layer for a gaseous shell with adjustable noise intensity and Gaussian blur.
    """
    noise_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(noise_layer)

    for t in range(thickness):
        # Add random perturbation for irregular edges based on noise intensity
        offset_x = np.random.uniform(-noise_level, noise_level)
        offset_y = np.random.uniform(-noise_level, noise_level)

        left = center[0] - semi_major + offset_x - t
        top = center[1] - semi_minor + offset_y - t
        right = center[0] + semi_major + offset_x + t
        bottom = center[1] + semi_minor + offset_y + t

        draw.ellipse(
            (left, top, right, bottom),
            outline=shell_color + (int(200 / (t + 1)),),  # Enhanced opacity gradient for visibility
            width=1
        )

    # Apply Gaussian blur to simulate diffusion with adjustable blur radius
    noise_layer = noise_layer.filter(ImageFilter.GaussianBlur(radius=blur_level))
    return noise_layer

# Updated function to composite all shells with noise and blur sliders
def draw_gaseous_shells(image, shells, noise_level, blur_level):
    """
    Draw multiple gaseous shells as independent noise layers and composite them onto the image.
    """
    composite_image = image.copy()  # Start with the existing image

    for i, shell in enumerate(shells):
        semi_major, semi_minor, angle, shell_color, thickness = shell

        # Generate the noise layer for this shell with noise and blur settings
        noise_layer = generate_noise_layer(
            image_size=image.size,
            semi_major=semi_major,
            semi_minor=semi_minor,
            center=(400, 400),  # Center of the shell
            angle=angle,
            shell_color=ImageColor.getrgb(shell_color),
            thickness=thickness,
            noise_level=noise_level,
            blur_level=blur_level
        )

        # Composite the noise layer onto the current image
        composite_image = Image.alpha_composite(composite_image, noise_layer)

    return composite_image

# Define shells with adjustable parameters
shells = [
    (150, 120, 0, "#00FFFF", 30),  # Enhanced thickness for better visibility
    (200, 170, 30, "#00FF00", 35),
]

# Draw shells on top of the existing star field
final_image_with_shells = draw_gaseous_shells(final_image, shells, noise_level=noise_intensity, blur_level=blur_radius)

# Display the updated composite image with shells
st.image(final_image_with_shells, use_column_width=True)



