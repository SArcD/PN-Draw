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

# Function to draw gaseous shells with angle control
def draw_gaseous_shells(image_size, shells):
    """
    Draws diffuse gaseous shells on an image.
    
    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        shells: List[Dict] - List of dictionaries defining shell properties:
            - "center": Tuple[int, int] - Center of the shell.
            - "semimajor_axis": int - Semimajor axis (or radius for circular shells).
            - "semiminor_axis": int - Semiminor axis.
            - "angle": float - Rotation angle of the shell in degrees.
            - "color": str - Shell color (e.g., "#FF0000").
            - "thickness": int - Edge thickness.
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

        # Draw concentric transparent ellipses for a diffuse effect
        for t in range(thickness, 0, -1):
            alpha = int(255 * (t / thickness) ** 2)  # Decreasing alpha
            
            # Create bounding box with rotation
            ellipse_bbox = (
                center[0] - a - t,
                center[1] - b - t,
                center[0] + a + t,
                center[1] + b + t,
            )
            
            # Rotate the ellipse
            rotated_ellipse = Image.new("RGBA", image_size, (0, 0, 0, 0))
            rotated_draw = ImageDraw.Draw(rotated_ellipse)
            rotated_draw.ellipse(ellipse_bbox, outline=color + (alpha,), width=1)
            rotated_ellipse = rotated_ellipse.rotate(angle, center=center)
            
            img = Image.alpha_composite(img, rotated_ellipse)
    
    return img



def draw_gaseous_shells_with_noise_and_blur(image_size, shells, noise_intensity=3, blur_radius=5):
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


#######################################

# Sidebar inputs for gaseous shells
st.sidebar.markdown("### Gaseous Shells")
num_shells = st.sidebar.slider("Number of Shells", min_value=1, max_value=5, value=2)

# Checkbox for enabling noise and blur
use_noise_and_blur = st.sidebar.checkbox("Enable Noise and Gaussian Blur", value=True)

# If noise and blur are enabled, add sliders for their intensities
if use_noise_and_blur:
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

# Decide which function to use for drawing shells
if use_noise_and_blur:
    # Generate gaseous shells with noise and blur
    gaseous_shells = draw_gaseous_shells_with_noise_and_blur(
        image_size,
        shells,
        noise_intensity=noise_intensity,
        blur_radius=blur_radius,
    )
else:
    # Generate standard gaseous shells without noise and blur
    gaseous_shells = draw_gaseous_shells(image_size, shells)

# Combine gaseous shells with the existing image
final_image_with_shells = Image.alpha_composite(final_image, gaseous_shells)

# Display the updated image
st.image(final_image_with_shells, use_column_width=True)

#################################


# Function to add fractal noise to specific shells
def add_fractal_noise_to_shells(image, shells, noise_intensity=5, blur_radius=5, noise_area="All Shells"):
    """
    Adds fractal noise to enhance the diffuse and irregular nature of shells.
    
    Parameters:
        image: PIL.Image - Base image to add noise to.
        shells: List[Dict] - Shell definitions to restrict noise application.
        noise_intensity: int - Intensity of fractal noise.
        blur_radius: int - Radius for Gaussian blur.
        noise_area: str - Area where noise will be applied: "All Shells" or "Specific Shell".
    
    Returns:
        PIL.Image - Image with enhanced noise details.
    """
    noise_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(noise_layer)

    if noise_area == "All Shells":
        # Apply noise across all shells
        for shell in shells:
            center = shell["center"]
            a = shell["semimajor_axis"]
            b = shell["semiminor_axis"]
            
            for _ in range(noise_intensity):
                # Random perturbations within shell bounds
                offset_x = np.random.randint(-a // 2, a // 2)
                offset_y = np.random.randint(-b // 2, b // 2)
                noise_x = center[0] + offset_x
                noise_y = center[1] + offset_y
                width = np.random.randint(10, a // 5)
                height = np.random.randint(10, b // 5)
                alpha = np.random.randint(20, 80)  # Semi-transparent noise

                bbox = (noise_x - width, noise_y - height, noise_x + width, noise_y + height)
                draw.ellipse(bbox, fill=(255, 255, 255, alpha))  # Noise is white initially
    elif noise_area == "Specific Shell":
        # Apply noise only to the first shell for demonstration
        if len(shells) > 0:
            shell = shells[0]  # Take the first shell
            center = shell["center"]
            a = shell["semimajor_axis"]
            b = shell["semiminor_axis"]
            
            for _ in range(noise_intensity):
                offset_x = np.random.randint(-a // 2, a // 2)
                offset_y = np.random.randint(-b // 2, b // 2)
                noise_x = center[0] + offset_x
                noise_y = center[1] + offset_y
                width = np.random.randint(10, a // 5)
                height = np.random.randint(10, b // 5)
                alpha = np.random.randint(20, 80)

                bbox = (noise_x - width, noise_y - height, noise_x + width, noise_y + height)
                draw.ellipse(bbox, fill=(255, 255, 255, alpha))

    # Apply Gaussian blur
    noise_layer = noise_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Composite the noise layer with the base image
    blended_image = Image.alpha_composite(image, noise_layer)

    return blended_image

##########################################################
# New section for controlling fractal noise in Streamlit

# Sidebar controls for fractal noise
st.sidebar.markdown("### Fractal Noise")
apply_noise = st.sidebar.checkbox("Enable Fractal Noise", value=False)

if apply_noise:
    noise_intensity = st.sidebar.slider("Noise Intensity", min_value=1, max_value=20, value=5)
    blur_radius = st.sidebar.slider("Gaussian Blur Radius", min_value=1, max_value=20, value=5)
    noise_area = st.sidebar.radio("Noise Area", ["All Shells", "Specific Shell"], index=0)
    
    # Apply fractal noise to the shells
    final_image_with_noise = add_fractal_noise_to_shells(
        final_image_with_shells, 
        shells, 
        noise_intensity=noise_intensity, 
        blur_radius=blur_radius, 
        noise_area=noise_area
    )
    
    # Display the updated image
    st.image(final_image_with_noise, use_column_width=True)
else:
    # Display the image without additional noise
    st.image(final_image_with_shells, use_column_width=True)



