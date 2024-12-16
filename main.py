import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageColor
from noise import pnoise2


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


# Function to draw a central star with a diffuse halo and radial filaments
def draw_central_star_with_filaments(image_size, position, star_size, halo_size, color, num_filaments, dispersion, blur_radius):
    """
    Draws a central star with a diffuse halo and radial filaments to simulate light dispersion.

    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        position: Tuple[int, int] - Position of the central star.
        star_size: int - Size of the star.
        halo_size: int - Size of the halo.
        color: Tuple[int, int, int] - Color of the star.
        num_filaments: int - Number of radial filaments.
        dispersion: int - Degree of dispersion for the radial filaments.
        blur_radius: int - Gaussian blur radius for light diffusion.
    """
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

    # Draw radial filaments
    filament_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
    filament_draw = ImageDraw.Draw(filament_layer)
    for i in range(num_filaments):
        angle = 2 * np.pi * i / num_filaments
        end_x = position[0] + (halo_size + np.random.uniform(-dispersion, dispersion)) * np.cos(angle)
        end_y = position[1] + (halo_size + np.random.uniform(-dispersion, dispersion)) * np.sin(angle)

        filament_draw.line(
            [(position[0], position[1]), (end_x, end_y)],
            fill=color + (100,),  # Semi-transparent
            width=2,
        )

    # Apply Gaussian blur to the filaments for a diffuse effect
    filament_layer = filament_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Combine the filaments with the star image
    img = Image.alpha_composite(img, filament_layer)

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
image_size = (1200, 1200)
star_field = generate_star_field(num_stars, image_size)
#central_star = draw_central_star(image_size, (star_x, star_y), star_size, halo_size, ImageColor.getrgb(star_color))
# Sidebar inputs for the central star with radial filaments
st.sidebar.markdown("### Central Star with Radial Filaments")
num_filaments = st.sidebar.slider("Number of Filaments", min_value=10, max_value=100, value=30, step=5)
filament_dispersion = st.sidebar.slider("Filament Dispersion", min_value=1, max_value=50, value=10, step=1)
filament_blur_radius = st.sidebar.slider("Filament Blur Radius", min_value=1, max_value=20, value=5, step=1)

# Generate the central star with radial filaments
central_star = draw_central_star_with_filaments(
    image_size=image_size,
    position=(star_x, star_y),
    star_size=star_size,
    halo_size=halo_size,
    color=ImageColor.getrgb(star_color),
    num_filaments=num_filaments,
    dispersion=filament_dispersion,
    blur_radius=filament_blur_radius,
)

# Combine the central star with the existing image
final_image_with_star_and_filaments = Image.alpha_composite(star_field, central_star)

# Display the final image
st.image(final_image_with_star_and_filaments, use_column_width=True)


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
apply_noise = st.sidebar.checkbox("Enabler Fractal Noise", value=False)

if apply_noise:
    noise_intensity = st.sidebar.slider("Noise Intensity", min_value=1, max_value=20, value=5)
    blur_radius = st.sidebar.slider("Blur Radius", min_value=1, max_value=20, value=5)
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

from PIL import Image, ImageFilter, ImageDraw, ImageColor
import numpy as np


# Function to add radiating filaments with noise and Gaussian diffusion
def add_radiating_filaments_with_noise(
    image, center, filament_length, num_filaments, filament_width, filament_color, noise_intensity, blur_radius
):
    """
    Adds radiating filaments with noise and Gaussian diffusion.
    
    Parameters:
        image: PIL.Image - The base image to add filaments to.
        center: Tuple[int, int] - The center point from which filaments radiate.
        filament_length: int - Maximum length of the filaments.
        num_filaments: int - Number of filaments to draw.
        filament_width: int - Width of each filament.
        filament_color: str - Color of the filaments (e.g., "#00FF00").
        noise_intensity: int - Intensity of random noise for the filaments.
        blur_radius: int - Radius for Gaussian blur to diffuse the filaments.
    
    Returns:
        PIL.Image - Image with added filaments.
    """
    # Create a blank layer for the filaments
    filament_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(filament_layer)

    for _ in range(num_filaments):
        # Random angle for the filament
        angle = np.random.uniform(0, 2 * np.pi)
        # Random length within the defined range
        length = np.random.uniform(filament_length * 0.5, filament_length)

        # Calculate end point using trigonometry
        end_x = center[0] + int(length * np.cos(angle))
        end_y = center[1] + int(length * np.sin(angle))

        # Add noise to the end point
        noise_x = np.random.uniform(-noise_intensity, noise_intensity)
        noise_y = np.random.uniform(-noise_intensity, noise_intensity)

        # Apply the noise to the end point
        end_x += int(noise_x)
        end_y += int(noise_y)

        # Random opacity for the filament
        opacity = np.random.randint(100, 200)
        color_with_opacity = ImageColor.getrgb(filament_color) + (opacity,)

        # Draw the filament as a line
        draw.line(
            [center, (end_x, end_y)],
            fill=color_with_opacity,
            width=filament_width,
        )

    # Apply Gaussian blur to the filament layer
    filament_layer = filament_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Composite the filament layer onto the base image
    image_with_filaments = Image.alpha_composite(image, filament_layer)

    return image_with_filaments


######################################################
# New section for radiating filaments with Streamlit

# Sidebar controls for radiating filaments
st.sidebar.markdown("### Radiating Filaments with Gas Effect")
enable_filaments_with_gas = st.sidebar.checkbox("Enable Radiating Filaments with Gas Effect", value=False)

if enable_filaments_with_gas:
    st.sidebar.markdown("#### Filament Settings")
    filament_length = st.sidebar.slider("Filament Length", min_value=10, max_value=300, value=150)
    num_filaments = st.sidebar.slider("Number of Filaments", min_value=5, max_value=500, value=100)
    filament_width = st.sidebar.slider("Filament Width", min_value=1, max_value=10, value=2)
    filament_color = st.sidebar.color_picker("Filament Color", "#00FF00")
    
    st.sidebar.markdown("#### Gas Effect Settings")
    noise_intensity = st.sidebar.slider("Ne Intensity", min_value=1, max_value=50, value=10)
    blur_radius = st.sidebar.slider("Gauss Blur Radius", min_value=1, max_value=20, value=5)

    # Add radiating filaments with noise and gas effect
    final_image_with_filaments_gas = add_radiating_filaments_with_noise(
        final_image_with_shells.copy(),  # Start with the existing image
        center=(400, 400),  # Central position (e.g., same as the star or user-defined)
        filament_length=filament_length,
        num_filaments=num_filaments,
        filament_width=filament_width,
        filament_color=filament_color,
        noise_intensity=noise_intensity,
        blur_radius=blur_radius,
    )
    
    # Display the updated image
    st.image(final_image_with_filaments_gas, use_column_width=True)
else:
    # Display the image without filaments
    st.image(final_image_with_shells, use_column_width=True)


def draw_filaments_between_selected_shells(image_size, origin_shell, target_shell, num_filaments, filament_color, noise_intensity, blur_radius):
    """
    Draws radiating filaments between the specified origin and target shells with noise and Gaussian diffusion.
    
    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        origin_shell: Dict - Properties of the origin shell.
        target_shell: Dict - Properties of the target shell.
        num_filaments: int - Total number of filaments to draw.
        filament_color: Tuple[int, int, int] - Color of the filaments.
        noise_intensity: int - Intensity of random noise added to the filaments.
        blur_radius: int - Radius of the Gaussian blur applied to the filaments.
    """
    filaments_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(filaments_layer)

    center1 = origin_shell["center"]
    center2 = target_shell["center"]
    a1, b1 = origin_shell["semimajor_axis"], origin_shell["semiminor_axis"]
    a2, b2 = target_shell["semimajor_axis"], target_shell["semiminor_axis"]
    angle1 = np.radians(origin_shell["angle"])
    angle2 = np.radians(target_shell["angle"])

    for _ in range(num_filaments):
        # Randomly sample points on the boundary of the origin shell
        theta1 = np.random.uniform(0, 2 * np.pi)
        x1 = center1[0] + a1 * np.cos(theta1) * np.cos(angle1) - b1 * np.sin(theta1) * np.sin(angle1)
        y1 = center1[1] + a1 * np.cos(theta1) * np.sin(angle1) + b1 * np.sin(theta1) * np.cos(angle1)

        # Randomly sample points on the boundary of the target shell
        theta2 = np.random.uniform(0, 2 * np.pi)
        x2 = center2[0] + a2 * np.cos(theta2) * np.cos(angle2) - b2 * np.sin(theta2) * np.sin(angle2)
        y2 = center2[1] + a2 * np.cos(theta2) * np.sin(angle2) + b2 * np.sin(theta2) * np.cos(angle2)

        # Add random noise to create irregular filaments
        num_points = 10  # Number of intermediate points along the filament
        points = [
            (
                x1 + i * (x2 - x1) / num_points + np.random.uniform(-noise_intensity, noise_intensity),
                y1 + i * (y2 - y1) / num_points + np.random.uniform(-noise_intensity, noise_intensity),
            )
            for i in range(num_points)
        ]

        # Draw the filament as a line connecting the points
        draw.line(points, fill=filament_color, width=2)

    # Apply Gaussian blur for a diffuse appearance
    filaments_layer = filaments_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    return filaments_layer

# Sidebar inputs for inter-shell filaments
st.sidebar.markdown("### Filaments Between Selected Shells")
activate_filaments = st.sidebar.checkbox("Activate Filaments", value=True)
if activate_filaments:
    # Input box to specify the origin and target shell indices
    origin_shell_index = st.sidebar.number_input("Origin Shell Index", min_value=0, max_value=len(shells)-1, value=0)
    target_shell_index = st.sidebar.number_input("Target Shell Index", min_value=0, max_value=len(shells)-1, value=len(shells)-1)

    # Other filament controls
    num_filaments = st.sidebar.slider("Number of Filaments", min_value=10, max_value=200, value=50)
    filament_color = st.sidebar.color_picker("Filament Color", "#00FFFF")
    filament_noise_intensity = st.sidebar.slider("Filament Noise Intensity", min_value=1, max_value=20, value=5)
    filament_blur_radius = st.sidebar.slider("Filament Blur Radius", min_value=1, max_value=20, value=5)

    # Validate shell indices
    if origin_shell_index < len(shells) and target_shell_index < len(shells):
        origin_shell = shells[origin_shell_index]
        target_shell = shells[target_shell_index]

        # Draw filaments between the selected shells
        filaments_layer = draw_filaments_between_selected_shells(
            image_size=image_size,
            origin_shell=origin_shell,
            target_shell=target_shell,
            num_filaments=num_filaments,
            filament_color=ImageColor.getrgb(filament_color),
            noise_intensity=filament_noise_intensity,
            blur_radius=filament_blur_radius,
        )

        # Composite the filaments with the existing image
        final_image_with_filaments = Image.alpha_composite(final_image_with_shells, filaments_layer)

        # Display the updated image
        st.image(final_image_with_filaments, use_column_width=True)
        #st.image(final_image_with_filaments_gas, use_column_width=True)

    else:
        st.warning("Please ensure the origin and target shell indices are valid.")
else:
    st.image(final_image_with_shells, use_column_width=True)



########################################################

def darken_shell_sections_with_thickness(image, shells, darkened_sections):
    """
    Darkens specified sections of the shells based on the user-defined intervals,
    considering the thickness of the shell.
    
    Parameters:
        image: PIL.Image - The existing image with shells.
        shells: List[Dict] - List of dictionaries defining shell properties:
            - "center": Tuple[int, int] - Center of the shell.
            - "semimajor_axis": int - Semimajor axis.
            - "semiminor_axis": int - Semiminor axis.
            - "angle": float - Rotation angle in degrees.
            - "color": str - Shell color.
            - "thickness": int - Shell thickness.
        darkened_sections: Dict[int, List[Tuple[int, int]]] - Mapping of shell index to intervals of angles (in degrees) to be darkened.

    Returns:
        PIL.Image - Modified image with darkened sections.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for shell_index, shell in enumerate(shells):
        if shell_index in darkened_sections:
            center = shell["center"]
            a = shell["semimajor_axis"]
            b = shell["semiminor_axis"]
            thickness = shell["thickness"]

            # Extract the darkened sections for this shell
            intervals = darkened_sections[shell_index]

            # Draw darkened sections for each layer of thickness
            for t in range(thickness):
                for interval in intervals:
                    start_angle, end_angle = interval
                    bbox = (
                        center[0] - a - t,
                        center[1] - b - t,
                        center[0] + a + t,
                        center[1] + b + t,
                    )
                    draw.arc(
                        bbox,
                        start=start_angle,
                        end=end_angle,
                        fill=(0, 0, 0, int(255 / (t + 1))),  # Adjust transparency based on thickness layer
                        width=1,
                    )

    return img

# Sidebar section for darkening shell sections
st.sidebar.markdown("### Darken Shell Sections")
darkened_sections = {}
for i in range(num_shells):
    st.sidebar.markdown(f"#### Shell {i+1}")
    num_intervals = st.sidebar.slider(f"Number of Darkened Sections (Shell {i+1})", 0, 10, 0)
    intervals = []
    for j in range(num_intervals):
        start_angle = st.sidebar.slider(f"Shell {i+1} Section {j+1} Start Angle", 0, 360, j * 30)
        end_angle = st.sidebar.slider(f"Shell {i+1} Section {j+1} End Angle", 0, 360, j * 30 + 15)
        intervals.append((start_angle, end_angle))
    darkened_sections[i] = intervals

# Apply the darkening effect to the shells
final_image_with_darkened_sections = darken_shell_sections_with_thickness(
    final_image_with_shells, 
    shells, 
    darkened_sections
)

# Display the updated image with darkened sections
st.image(final_image_with_darkened_sections, use_column_width=True)


#################################################################3

from PIL import Image, ImageDraw
from noise import pnoise2

def generate_perlin_texture(image_size, scale=50):
    """
    Generate a Perlin noise texture as a PIL image.
    
    Parameters:
        image_size: Tuple[int, int] - Size of the image (width, height).
        scale: int - Scale of the Perlin noise.

    Returns:
        PIL.Image - Perlin noise texture.
    """
    texture = Image.new("RGBA", image_size, (0, 0, 0, 0))
    pixels = texture.load()

    for x in range(image_size[0]):
        for y in range(image_size[1]):
            # Generate Perlin noise value
            value = int((pnoise2(x / scale, y / scale, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=image_size[0], repeaty=image_size[1], base=42) + 1) * 128)
            pixels[x, y] = (value, value, value, int(value * 0.5))  # Grayscale + alpha

    return texture



# Function to create advanced gaseous shell textures
def generate_advanced_gaseous_shells(image_size, shells, fractal_scale=100, blur_radius=10):
    """
    Creates advanced gaseous shells with fractal noise, diffuse textures, and filaments.
    
    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        shells: List[Dict] - List of shell properties.
        fractal_scale: int - Scale of fractal noise.
        blur_radius: int - Gaussian blur radius.

    Returns:
        PIL.Image - Image with textured shells.
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
    for shell in shells:
        # Shell parameters
        center = shell["center"]
        a = shell["semimajor_axis"]
        b = shell["semiminor_axis"]
        angle = shell["angle"]
        color = ImageColor.getrgb(shell["color"])
        thickness = shell["thickness"]

        # Create a mask for the shell
        mask = Image.new("L", image_size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for t in range(thickness):
            mask_draw.ellipse(
                (center[0] - a - t, center[1] - b - t, center[0] + a + t, center[1] + b + t),
                fill=int(255 * (1 - t / thickness))  # Gradient opacity
            )
        
        # Generate fractal noise for the shell
        noise_layer = generate_perlin_texture(image_size, scale=fractal_scale)
        noise_layer = noise_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Apply the mask to the noise
        textured_shell = Image.composite(Image.new("RGBA", image_size, color + (255,)), noise_layer, mask)
        img = Image.alpha_composite(img, textured_shell)
    
    return img

# Updated shells with advanced textures
gaseous_shells = generate_advanced_gaseous_shells(
    image_size,
    shells,
    fractal_scale=50,  # Scale for the fractal noise
    blur_radius=15     # Blur radius for diffusion
)

# Combine the shells with the existing image
final_image_with_textures = Image.alpha_composite(final_image, gaseous_shells)

# Display the image
st.image(final_image_with_textures, use_column_width=True)

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter


import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter


import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

# Función para mapear colores seleccionados a valores RGB
def hex_color_map(color_name):
    colors = {
        "blue": "0, 0, 255",
        "red": "255, 0, 0",
        "green": "0, 255, 0",
        "yellow": "255, 255, 0",
        "purple": "128, 0, 128",
        "orange": "255, 165, 0",
        "white": "255, 255, 255"
    }
    return colors.get(color_name, "0, 0, 255")

# Función recursiva para generar un patrón de ramas fractales con ruido y opacidad decreciente
def generate_fractal_branches(x, y, angle, length, depth, scale_factor=0.7, angle_variation=30, noise=5, opacity=1.0):
    branches = []
    if depth == 0 or length < 2:  # Criterio de parada
        return branches

    # Calcular ruido aleatorio en la posición final
    x_end = x + length * np.cos(np.radians(angle)) + np.random.uniform(-noise, noise)
    y_end = y + length * np.sin(np.radians(angle)) + np.random.uniform(-noise, noise)
    branches.append(((x, y), (x_end, y_end), opacity))

    # Variabilidad adicional en escala, ángulo y opacidad
    scale_factor_noisy = scale_factor * np.random.uniform(0.85, 1.15)
    angle_varied = angle_variation * np.random.uniform(0.7, 1.3)
    opacity_next = max(opacity * 0.8, 0.1)  # Decrecer opacidad

    # Generar ramas hijas (izquierda y derecha)
    branches += generate_fractal_branches(x_end, y_end, angle - angle_varied, length * scale_factor_noisy, depth - 1, scale_factor, angle_variation, noise, opacity_next)
    branches += generate_fractal_branches(x_end, y_end, angle + angle_varied, length * scale_factor_noisy, depth - 1, scale_factor, angle_variation, noise, opacity_next)

    return branches

# Función para generar ramas fractales desde el contorno interno (elíptico)
def generate_fractal_tree_on_ellipse(center_x, center_y, a, b, initial_length, depth, num_points=20, position_noise=5):
    trees = []
    theta = np.linspace(0, 2 * np.pi, num_points)
    for angle in theta:
        # Calcular punto en el contorno elíptico
        x = center_x + a * np.cos(angle) + np.random.uniform(-position_noise, position_noise)
        y = center_y + b * np.sin(angle) + np.random.uniform(-position_noise, position_noise)
        
        # Ángulo base (desde el centro hacia afuera)
        base_angle = np.degrees(angle) + np.random.uniform(-10, 10)
        
        # Generar ramas desde el contorno
        trees.extend(generate_fractal_branches(x, y, base_angle, initial_length, depth, noise=position_noise))
    return trees

# Función para generar contorno elíptico
def generate_ellipse_contour(center_x, center_y, a, b, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    contour_x = center_x + a * np.cos(theta)
    contour_y = center_y + b * np.sin(theta)
    return contour_x, contour_y

# Función para agregar ruido difuso tipo nube usando Perlin noise
def generate_perlin_noise(size, scale=10):
    np.random.seed(42)
    noise = np.random.randn(size, size)
    return gaussian_filter(noise, sigma=scale, mode="reflect")

# Función para calcular el perímetro conectado (Convex Hull)
def compute_outer_perimeter(points):
    hull = ConvexHull(points)
    return points[hull.vertices, 0], points[hull.vertices, 1]

# Streamlit UI
st.title("Patrón Fractal de Ramas con Efecto Difuso y Contorno Conectado")

# Parámetros de la elipse
st.sidebar.header("Parámetros de la Elipse")
center_x = st.sidebar.slider("Centro X", 0, 500, 250)
center_y = st.sidebar.slider("Centro Y", 0, 500, 250)
a = st.sidebar.slider("Semieje Mayor (a)", 50, 200, 100)
b = st.sidebar.slider("Semieje Menor (b)", 50, 200, 100)
external_a = st.sidebar.slider("Semieje Mayor Externo", 100, 300, 150)
external_b = st.sidebar.slider("Semieje Menor Externo", 100, 300, 150)

# Parámetros de las ramas fractales
st.sidebar.header("Parámetros del Patrón Fractal")
initial_length = st.sidebar.slider("Longitud Inicial", 5, 50, 20)
depth = st.sidebar.slider("Profundidad del Fractal", 1, 10, 6)
num_points = st.sidebar.slider("Puntos Iniciales", 10, 50, 30)
position_noise = st.sidebar.slider("Ruido en Posición", 0, 20, 8)

# Menú desplegable para seleccionar el color de las ramas
st.sidebar.header("Parámetros de Colores")
branch_color = st.sidebar.selectbox("Color de las Ramas", ["blue", "red", "green", "yellow", "purple", "orange"])

# Generar ramas fractales desde el contorno de la elipse interna
tree_branches = generate_fractal_tree_on_ellipse(center_x, center_y, a, b, initial_length, depth, num_points, position_noise)

# Crear la figura con Plotly
fig = go.Figure()

# Agregar efecto difuso con ruido de Perlin
size = 500
diffuse_noise = generate_perlin_noise(size)
x_noise, y_noise = np.meshgrid(np.linspace(center_x - a, center_x + a, size), np.linspace(center_y - b, center_y + b, size))
fig.add_trace(go.Contour(
    x=x_noise[0],
    y=y_noise[:, 0],
    z=diffuse_noise,
    colorscale=[[0, f'rgba({hex_color_map("orange")}, 0.2)'], [1, 'rgba(0,0,0,0)']],
    showscale=False,
))

# Dibujar las ramas fractales
points = []
for branch in tree_branches:
    (x1, y1), (x2, y2), opacity = branch
    # Desaparecer ramas que tocan el contorno interno
    if ((x2 - center_x)**2 / a**2) + ((y2 - center_y)**2 / b**2) <= 1.0:
        continue  # No dibujar ramas que tocan el círculo interno

    points.append([x2, y2])
    fig.add_trace(go.Scatter(
        x=[x1, x2],
        y=[y1, y2],
        mode='lines',
        line=dict(color=f'rgba({hex_color_map(branch_color)}, {opacity})', width=1),
        showlegend=False
    ))

# Dibujar el perímetro exterior conectado
if points:
    points = np.array(points)
    perimeter_x, perimeter_y = compute_outer_perimeter(points)
    fig.add_trace(go.Scatter(
        x=perimeter_x,
        y=perimeter_y,
        mode='lines',
        line=dict(color='white', width=2),
        name='Contorno Conectado'
    ))

# Dibujar el contorno interno
ellipse_x, ellipse_y = generate_ellipse_contour(center_x, center_y, a, b)
fig.add_trace(go.Scatter(
    x=ellipse_x,
    y=ellipse_y,
    mode='lines',
    line=dict(color='white', width=2),
    name='Contorno Interno'
))

# Configuración del layout
fig.update_layout(
    paper_bgcolor="black",
    plot_bgcolor="black",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    title="Envolturas con Fractales de Bifurcación y Ruido Perlin",
)

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig, use_container_width=True)


