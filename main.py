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


import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

def generate_filaments(image_size, center, num_filaments, radius, filament_length, start_color, end_color, blur_radius):
    """
    Generate radial filaments starting from points on a reference circle.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        center (tuple): Center of the reference circle (x, y).
        num_filaments (int): Number of filaments to generate.
        radius (int): Radius of the reference circle.
        filament_length (int): Length of the filaments.
        start_color (tuple): RGB color at the start of the filament.
        end_color (tuple): RGB color at the end of the filament.
        blur_radius (int): Gaussian blur radius for smoothing filaments.

    Returns:
        PIL.Image: Image with generated filaments.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for _ in range(num_filaments):
        # Generate a random point on the reference circle
        angle = np.random.uniform(0, 2 * np.pi)
        start_x = int(center[0] + radius * np.cos(angle))
        start_y = int(center[1] + radius * np.sin(angle))

        # Calculate the end point of the filament
        end_x = int(start_x + filament_length * np.cos(angle))
        end_y = int(start_y + filament_length * np.sin(angle))

        # Draw the filament with thickness and color gradient
        for i in range(filament_length):
            t = i / filament_length
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            thickness = max(1, int(5 * (1 - t**2)))  # Thickness decreases smoothly with distance
            alpha = int(255 * (1 - t))  # Opacity decreases with distance

            # Interpolate color
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

    # Apply Gaussian blur for a smooth effect
    return img.filter(ImageFilter.GaussianBlur(blur_radius))

def generate_diffuse_gas(image_size, center, inner_radius, outer_radius, start_color, end_color, blur_radius):
    """
    Generate a diffuse gas layer between two radii with a Gaussian blur and color gradient.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        center (tuple): Center of the gas layer (x, y).
        inner_radius (int): Inner radius of the gas.
        outer_radius (int): Outer radius of the gas.
        start_color (tuple): RGB color at the inner radius.
        end_color (tuple): RGB color at the outer radius.
        blur_radius (int): Gaussian blur radius for smoothing.

    Returns:
        PIL.Image: Image with generated diffuse gas.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for r in range(inner_radius, outer_radius):
        t = (r - inner_radius) / (outer_radius - inner_radius)
        alpha = int(255 * (1 - t))

        # Interpolate color
        r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
        g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
        b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

        draw.ellipse(
            [
                center[0] - r, center[1] - r, center[0] + r, center[1] + r
            ],
            outline=(r_color, g_color, b_color, alpha), width=1
        )

    return img.filter(ImageFilter.GaussianBlur(blur_radius))

def generate_bubble(image_size, center, inner_radius, outer_radius, start_color, end_color, turbulence, blur_radius):
    """
    Generate the central bubble-like structure with gradients and noise.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        center (tuple): Center of the bubble (x, y).
        inner_radius (int): Inner radius of the bubble.
        outer_radius (int): Outer radius of the bubble.
        start_color (tuple): RGB color at the inner radius.
        end_color (tuple): RGB color at the outer radius.
        turbulence (float): Amount of turbulence/noise to apply.
        blur_radius (int): Gaussian blur radius for smoothing.

    Returns:
        PIL.Image: Image with generated bubble.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for r in range(inner_radius, outer_radius):
        t = (r - inner_radius) / (outer_radius - inner_radius)
        alpha = int(255 * (1 - t))

        # Interpolate color
        r_color = int(start_color[0] + t * (end_color[0] - start_color[0]))
        g_color = int(start_color[1] + t * (end_color[1] - start_color[1]))
        b_color = int(start_color[2] + t * (end_color[2] - start_color[2]))

        # Add turbulence
        offset_x = int(turbulence * np.random.uniform(-1, 1))
        offset_y = int(turbulence * np.random.uniform(-1, 1))

        draw.ellipse(
            [
                center[0] - r + offset_x, center[1] - r + offset_y,
                center[0] + r + offset_x, center[1] + r + offset_y
            ],
            outline=(r_color, g_color, b_color, alpha), width=1
        )

    return img.filter(ImageFilter.GaussianBlur(blur_radius))

def generate_gas_arcs(image_size, center, radius, thickness, start_angle, end_angle, start_color, end_color, turbulence, blur_radius):
    """
    Generate semicircular gas arcs with turbulence and color gradients.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        center (tuple): Center of the arcs (x, y).
        radius (int): Radius of the arcs.
        thickness (int): Thickness of the arcs.
        start_angle (float): Starting angle of the arc in degrees.
        end_angle (float): Ending angle of the arc in degrees.
        start_color (tuple): RGB color at the start of the arc.
        end_color (tuple): RGB color at the end of the arc.
        turbulence (float): Amount of turbulence/noise to apply.
        blur_radius (int): Gaussian blur radius for smoothing.

    Returns:
        PIL.Image: Image with generated gas arcs.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for t in range(thickness):
        for angle in np.linspace(np.radians(start_angle), np.radians(end_angle), 500):
            # Apply turbulence to the radius
            current_radius = radius + t + int(turbulence * np.random.uniform(-1, 1))

            # Calculate the position of the point on the arc
            x = int(center[0] + current_radius * np.cos(angle))
            y = int(center[1] + current_radius * np.sin(angle))

            # Interpolate color based on angle
            t_angle = (angle - np.radians(start_angle)) / (np.radians(end_angle) - np.radians(start_angle))
            r_color = int(start_color[0] + t_angle * (end_color[0] - start_color[0]))
            g_color = int(start_color[1] + t_angle * (end_color[1] - start_color[1]))
            b_color = int(start_color[2] + t_angle * (end_color[2] - start_color[2]))

            draw.point((x, y), fill=(r_color, g_color, b_color, 255))

    return img.filter(ImageFilter.GaussianBlur(blur_radius))

def blend_filaments_with_gas(filaments_image, gas_image, transition_strength):
    """
    Blend the edges of the filaments with the gas layer for a smoother transition.

    Parameters:
        filaments_image (PIL.Image): Image containing the filaments.
        gas_image (PIL.Image): Image containing the gas.
        transition_strength (float): Strength of the blending (0 to 1).

    Returns:
        PIL.Image: Blended image.
    """
    filaments = np.array(filaments_image)
    gas = np.array(gas_image)

    blended = filaments.copy()
    alpha_filaments = filaments[..., 3] / 255.0
    alpha_gas = gas[..., 3] / 255.0

    for c in range(3):  # Blend RGB channels
        blended[..., c] = (
            alpha_filaments * filaments[..., c] * (1 - transition_strength) +
            alpha_gas * gas[..., c] * transition_strength
        ).astype(np.uint8)

    blended[..., 3] = (np.maximum(alpha_filaments, alpha_gas) * 255).astype(np.uint8)  # Update alpha

    return Image.fromarray(blended, "RGBA")

def generate_star_field(image_size, num_stars):
    """
    Generate a star field as a PIL image.

    Parameters:
        image_size (tuple): Size of the image (width, height).
        num_stars (int): Number of stars to generate.

    Returns:
        PIL.Image: Image with generated stars.
    """
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for _ in range(num_stars):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        size = np.random.randint(1, 3)
        brightness = np.random.randint(150, 255)
        draw.ellipse(
            [x - size, y - size, x + size, y + size],
            fill=(255, 255, 255, brightness)
        )

    return img

# Streamlit interface
st.title("Nebula Simulation with Filaments, Gas Layers, and Central Bubble")

# Sidebar inputs
st.sidebar.header("Filament Parameters")
image_width = st.sidebar.slider("Image Width", 400, 1600, 800)
image_height = st.sidebar.slider("Image Height", 400, 1600, 800)
center_x = st.sidebar.slider("Center X", 0, image_width, image_width // 2)
center_y = st.sidebar.slider("Center Y", 0, image_height, image_height // 2)
radius = st.sidebar.slider("Reference Circle Radius", 10, 500, 100)
num_filaments = st.sidebar.slider("Number of Filaments", 10, 500, 100)
filament_length = st.sidebar.slider("Filament Length", 10, 300, 100)

start_color_hex = st.sidebar.color_picker("Start Color", "#FFA500")
end_color_hex = st.sidebar.color_picker("End Color", "#FF4500")
start_color = tuple(int(start_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
end_color = tuple(int(end_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

blur_radius = st.sidebar.slider("Blur Radius", 0, 30, 5)

st.sidebar.header("Diffuse Gas Parameters")
inner_radius = st.sidebar.slider("Inner Radius", 50, 400, 150)
outer_radius = st.sidebar.slider("Outer Radius", 100, 500, 300)
gas_start_color_hex = st.sidebar.color_picker("Gas Start Color", "#FF4500")
gas_end_color_hex = st.sidebar.color_picker("Gas End Color", "#0000FF")
gas_start_color = tuple(int(gas_start_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
gas_end_color = tuple(int(gas_end_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
gas_blur_radius = st.sidebar.slider("Gas Blur Radius", 0, 50, 20)

st.sidebar.header("Bubble Parameters")
bubble_inner_radius = st.sidebar.slider("Bubble Inner Radius", 10, 200, 50)
bubble_outer_radius = st.sidebar.slider("Bubble Outer Radius", 50, 300, 150)
bubble_start_color_hex = st.sidebar.color_picker("Bubble Start Color", "#FF00FF")
bubble_end_color_hex = st.sidebar.color_picker("Bubble End Color", "#000000")
bubble_start_color = tuple(int(bubble_start_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
bubble_end_color = tuple(int(bubble_end_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
bubble_turbulence = st.sidebar.slider("Bubble Turbulence", 0.0, 10.0, 2.0)
bubble_blur_radius = st.sidebar.slider("Bubble Blur Radius", 0, 30, 10)

st.sidebar.header("Gas Arc Parameters")
arc_radius = st.sidebar.slider("Arc Radius", 50, 300, 150)
arc_thickness = st.sidebar.slider("Arc Thickness", 1, 20, 5)
arc_start_angle = st.sidebar.slider("Arc Start Angle", 0, 360, 0)
arc_end_angle = st.sidebar.slider("Arc End Angle", 0, 360, 180)
arc_start_color_hex = st.sidebar.color_picker("Arc Start Color", "#FFFFFF")
arc_end_color_hex = st.sidebar.color_picker("Arc End Color", "#CCCCCC")
arc_start_color = tuple(int(arc_start_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
arc_end_color = tuple(int(arc_end_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
arc_turbulence = st.sidebar.slider("Arc Turbulence", 0.0, 10.0, 2.0)
arc_blur_radius = st.sidebar.slider("Arc Blur Radius", 0, 30, 5)

st.sidebar.header("Star Field Parameters")
num_stars = st.sidebar.slider("Number of Stars", 50, 1000, 200)

st.sidebar.header("Blending Parameters")
transition_strength = st.sidebar.slider("Transition Strength", 0.0, 1.0, 0.5, step=0.1)

# Generate the layers
image_size = (image_width, image_height)
center = (center_x, center_y)
filaments_image = generate_filaments(image_size, center, num_filaments, radius, filament_length, start_color, end_color, blur_radius)
diffuse_gas_image = generate_diffuse_gas(image_size, center, inner_radius, outer_radius, gas_start_color, gas_end_color, gas_blur_radius)
bubble_image = generate_bubble(image_size, center, bubble_inner_radius, bubble_outer_radius, bubble_start_color, bubble_end_color, bubble_turbulence, bubble_blur_radius)
gas_arcs_image = generate_gas_arcs(image_size, center, arc_radius, arc_thickness, arc_start_angle, arc_end_angle, arc_start_color, arc_end_color, arc_turbulence, arc_blur_radius)
star_field_image = generate_star_field(image_size, num_stars)

# Blend filaments and gas
blended_image = blend_filaments_with_gas(filaments_image, diffuse_gas_image, transition_strength)

                                    # Combine with bubble, arcs, and star field
final_with_bubble = Image.alpha_composite(blended_image, bubble_image)
final_with_arcs = Image.alpha_composite(final_with_bubble, gas_arcs_image)
final_image = Image.alpha_composite(final_with_arcs, star_field_image)

# Display the final image
st.image(final_image, caption="Nebula Simulation", use_column_width=True)

