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

def generate_star_field(image_size, num_stars):
    """
    Generate a star field as a PIL image with white stars.

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
        size = np.random.randint(1, 4)
        brightness = np.random.randint(200, 255)  # Brightness for white stars
        draw.ellipse(
            [x - size, y - size, x + size, y + size],
            fill=(255, 255, 255, brightness)
        )

    return img


def generate_filaments(image_size, center, num_filaments, radius, filament_length, start_color, end_color, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

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

def generate_diffuse_gas(image_size, center, inner_radius, outer_radius, start_color, end_color, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

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

def generate_bubble(image_size, center, inner_radius, outer_radius, start_color, end_color, turbulence, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

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

def generate_gas_arcs(image_size, center, radius, thickness, start_angle, end_angle, start_color, end_color, turbulence, blur_radius, elliptical):
    width, height = image_size
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

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

def draw_central_star_with_filaments(image_size, position, star_size, halo_size, color, num_filaments, dispersion, blur_radius):
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

image_width = st.sidebar.slider("Image Width", 400, 1600, 800)
image_height = st.sidebar.slider("Image Height", 400, 1600, 800)
center_x = st.sidebar.slider("Center X", 0, image_width, image_width // 2)
center_y = st.sidebar.slider("Center Y", 0, image_height, image_height // 2)
center = (center_x, center_y)

# Filament parameters
num_filaments = st.sidebar.slider("Number of Filaments", 10, 500, 100)
filament_radius = st.sidebar.slider("Filament Radius", 10, 400, 200)
filament_length = st.sidebar.slider("Filament Length", 10, 300, 100)
filament_start_color = st.sidebar.color_picker("Filament Start Color", "#FFA500")
filament_end_color = st.sidebar.color_picker("Filament End Color", "#FF4500")
filament_blur = st.sidebar.slider("Filament Blur", 0, 30, 5)
filament_elliptical = st.sidebar.checkbox("Elliptical Filaments", False)

# Diffuse gas parameters
gas_inner_radius = st.sidebar.slider("Gas Inner Radius", 50, 400, 150)
gas_outer_radius = st.sidebar.slider("Gas Outer Radius", 100, 500, 300)
gas_start_color = st.sidebar.color_picker("Gas Start Color", "#FF4500")
gas_end_color = st.sidebar.color_picker("Gas End Color", "#0000FF")
gas_blur = st.sidebar.slider("Gas Blur", 0, 50, 20)
gas_elliptical = st.sidebar.checkbox("Elliptical Gas", False)

# Bubble parameters (continuación)
bubble_inner_radius = st.sidebar.slider("Bubble Inner Radius", 10, 200, 50)
bubble_outer_radius = st.sidebar.slider("Bubble Outer Radius", 50, 300, 150)
bubble_start_color = st.sidebar.color_picker("Bubble Start Color", "#FF00FF")
bubble_end_color = st.sidebar.color_picker("Bubble End Color", "#000000")
bubble_turbulence = st.sidebar.slider("Bubble Turbulence", 0.0, 10.0, 2.0)
bubble_blur = st.sidebar.slider("Bubble Blur", 0, 30, 10)
bubble_elliptical = st.sidebar.checkbox("Elliptical Bubble", False)

# Arc parameters
arc_radius = st.sidebar.slider("Arc Radius", 50, 300, 150)
arc_thickness = st.sidebar.slider("Arc Thickness", 1, 20, 5)
arc_start_angle = st.sidebar.slider("Arc Start Angle", 0, 360, 0)
arc_end_angle = st.sidebar.slider("Arc End Angle", 0, 360, 180)
arc_start_color = st.sidebar.color_picker("Arc Start Color", "#FFFFFF")
arc_end_color = st.sidebar.color_picker("Arc End Color", "#CCCCCC")
arc_turbulence = st.sidebar.slider("Arc Turbulence", 0.0, 10.0, 2.0)
arc_blur = st.sidebar.slider("Arc Blur", 0, 30, 5)
arc_elliptical = st.sidebar.checkbox("Elliptical Arcs", False)

# Star field parameters
num_stars = st.sidebar.slider("Number of Stars", 50, 1000, 200)
star_colors = ["#FFFFFF", "#FFD700", "#87CEEB"]

# Central star parameters
st.sidebar.header("Central Star")
star_size = st.sidebar.slider("Star Size", 5, 50, 20)
halo_size = st.sidebar.slider("Halo Size", 10, 100, 50)
star_color_hex = st.sidebar.color_picker("Star Color", "#FFFF00")
star_color = tuple(int(star_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
num_star_filaments = st.sidebar.slider("Number of Star Filaments", 5, 100, 30)
filament_dispersion = st.sidebar.slider("Filament Dispersion", 1, 50, 10)
star_blur_radius = st.sidebar.slider("Star Blur Radius", 0, 20, 5)

# Generate layers
image_size = (image_width, image_height)
filaments_image = generate_filaments(
    image_size, center, num_filaments, filament_radius, filament_length,
    tuple(int(filament_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    tuple(int(filament_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    filament_blur, filament_elliptical
)
diffuse_gas_image = generate_diffuse_gas(
    image_size, center, gas_inner_radius, gas_outer_radius,
    tuple(int(gas_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    tuple(int(gas_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    gas_blur, gas_elliptical
)
bubble_image = generate_bubble(
    image_size, center, bubble_inner_radius, bubble_outer_radius,
    tuple(int(bubble_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    tuple(int(bubble_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    bubble_turbulence, bubble_blur, bubble_elliptical
)
gas_arcs_image = generate_gas_arcs(
    image_size, center, arc_radius, arc_thickness, arc_start_angle, arc_end_angle,
    tuple(int(arc_start_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    tuple(int(arc_end_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)),
    arc_turbulence, arc_blur, arc_elliptical
)

star_field_image = generate_star_field(image_size, num_stars)

central_star_image = draw_central_star_with_filaments(
    image_size, center, star_size, halo_size, star_color, num_star_filaments,
    filament_dispersion, star_blur_radius
)




# Combine images
final_image = Image.alpha_composite(star_field_image, filaments_image)
final_image = Image.alpha_composite(final_image, diffuse_gas_image)
final_image = Image.alpha_composite(final_image, bubble_image)
final_image = Image.alpha_composite(final_image, gas_arcs_image)
final_image = Image.alpha_composite(final_image, central_star_image)

# Display the final image
st.image(final_image, use_column_width=True)




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
final_image = Image.alpha_composite(final_image, gas_arcs_image)
final_image = Image.alpha_composite(final_image, central_star_image)

# Display the final image
st.image(final_image, caption="Nebula Simulation", use_column_width=True)


############################################################
def generate_gaseous_shells(image_size, center, semi_major, semi_minor, angle, inner_radius, outer_radius, start_color, end_color, deformity, blur_radius):
    """
    Generate gaseous shells with elliptical profiles, deformities, and sinusoidal variations.

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
        deformity: float - Degree of irregularity in the shell shape.
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

        # Generate points for the elliptical profile with sinusoidal deformity
        points = []
        for theta in np.linspace(0, 2 * np.pi, 200):  # 200 points for smooth shell
            noise = deformity * np.sin(3 * theta + t * np.pi)  # Sinusoidal deformity
            radius = r + noise
            x_ellipse = radius * semi_major * np.cos(theta) / outer_radius
            y_ellipse = radius * semi_minor * np.sin(theta) / outer_radius

            # Rotate the points by the specified angle
            x_rotated = cos_theta * x_ellipse - sin_theta * y_ellipse
            y_rotated = sin_theta * x_ellipse + cos_theta * y_ellipse

            # Translate to the center
            x = int(center[0] + x_rotated)
            y = int(center[1] + y_rotated)
            points.append((x, y))

        draw.polygon(points, outline=(r_color, g_color, b_color, alpha))

    # Apply Gaussian blur to smooth out edges
    return img.filter(ImageFilter.GaussianBlur(blur_radius))

# Streamlit: Gaseous Shell Parameters
st.sidebar.header("Gaseous Shells")
num_shells = st.sidebar.slider("Number of Shells", 1, 5, 3)

shells = []
for i in range(num_shells):
    st.sidebar.subheader(f"Shell {i + 1}")
    semi_major = st.sidebar.slider(f"Semi-Major Axis (Shell {i + 1})", 10, 800, 200)
    semi_minor = st.sidebar.slider(f"Semi-Minor Axis (Shell {i + 1})", 10, 800, 150)
    angle = st.sidebar.slider(f"Inclination Angle (Shell {i + 1})", 0, 360, 45)
    inner_radius = st.sidebar.slider(f"Inner Radius (Shell {i + 1})", 10, 400, 100)
    outer_radius = st.sidebar.slider(f"Outer Radius (Shell {i + 1})", inner_radius, 500, inner_radius + 50)
    deformity = st.sidebar.slider(f"Deformity (Shell {i + 1})", 0.0, 20.0, 5.0)
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
        shell["start_color"], shell["end_color"], shell["deformity"], shell["blur_radius"]
    )
    gaseous_shells = Image.alpha_composite(gaseous_shells, shell_image)

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
    shift_factor = np.sqrt(1 - 2 * schwarzschild_radius / r)
    shift_factor = np.clip(shift_factor, 0.8, 1.2)  # Limitar rangos para evitar extremos

    # Crear una máscara para identificar píxeles válidos (que no sean completamente negros)
    mask = np.any(img_array > 0, axis=-1)  # Excluir píxeles completamente negros

    # Aplicar corrimientos de color
    img_array[mask, 0] = img_array[mask, 0] * shift_factor[mask]  # Corrimiento al rojo
    img_array[mask, 2] = img_array[mask, 2] / shift_factor[mask]  # Corrimiento al azul

    # Ajustar el canal verde para mantener balance cromático
    img_array[mask, 1] = img_array[mask, 1] * (0.5 + 0.5 * shift_factor[mask])

    # Limitar los valores dentro del rango válido [0, 255]
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
black_hole_x_fixed = st.sidebar.slider("Black Hole X Position (Static Image)", 0, 800, 400)
black_hole_y_fixed = st.sidebar.slider("Black Hole Y Position (Static Image)", 0, 800, 400)
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



################################









import numpy as np
from PIL import Image, ImageDraw
import streamlit as st


def create_photon_ring(image_size, shadow_radius, ring_width):
    """
    Create a photon ring image using Pillow.
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    center = (image_size[0] // 2, image_size[1] // 2)

    for r in range(shadow_radius, shadow_radius + ring_width):
        intensity = int(255 * (1 - (r - shadow_radius) / ring_width))  # Gradient intensity
        color = (255, 140, 0, intensity)  # Orange-like color with gradient alpha
        draw.ellipse(
            [center[0] - r, center[1] - r, center[0] + r, center[1] + r],
            outline=color,
            width=1
        )

    return img


# Streamlit UI
st.title("Black Hole Shadow and Photon Ring")

# Parameters for the photon ring
image_size = (800, 800)  # Size must match the nebulosa and field of stars
shadow_radius = st.sidebar.slider("Shadow Radius", 50, 300, 150)
ring_width = st.sidebar.slider("Ring Width", 10, 100, 30)

# Generate the photon ring (shadow and ring)
photon_ring = create_photon_ring(image_size, shadow_radius, ring_width)

# Simulate loading the previously generated image of nebulosa and stars (final_image)
# Replace this with your actual `final_image`
final_image = Image.new("RGBA", image_size, (0, 0, 0, 0))  # Placeholder for nebulosa image
# Load or generate the nebulosa image here (e.g., from your previous process)
# Example:
# final_image = Image.open("nebula_stars.png").convert("RGBA")

# Combine the photon ring with the nebulosa and star field
combined_image = Image.alpha_composite(final_image, photon_ring)

# Display the final combined image
st.image(combined_image, caption="Black Hole Shadow with Nebulosa and Photon Ring", use_column_width=True)

