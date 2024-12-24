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

# Display the updated image
st.image(final_image, caption="Nebula Simulation with Gaseous Elliptical Shells", use_column_width=True)



##############################################################################3

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import streamlit as st


def apply_weak_lensing(image, black_hole_center, schwarzschild_radius):
    """
    Apply weak gravitational lensing effect to an image.
    """
    img_array = np.array(image)
    height, width, channels = img_array.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center

    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)

    deflection = schwarzschild_radius / r**2

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


def apply_strong_lensing(image, black_hole_center, schwarzschild_radius):
    """
    Apply strong gravitational lensing effect to an image.
    """
    img_array = np.array(image)
    height, width, channels = img_array.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center

    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)

    deflection = schwarzschild_radius**2 / r

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


def apply_microlensing(image, lens_center, einstein_radius):
    """
    Apply microlensing effect to an image.
    """
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dx = x - lens_center[0]
    dy = y - lens_center[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)

    # Microlensing amplification
    u = r / einstein_radius
    amplification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    amplification = np.clip(amplification, 1, 5)

    for channel in range(channels):
        img_array[..., channel] *= amplification

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def apply_caustic_crossing(image, caustic_position, source_path, lens_strength, source_radius):
    """
    Simulate caustic crossing, where a source crosses a caustic line.
    """
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_caustic, y_caustic = caustic_position

    # Simulate the source path across the caustic
    source_center_x = np.linspace(source_path[0][0], source_path[1][0], num=50)
    source_center_y = np.linspace(source_path[0][1], source_path[1][1], num=50)

    for t in range(len(source_center_x)):
        dx = x - source_center_x[t]
        dy = y - source_center_y[t]
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 1e-5)

        # Amplification due to crossing the caustic
        amplification = lens_strength / (r + source_radius)
        amplification = np.clip(amplification, 1, 10)

        for channel in range(channels):
            img_array[..., channel] *= amplification

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def apply_kerr_lensing(image, black_hole_center, schwarzschild_radius, spin_parameter):
    """
    Apply Kerr lensing effect (rotating black hole) to an image.

    Parameters:
        image (PIL.Image): Input image to distort.
        black_hole_center (tuple): (x, y) coordinates of the black hole center.
        schwarzschild_radius (int): Schwarzschild radius in pixels.
        spin_parameter (float): Spin parameter of the black hole (dimensionless, 0 to 1).

    Returns:
        PIL.Image: Deformed image with Kerr lensing effect.
    """
    img_array = np.array(image)
    height, width, channels = img_array.shape

    # Generate coordinate grids
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center

    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)

    # Add spin effect: Frame-dragging introduces azimuthal deflection
    phi = np.arctan2(dy, dx) + spin_parameter * schwarzschild_radius / r
    deflection = schwarzschild_radius**2 / r

    new_x = x_center + r * np.cos(phi) + deflection * dx / r
    new_y = y_center + r * np.sin(phi) + deflection * dy / r

    # Ensure new coordinates are within image bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    deformed_img_array = np.zeros_like(img_array)
    for channel in range(channels):
        deformed_img_array[..., channel] = map_coordinates(
            img_array[..., channel], [new_y.ravel(), new_x.ravel()], order=1, mode="constant", cval=0
        ).reshape((height, width))

    return Image.fromarray(deformed_img_array)



# Streamlit UI
#st.title("Gravitational Lensing Simulation")

# Select lensing type
#lensing_type = st.sidebar.selectbox("Select Lensing Type", ["Weak Lensing", "Strong Lensing", "Microlensing", "Caustic Crossing"])


def apply_kerr_lensing(image, black_hole_center, schwarzschild_radius, spin_parameter):
    """
    Apply Kerr lensing effect (rotating black hole) to an image.

    Parameters:
        image (PIL.Image): Input image to distort.
        black_hole_center (tuple): (x, y) coordinates of the black hole center.
        schwarzschild_radius (int): Schwarzschild radius in pixels.
        spin_parameter (float): Spin parameter of the black hole (dimensionless, 0 to 1).

    Returns:
        PIL.Image: Deformed image with Kerr lensing effect.
    """
    img_array = np.array(image)
    height, width, channels = img_array.shape

    # Generate coordinate grids
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_center, y_center = black_hole_center

    dx = x - x_center
    dy = y - y_center
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-5)

    # Add spin effect: Frame-dragging introduces azimuthal deflection
    phi = np.arctan2(dy, dx) + spin_parameter * schwarzschild_radius / r
    deflection = schwarzschild_radius**2 / r

    new_x = x_center + r * np.cos(phi) + deflection * dx / r
    new_y = y_center + r * np.sin(phi) + deflection * dy / r

    # Ensure new coordinates are within image bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    deformed_img_array = np.zeros_like(img_array)
    for channel in range(channels):
        deformed_img_array[..., channel] = map_coordinates(
            img_array[..., channel], [new_y.ravel(), new_x.ravel()], order=1, mode="constant", cval=0
        ).reshape((height, width))

    return Image.fromarray(deformed_img_array)

# Agregar Kerr Lensing en el menú desplegable
lensing_type = st.sidebar.selectbox(
    "Select Lensing Type",
    ["Weak Lensing", "Strong Lensing", "Microlensing", "Caustic Crossing", "Kerr Lensing"]
)



# Parameters for the lens
black_hole_x = st.sidebar.slider("Black Hole X Position", 0, 800, 400)
black_hole_y = st.sidebar.slider("Black Hole Y Position", 0, 800, 400)
schwarzschild_radius = st.sidebar.slider("Schwarzschild Radius (pixels)", 1, 300, 50)

# Parameters for Microlensing
einstein_radius = st.sidebar.slider("Einstein Radius (pixels) for Microlensing", 10, 200, 50)

# Parameters for Caustic Crossing
caustic_x = st.sidebar.slider("Caustic X Position", 0, 800, 400)
caustic_y = st.sidebar.slider("Caustic Y Position", 0, 800, 400)
lens_strength = st.sidebar.slider("Lens Strength", 1, 10, 5)
source_start_x = st.sidebar.slider("Source Start X", 0, 800, 200)
source_start_y = st.sidebar.slider("Source Start Y", 0, 800, 300)
source_end_x = st.sidebar.slider("Source End X", 0, 800, 600)
source_end_y = st.sidebar.slider("Source End Y", 0, 800, 500)
source_radius = st.sidebar.slider("Source Radius (pixels)", 1, 50, 10)

# Example image generation (Replace this with your nebula image)
original_image = final_image  # Use the nebula image you created earlier



# Apply lensing effect
if lensing_type == "Weak Lensing":
    final_image = apply_weak_lensing(original_image, (black_hole_x, black_hole_y), schwarzschild_radius)
elif lensing_type == "Strong Lensing":
    final_image = apply_strong_lensing(original_image, (black_hole_x, black_hole_y), schwarzschild_radius)
elif lensing_type == "Microlensing":
    final_image = apply_microlensing(original_image, (black_hole_x, black_hole_y), einstein_radius)
elif lensing_type == "Caustic Crossing":
    final_image = apply_caustic_crossing(
        original_image,
        (caustic_x, caustic_y),
        [(source_start_x, source_start_y), (source_end_x, source_end_y)],
        lens_strength,
        source_radius
    )

# Parámetros para Kerr Lensing
if lensing_type == "Kerr Lensing":
    spin_parameter = st.sidebar.slider("Black Hole Spin Parameter (a)", 0.0, 2.0, 0.5)

# Aplicar el efecto según la selección
if lensing_type == "Kerr Lensing":
    final_image = apply_kerr_lensing(
        original_image,
        (black_hole_x, black_hole_y),
        schwarzschild_radius,
        spin_parameter
    )




# Display the final result
st.image(final_image, caption=f"{lensing_type} Applied", use_column_width=True)
##############################

import numpy as np
from PIL import Image
import streamlit as st


# Función de animación genérica
def generate_animation(final_image, lensing_type, animation_range, num_frames, schwarzschild_radius, einstein_radius, spin_parameter, lens_strength, source_radius, inclination_angle):
    """
    Generate an animation by varying black hole X position.

    Parameters:
        final_image (PIL.Image): Base image to apply lensing on.
        lensing_type (str): Type of lensing effect.
        animation_range (tuple): Start and end X positions for the black hole.
        num_frames (int): Number of frames in the animation.
        schwarzschild_radius (int): Schwarzschild radius (for applicable lensing types).
        einstein_radius (int): Einstein radius (for microlensing).
        spin_parameter (float): Black hole spin parameter (for Kerr lensing).
        lens_strength (int): Lens strength (for caustic crossing).
        source_radius (int): Source radius (for caustic crossing).
        inclination_angle (float): Inclination angle in degrees.

    Returns:
        list: Frames of the animation as PIL.Image objects.
    """
    start_x, end_x = animation_range
    frames = []

    for x_position in np.linspace(start_x, end_x, num_frames):
        y_position = int(final_image.height / 2 + (x_position - start_x) * np.tan(np.radians(inclination_angle)))
        black_hole_center = (int(x_position), y_position)

        if lensing_type == "Weak Lensing":
            frame = apply_weak_lensing(final_image, black_hole_center, schwarzschild_radius)
        elif lensing_type == "Strong Lensing":
            frame = apply_strong_lensing(final_image, black_hole_center, schwarzschild_radius)
        elif lensing_type == "Microlensing":
            frame = apply_microlensing(final_image, black_hole_center, einstein_radius)
        elif lensing_type == "Caustic Crossing":
            source_path = [(start_x, y_position), (end_x, y_position)]
            frame = apply_caustic_crossing(final_image, black_hole_center, source_path, lens_strength, source_radius)
        elif lensing_type == "Kerr Lensing":
            frame = apply_kerr_lensing(final_image, black_hole_center, schwarzschild_radius, spin_parameter)
        else:
            raise ValueError(f"Unsupported lensing type: {lensing_type}")

        # Ensure frames retain the correct mode and colors
        frame = frame.convert("RGBA")
        frames.append(frame)

    return frames


# Streamlit UI
st.title("Gravitational Lensing Animation")

# Select lensing type
#lensing_type = st.sidebar.selectbox(
#    "Select Lensing Type",
#    ["Weak Lensing", "Strong Lensing", "Microlensing", "Caustic Crossing", "Kerr Lensing"]
#)

# Parameters for the lens
#schwarzschild_radius = st.sidebar.slider("Schwarzschild Radius (pixels)", 10, 300, 50)
#einstein_radius = st.sidebar.slider("Einstein Radius (pixels)", 10, 200, 50)
#spin_parameter = st.sidebar.slider("Black Hole Spin Parameter (a)", 0.0, 1.0, 0.5)
#lens_strength = st.sidebar.slider("Lens Strength", 1, 10, 5)
#source_radius = st.sidebar.slider("Source Radius (pixels)", 1, 50, 10)

# Animation parameters
start_x = st.sidebar.slider("Start X Position", 0, 1600, 100)
end_x = st.sidebar.slider("End X Position", 0, 1600, 700)
inclination_angle = st.sidebar.slider("Inclination Angle (degrees)", -45, 45, 0)
num_frames = st.sidebar.slider("Number of Frames", 10, 100, 30)

# Generate animation button
if st.button("Generate Animation"):
    # Generate animation frames
    frames = generate_animation(
        final_image,
        lensing_type,
        (start_x, end_x),
        num_frames,
        schwarzschild_radius,
        einstein_radius,
        spin_parameter,
        lens_strength,
        source_radius,
        inclination_angle
    )

    # Preview the first frame
    st.image(frames[0], caption="First Frame of Animation", use_column_width=True)

    # Save animation as GIF
    gif_path = "/tmp/lensing_animation.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

    # Download the animation
    with open(gif_path, "rb") as f:
        st.download_button(
            label="Download Animation (GIF)",
            data=f,
            file_name="lensing_animation.gif",
            mime="image/gif",
        )



###########################3333











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

