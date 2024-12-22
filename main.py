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
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageColor

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageColor

# Function to draw textured gaseous shells with deformities and various profiles
def draw_textured_gaseous_shells(image_size, shells):
    """
    Draws diffuse, textured gaseous shells with deformities and various profiles on an image.

    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        shells: List[Dict] - List of dictionaries defining shell properties:
            - "center": Tuple[int, int] - Center of the shell.
            - "semimajor_axis": int - Semimajor axis (radius for ellipse).
            - "semiminor_axis": int - Semiminor axis.
            - "inner_radius": int - Inner radius of the shell.
            - "outer_radius": int - Outer radius of the shell.
            - "deformity": float - Degree of deformity (0 = perfect shape).
            - "angle": float - Rotation angle of the shell in degrees.
            - "color_start": str - Start color (e.g., "#FF0000").
            - "color_end": str - End color (e.g., "#000000").
            - "blur": int - Gaussian blur radius.
            - "profile": str - Shape profile ("circular", "elliptical", "bipolar", "spiral", "irregular").
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))

    for shell in shells:
        center = shell["center"]
        a = shell["semimajor_axis"]
        b = shell["semiminor_axis"]
        inner_radius = shell["inner_radius"]
        outer_radius = shell["outer_radius"]
        deformity = shell["deformity"]
        angle = shell["angle"]
        color_start = ImageColor.getrgb(shell["color_start"])
        color_end = ImageColor.getrgb(shell["color_end"])
        blur_radius = shell["blur"]
        profile = shell["profile"]

        shell_img = Image.new("RGBA", image_size, (0, 0, 0, 0))
        shell_draw = ImageDraw.Draw(shell_img)

        # Generate deformable contours for each radius
        for r in range(inner_radius, outer_radius):
            t = (r - inner_radius) / (outer_radius - inner_radius)
            r_color = int(color_start[0] + t * (color_end[0] - color_start[0]))
            g_color = int(color_start[1] + t * (color_end[1] - color_start[1]))
            b_color = int(color_start[2] + t * (color_end[2] - color_start[2]))
            alpha = int(255 * (1 - t))

            points = []
            if profile == "circular":
                for theta in np.linspace(0, 2 * np.pi, 200):
                    x = center[0] + (r + deformity * np.sin(theta * np.random.uniform(1, 3))) * np.cos(theta)
                    y = center[1] + (r + deformity * np.cos(theta * np.random.uniform(1, 3))) * np.sin(theta)
                    points.append((x, y))
            elif profile == "elliptical":
                for theta in np.linspace(0, 2 * np.pi, 200):
                    x = center[0] + (a + deformity * np.sin(theta * np.random.uniform(1, 3))) * np.cos(theta)
                    y = center[1] + (b + deformity * np.cos(theta * np.random.uniform(1, 3))) * np.sin(theta)
                    points.append((x, y))
            elif profile == "bipolar":
                for theta in np.linspace(0, 2 * np.pi, 200):
                    factor = np.sin(2 * theta)
                    x = center[0] + (a + r * factor + deformity * np.random.uniform(-1, 1)) * np.cos(theta)
                    y = center[1] + (b + r * factor + deformity * np.random.uniform(-1, 1)) * np.sin(theta)
                    points.append((x, y))
            elif profile == "spiral":
                for theta in np.linspace(0, 6 * np.pi, 300):
                    spiral_factor = r / outer_radius
                    x = center[0] + (spiral_factor * r + deformity * np.sin(theta)) * np.cos(theta)
                    y = center[1] + (spiral_factor * r + deformity * np.cos(theta)) * np.sin(theta)
                    points.append((x, y))
            elif profile == "irregular":
                for theta in np.linspace(0, 2 * np.pi, 200):
                    noise = np.random.uniform(-deformity, deformity)
                    x = center[0] + (r + noise) * np.cos(theta)
                    y = center[1] + (r + noise) * np.sin(theta)
                    points.append((x, y))

            rotated_shell = Image.new("RGBA", image_size, (0, 0, 0, 0))
            rotated_draw = ImageDraw.Draw(rotated_shell)
            rotated_draw.polygon(points, outline=(r_color, g_color, b_color, alpha))

            rotated_shell = rotated_shell.rotate(angle, center=center)
            shell_img = Image.alpha_composite(shell_img, rotated_shell)

        shell_img = shell_img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = Image.alpha_composite(img, shell_img)

    return img

# Streamlit section for adding textured gaseous shells
st.sidebar.markdown("### Textured Gaseous Shells")
num_shells = st.sidebar.slider("Number of Shells", 1, 5, 2)
shells = []

for i in range(num_shells):
    st.sidebar.markdown(f"#### Shell {i+1}")
    center_x = st.sidebar.slider(f"Shell {i+1} Center X", 0, image_size[0], image_size[0] // 2)
    center_y = st.sidebar.slider(f"Shell {i+1} Center Y", 0, image_size[1], image_size[1] // 2)
    semimajor_axis = st.sidebar.slider(f"Shell {i+1} Semimajor Axis", 50, 400, 200)
    semiminor_axis = st.sidebar.slider(f"Shell {i+1} Semiminor Axis", 50, 400, 150)
    inner_radius = st.sidebar.slider(f"Shell {i+1} Inner Radius", 10, 200, 50)
    outer_radius = st.sidebar.slider(f"Shell {i+1} Outer Radius", inner_radius, 400, 150)
    deformity = st.sidebar.slider(f"Shell {i+1} Deformity", 0.0, 10.0, 1.0)
    angle = st.sidebar.slider(f"Shell {i+1} Angle", 0, 360, 0)
    color_start = st.sidebar.color_picker(f"Shell {i+1} Start Color", "#FF4500")
    color_end = st.sidebar.color_picker(f"Shell {i+1} End Color", "#0000FF")
    blur_radius = st.sidebar.slider(f"Shell {i+1} Blur Radius", 1, 50, 10)
    profile = st.sidebar.selectbox(f"Shell {i+1} Profile", ["circular", "elliptical", "bipolar", "spiral", "irregular"], index=0)

    shells.append({
        "center": (center_x, center_y),
        "semimajor_axis": semimajor_axis,
        "semiminor_axis": semiminor_axis,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "deformity": deformity,
        "angle": angle,
        "color_start": color_start,
        "color_end": color_end,
        "blur": blur_radius,
        "profile": profile,
    })

# Generate and combine shells with the previous final image
textured_shells = draw_textured_gaseous_shells(image_size, shells)
final_image_with_textured_shells = Image.alpha_composite(final_image, textured_shells)

# Display the updated image
st.image(final_image_with_textured_shells, caption="Nebula Simulation with Textured Gaseous Shells", use_column_width=True)


