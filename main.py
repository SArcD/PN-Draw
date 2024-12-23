import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageColor

def hex_to_rgb(hex_color):
    """Convert hexadecimal color to an RGB tuple."""
    return tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

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

def draw_textured_gaseous_shells(image_size, shells):
    """
    Draws diffuse, textured gaseous shells with deformities and various profiles on an image.

    Parameters:
        image_size: Tuple[int, int] - Dimensions of the image.
        shells: List[Dict] - List of dictionaries defining shell properties:
            - "center": Tuple[int, int] - Center of the shell.
            - "lobule_positive": int - Size of the positive lobe (for bipolar).
            - "lobule_negative": int - Size of the negative lobe (for bipolar).
            - "spiral_turns": int - Number of turns (for spiral).
            - "spiral_amplitude": float - Amplitude of the spiral (for spiral).
            - "deformity": float - Degree of deformity (0 = perfect shape).
            - "color_start": str - Start color (e.g., "#FF0000").
            - "color_end": str - End color (e.g., "#000000").
            - "blur": int - Gaussian blur radius.
            - "profile": str - Shape profile ("circular", "elliptical", "bipolar", "spiral", "irregular").
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))

    for shell in shells:
        center = shell["center"]
        deformity = shell["deformity"]
        color_start = ImageColor.getrgb(shell["color_start"])
        color_end = ImageColor.getrgb(shell["color_end"])
        blur_radius = shell["blur"]
        profile = shell["profile"]

        shell_img = Image.new("RGBA", image_size, (0, 0, 0, 0))
        shell_draw = ImageDraw.Draw(shell_img)

        if profile == "circular":
            radius = shell["radius"]
            for t in np.linspace(0, 1, 200):
                alpha = int(255 * (1 - t))
                r_color = int(color_start[0] + t * (color_end[0] - color_start[0]))
                g_color = int(color_start[1] + t * (color_end[1] - color_start[1]))
                b_color = int(color_start[2] + t * (color_end[2] - color_start[2]))

                current_radius = radius + deformity * np.random.uniform(-1, 1)
                shell_draw.ellipse(
                    [
                        center[0] - current_radius,
                        center[1] - current_radius,
                        center[0] + current_radius,
                        center[1] + current_radius,
                    ],
                    outline=(r_color, g_color, b_color, alpha),
                )

        shell_img = shell_img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = Image.alpha_composite(img, shell_img)

    return img

# Streamlit UI
st.title("Nebula Simulation with Circular and Elliptical Layers")

image_width = st.sidebar.slider("Image Width", 400, 1600, 800)
image_height = st.sidebar.slider("Image Height", 400, 1600, 800)
image_size = (image_width, image_height)

# Shell Parameters
st.sidebar.markdown("### Textured Gaseous Shells")
num_shells = st.sidebar.slider("Number of Shells", 1, 5, 2)
shells = []

for i in range(num_shells):
    st.sidebar.markdown(f"#### Shell {i+1}")
    profile = st.sidebar.selectbox(f"Shell {i+1} Profile", ["circular"], index=0)
    center_x = st.sidebar.slider(f"Shell {i+1} Center X", 0, image_width, image_width // 2)
    center_y = st.sidebar.slider(f"Shell {i+1} Center Y", 0, image_height, image_height // 2)
    radius = st.sidebar.slider(f"Shell {i+1} Radius", 10, 400, 200)
    deformity = st.sidebar.slider(f"Shell {i+1} Deformity", 0.0, 10.0, 1.0)
    color_start = st.sidebar.color_picker(f"Shell {i+1} Start Color", "#FF4500")
    color_end = st.sidebar.color_picker(f"Shell {i+1} End Color", "#0000FF")
    blur_radius = st.sidebar.slider(f"Shell {i+1} Blur Radius", 1, 50, 10)

    shells.append({
        "center": (center_x, center_y),
        "radius": radius,
        "deformity": deformity,
        "color_start": color_start,
        "color_end": color_end,
        "blur": blur_radius,
        "profile": profile
    })

# Generate layers
star_field_image = generate_star_field(image_size, num_stars=200)
textured_shells = draw_textured_gaseous_shells(image_size, shells)

# Combine layers
final_image = Image.alpha_composite(star_field_image.convert("RGBA"), textured_shells.convert("RGBA"))

# Display the final image
st.image(final_image, caption="Nebula Simulation with Textured Gaseous Shells", use_column_width=True)
