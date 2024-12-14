import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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

# Streamlit interface
st.title("Nebula Simulation")

# Sidebar controls
num_stars = st.sidebar.slider("Number of Stars", min_value=100, max_value=2000, value=500, step=100)
star_x = st.sidebar.slider("Central Star X Position", min_value=0, max_value=800, value=400)
star_y = st.sidebar.slider("Central Star Y Position", min_value=0, max_value=800, value=400)
star_size = st.sidebar.slider("Central Star Size", min_value=5, max_value=50, value=20)
halo_size = st.sidebar.slider("Halo Size", min_value=10, max_value=100, value=50)
star_color = st.sidebar.color_picker("Central Star Color", "#FFFFFF")

# Generate images
image_size = (800, 800)
star_field = generate_star_field(num_stars, image_size)
central_star = draw_central_star(image_size, (star_x, star_y), star_size, halo_size, ImageColor.getrgb(star_color))

# Combine images
combined_image = Image.alpha_composite(star_field, central_star)

# Display image
st.image(combined_image, use_column_width=True)
