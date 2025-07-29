import requests
from PIL import Image, ImageDraw, ImageFont
import random
import streamlit as st
import textwrap

def get_memes():
    """Fetch memes from imgflip API"""
    try:
        response = requests.get("https://api.imgflip.com/get_memes")
        response.raise_for_status()
        data = response.json()
        if data["success"]:
            return data["data"]["memes"]
        else:
            st.error("Could not fetch memes from imgflip API.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching memes: {e}")
        return []

def create_meme(image_url, top_text, bottom_text, font_path="arial.ttf", font_size=40):
    """Create a meme by adding text to an image"""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        def draw_text(text, position):
            text_width, text_height = draw.textsize(text, font)
            x = (image.width - text_width) / 2
            if position == "top":
                y = 10
            else:
                y = image.height - text_height - 10

            # Add text outline
            for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + offset[0], y + offset[1]), text, font=font, fill="black")

            draw.text((x, y), text, font=font, fill="white")

        # Wrap text
        wrapper = textwrap.TextWrapper(width=30)
        top_lines = wrapper.wrap(top_text)
        bottom_lines = wrapper.wrap(bottom_text)

        for i, line in enumerate(top_lines):
            draw_text(line, "top")

        for i, line in enumerate(bottom_lines):
            draw_text(line, "bottom")

        return image

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating meme: {e}")
        return None
