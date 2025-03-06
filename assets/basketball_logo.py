"""
Generate a basketball logo for the app.
"""
import base64
from io import BytesIO
from PIL import Image, ImageDraw

def create_basketball_logo():
    # Create a new image with a transparent background
    width, height = 200, 200
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Draw the basketball (orange circle)
    orange = (255, 140, 0, 255)
    draw.ellipse((10, 10, width-10, height-10), fill=orange)
    
    # Draw the black lines on the basketball
    black = (0, 0, 0, 255)
    # Horizontal line
    draw.arc((10, 10, width-10, height-10), 0, 180, fill=black, width=5)
    draw.arc((10, 10, width-10, height-10), 180, 360, fill=black, width=5)
    
    # Vertical line
    draw.arc((10, 10, width-10, height-10), 90, 270, fill=black, width=5)
    draw.arc((10, 10, width-10, height-10), -90, 90, fill=black, width=5)
    
    # Draw curved lines to simulate the basketball texture
    draw.arc((40, 40, width-40, height-40), 30, 150, fill=black, width=3)
    draw.arc((40, 40, width-40, height-40), 210, 330, fill=black, width=3)
    
    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    
    # Encode the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

def get_logo_html(size=35):
    img_str = create_basketball_logo()
    return f'<img src="data:image/png;base64,{img_str}" width="{size}" height="{size}">'

if __name__ == "__main__":
    # Test the function
    img_str = create_basketball_logo()
    with open("basketball_logo.png", "wb") as f:
        f.write(base64.b64decode(img_str))
    print("Basketball logo saved to basketball_logo.png")
