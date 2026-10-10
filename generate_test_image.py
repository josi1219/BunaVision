"""
Helper script to generate a synthetic test image representing green coffee beans
with a few defects on a clean white background.
"""
from PIL import Image, ImageDraw
import random
import os

def create_synthetic_coffee_image(output_path="test_sample.jpg", num_beans=30):
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Bean color palettes
    healthy_colors = [
        (134, 168, 115), # Pale olive green
        (148, 175, 128),
        (120, 155, 105),
        (158, 185, 138),
        (140, 160, 110)
    ]
    black_defect = (35, 30, 28)      # Full black defect
    sour_defect = (160, 110, 60)      # Sour/brown defect
    broken_defect = (130, 160, 110)

    random.seed(42)

    # Place beans in a grid-like or scattered layout with spacing
    cols = 6
    rows = 5
    x_step = width // (cols + 1)
    y_step = height // (rows + 1)

    bean_idx = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if bean_idx >= num_beans:
                break
            
            # Position with slight jitter
            cx = c * x_step + random.randint(-15, 15)
            cy = r * y_step + random.randint(-15, 15)
            rx = random.randint(22, 28)
            ry = random.randint(14, 18)

            # Assign defect types to a few beans
            if bean_idx == 4:
                fill_color = black_defect # Full black bean
            elif bean_idx == 11:
                fill_color = sour_defect  # Sour bean
            elif bean_idx == 18:
                fill_color = sour_defect
            else:
                fill_color = random.choice(healthy_colors)

            # Draw coffee bean (ellipse)
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
            draw.ellipse(bbox, fill=fill_color, outline=(80, 80, 70), width=1)
            
            # Draw center fissure (crease)
            crease_color = (max(0, fill_color[0] - 40), max(0, fill_color[1] - 40), max(0, fill_color[2] - 40))
            draw.line([(cx - rx + 5, cy), (cx + rx - 5, cy)], fill=crease_color, width=2)

            bean_idx += 1

    img.save(output_path, quality=95)
    print(f"Generated test sample with {num_beans} beans at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_synthetic_coffee_image()
