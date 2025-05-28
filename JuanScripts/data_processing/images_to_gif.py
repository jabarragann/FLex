import os
from pathlib import Path

import click
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@click.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Input image file",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=1000,
    help="Duration of each frame in milliseconds",
)
def main(input, duration):
    input = Path(input)

    # List of image filenames (replace or auto-generate as needed)
    path = input
    image_files = sorted(path.glob("*.png"))

    # Optional: Sort the filenames if needed
    image_files.sort()

    # Write to images.txt in FFmpeg concat format
    with open("images.txt", "w") as f:
        for filename in image_files:
            f.write(f"file '{filename}'\n")

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=32)
    except OSError:
        # Fallback to default if font not found
        font = ImageFont.load_default()

    # Read and write to GIF
    output_gif = input / "output.gif"
    with imageio.get_writer(output_gif, mode="I", duration=duration, loop=0) as writer:
        for idx, filename in enumerate(image_files):
            # image = imageio.imread(filename)
            image = Image.open(filename).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), f"{idx+1:02d}", font=font, fill=(255, 0, 0))

            writer.append_data(np.asarray(image))

        writer.append_data(image)
        writer.append_data(image)


if __name__ == "__main__":
    main()
