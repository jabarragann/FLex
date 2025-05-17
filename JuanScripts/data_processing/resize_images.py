import os
from dataclasses import dataclass
from pathlib import Path

import click
import cv2


@dataclass
class MyContext:
    input_folder: Path
    output_folder: Path
    target_resolution: tuple[int, int]


def resize_images(ctx: MyContext):

    files = sorted(ctx.input_folder.glob("*.png"))

    for filename in files:
        img = cv2.imread(str(filename))

        if img is None:
            print(f"Warning: Unable to read {ctx.input_path}")
            continue

        resized_img = cv2.resize(
            img, ctx.target_resolution, interpolation=cv2.INTER_AREA
        )

        output_path = ctx.output_folder / filename.name
        cv2.imwrite(str(output_path), resized_img)


@click.group()
@click.option("--input-folder", type=click.Path(), required=True)
@click.option("--output-folder", type=click.Path(), required=True)
@click.option("--target-resolution", type=str, required=True, help="e.g. 1920x1080")
@click.pass_context
def cli(
    ctx: click.Context, input_folder: str, output_folder: str, target_resolution: str
):

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        raise Exception(f"Error: {input_folder} does not exist.")
    if not output_folder.exists():
        output_folder.mkdir(exist_ok=True)

    target_resolution = tuple(map(int, target_resolution.split("x")))

    ## Context object gets passed to subcommands - CLICK.
    ctx.obj = MyContext(
        input_folder=input_folder,
        output_folder=output_folder,
        target_resolution=target_resolution,
    )


@cli.command()
@click.pass_obj
def run(ctx: MyContext):
    resize_images(ctx)


# Entry point
if __name__ == "__main__":
    """
    run with

    python video_to_frames.py --input-video . --output-dir ../../ run

    see: https://click.palletsprojects.com/en/stable/api/#context
    """
    cli()
