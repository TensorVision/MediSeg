#!/usr/bin/env python

"""Utility functions for segmentation tasks."""

from PIL import Image
import scipy.ndimage


def replace_colors(segmentation, color_changes):
    """
    Replace the values in segmentation to the values defined in color_changes.

    Parameters
    ----------
    segmentation : numpy array
        Two dimensional
    color_changes : dict
        The key is the original color, the value is the color to change to.
        The key 'default' is used when the color is not in the dict.
        If default is not defined, no replacement is done.
        Each color has to be a tuple (r, g, b) with r, g, b in {0, 1, ..., 255}

    Returns
    -------
    PIL image
        The new colored segmentation
    """
    segmentation = scipy.misc.toimage(segmentation)
    segmentation = segmentation.convert('RGB')
    width, height = segmentation.size
    pix = segmentation.load()
    for x in range(0, width):
        for y in range(0, height):
            if pix[x, y] in color_changes:
                segmentation.putpixel((x, y), color_changes[pix[x, y]])
            elif 'default' in color_changes:
                segmentation.putpixel((x, y), color_changes['default'])
    return segmentation


def overlay_images(original_image,
                   overlay,
                   output_path):
    """
    Overlay original_image with segmentation_image.

    Store the result with the same name as segmentation_image, but with
    `-overlay`.

    Parameters
    ----------
    original_image : string
        Path to the an image file of the same size as overlay
    overlay : numpy array
    output_path : string
        Where the output will be stored.

    Returns
    -------
    str : Path of overlay image
    """
    background = Image.open(original_image)
    background = background.convert('RGB')
    overlay = overlay.convert('RGBA')

    # make black pixels transparent
    new_data = []
    for item in overlay.getdata():
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append((item[0], item[1], item[2], int(255 * 0.5)))
    overlay.putdata(new_data)
    background.paste(overlay, (0, 0), mask=overlay)
    background.save(output_path, 'PNG')
    return output_path
