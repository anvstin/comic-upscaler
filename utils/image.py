import os
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def divide(input_folder: str, output_folder: str, divide: int, format=None):
    """
    Divide the images in the input folder by the divide factor and save them in the output folder

    Args:
        input_folder (str): The input folder
        output_folder (str): The output folder
        divide (int): The divide factor
        format (str, optional): The format of the images. Defaults to None.
    """
    if divide <= 1:
        return

    # Divide the images by 2 to reduce the size
    dirs = os.listdir(input_folder)
    size = len(dirs)

    print(f"  0/{size}")
    for i, image in enumerate(dirs):
        if os.path.isdir(image):
            continue
        if image.endswith(f'.{format}'):
            img = Image.open(input_folder + '/' + image)
            img = img.resize((int(img.width / divide), int(img.height / divide)), Image.LANCZOS)
            img.save(output_folder + '/' + image, format=format)
            print(f"  {i + 1}/{size} {image}", end='\r')

        # Replace print

def fit_to_width(input_folder: str, width: int, format: str|None=None):
    """
    Resize all images in the input folder to fit the width

    Args:
        input_folder (str): The input folder
        width (int): The width to fit
        format (str, optional): The format of the images. Defaults to None.

    """
    if width <= 0:
        return
    # Walk each image and resize it to fit the width
    paths = list(enumerate(glob.glob(input_folder + '/**', recursive=True)))
    size = len(paths)
    for i, image in paths:
        if  format is not None and  not image.endswith(f'.{format}'):
            continue
        if os.path.isdir(image):
            continue

        img = Image.open(image)
        # Using CV2 to open the image
        # img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img.width < width:
            continue

        ratio = width / img.width
        if ratio >= 1:
            continue
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
        # Using CV2 to resize the image
        # img = cv2.resize(img, (width, int(img.height * ratio)), interpolation=cv2.INTER_LANCZOS4)

        # Save the image
        img.save(image, format=format)
        # Using CV2 to save the image
        # cv2.imencode(re.search(r'\.([a-zA-Z]+)$', image).group(1), img)[1].tofile(image)
        print(f"  {i + 1}/{size} {os.path.basename(image)}", end='\r')

