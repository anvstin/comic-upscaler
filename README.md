# Comic Upscaler

This script aims to automatically upscale comics it finds in a folder. It uses Real-ESRGAN to upscale the images.

The comics can be in a folder, CBZ, or any ZIP file format.
Most image formats are supported, including PNG, JPEG, and WEBP.
Animated WEBP are not supported and will be converted to a static image (the first frame).

## Requirements

When cloning the repo, use the --recurse option. If not, in the repo folder, type :

```
git submodule init
git submodule update
```

Use conda to install the requirements:

```bash
conda env create -f environment.yml
```

## Usage

```bash
python3 upscale.py <input_folder> <output_folder>
```

The script has numerous customization options, you can see them with:

```bash
python3 upscale.py --help
```

## Project used

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## Authors

- [**Aurélien Visentin**](https://github.com/anvstin)
