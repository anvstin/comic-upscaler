# Comic Upscaler

This script aims to automatically upscale comics it finds in a folder. It uses Real-ESRGAN to upscale the images.

## Requirements

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
