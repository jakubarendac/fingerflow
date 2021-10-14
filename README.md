# FingerFlow

[![Python](https://img.shields.io/pypi/pyversions/fingerflow.svg?style=plastic)](https://badge.fury.io/py/fingerflow)
[![PyPI](https://badge.fury.io/py/fingerflow.svg)](https://badge.fury.io/py/fingerflow)

FingerFlow is an end-to-end deep learning Python framework for fingerprint minutiae manipulation built on top of [Keras](https://keras.io/) - [TensorFlow](https://www.tensorflow.org/) high-level API.

In current stable version 1.0.0 following modules are provided:

- **extractor** - module responsible for extraction and classification of minutiae points from fingerprints

## GPU support

FingerFlow supports GPU acceleration on [CUDA®-enabled graphic cards](https://developer.nvidia.com/cuda-gpus).

## Software requirements

- **Python 3.9 or newer**
- **CUDA** - for TensorFlow GPU acceleration (if missing, CPU will be used for computation)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install FingerFlow. We reccomend to use it in **pip** or **anaconda** enviroment.

Installation in anaconda enviroment:

```bash
pip install fingerflow
```

## Modules

### Extractor

Module responsible for extraction and classification of fingerprint minutiae points. Extractor is built using [MinutiaeNet](https://github.com/luannd/MinutiaeNet) neural network architecture.

Extractor contains 2 modules:

- **MinutiaeNet** - module responsible for extracting minutiae points from fingerprint image. Using MinutiaeNet neural network architecture.
- **ClassifyNet** - module responsible for classifying extraced minutiae points. Architecture based on FineNet module of MinutiaeNet

#### Neural networks models

- **CoarseNet**: [Googledrive](https://drive.google.com/file/d/1alvw_kAyY4sxdzAkGABQR7waux-rgJKm/view?usp=sharing) || [Dropbox](https://www.dropbox.com/s/gppil4wybdjcihy/CoarseNet.h5?dl=0)
- **FineNet**: [Googledrive](https://drive.google.com/file/d/1wdGZKNNDAyN-fajjVKJoiyDtXAvl-4zq/view?usp=sharing) || [Dropbox](https://www.dropbox.com/s/k7q2vs9255jf2dh/FineNet.h5?dl=0)
- **ClassifyNet**: [Googledrive](https://drive.google.com/drive/folders/124M3iLy4yMlAtegO0OXo_bl4Q0IIgPWE)

#### API

#### `Extractor`

Class which provides all functionality for extraction of minutiae points

```python
fingerflow.extractor.Extractor()
```

**Arguments**

- `coarse_net_path` - used for setting path to pretrained model of submodule CoarseNet
- `fine_net_path` - used for setting path to pretrained model of submodule FineNet
- `classify_net_path` - used for setting path to pretrained model of submodule ClassifyNet

**Methods**

- `extract_minutiae(image_data)` - used for extracting minutiae points from input RGB image data. Methods accepts input data in form of [numpy](https://numpy.org/) array. Function returns numpy ndarray of extracted and classified minutiae points in following form:
  - **x** - x coordinate of minutiae point
  - **y** - y coordinate of minutiae point
  - **angle** - direction of minutiae point rotation
  - **score** - minutiae point extraction confidence
  - **class** - type of minutiae point. In FingerFlow 1.0.0 we support following minutiae classes:
    - **ending**
    - **bifurcation**
    - **fragment**
    - **enclosure**
    - **crossbar**
    - **other**

**Usage**

```python
import cv2
import numpy as np
from fingerflow.extractor import Extractor

extractor = Extractor("coarse_net", "fine_net", "classify_net")

image = np.array(cv2.imread("some_image"))

extracted_minutiae = extractor.extract_minutiae(image)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
