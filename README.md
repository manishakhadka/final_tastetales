# Minor age prdiction model

## Dataset

### UTKFaces dataset

This dataset has 23,708 images of human faces with ages ranging from 0 to 116 years old. The images are labeled with the age of the person in the image. The dataset is available at [Kaggle](https://www.kaggle.com/jangedoo/utkface-new).

## Getting started

### Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Usage

To generate migrations, run the following command:

```bash
alembic revision --autogenerate -m "<migration message>"
```

To apply migrations, run the following command:

```bash
alembic upgrade head
```

To train the model, run the following command:

```bash
python train.py --epochs <number of epochs> --batch_size <batch size>
```

To predict the age of an image, run the following command:

```bash
python train.py --predict-image <path to image>
```

### Data preprocessing

The images were preprocessed to have the same size and to be grayscale. The images were resized to 128x128 pixels and the pixel values were normalized to be between 0 and 1.

## Model

The model is a convolutional neural network with 4 convolutional layers and 2 fully connected layers. The model was trained using the Adam optimizer and the mean squared error loss function.

