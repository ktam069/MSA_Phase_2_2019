# MSA Phase 2 Project 2019 (AI Project)

By King Hang Tam

## Description

This is the phase 2 project for the MSA AI stream. The CNN model is trained to classify 32x32 images in the CIFAR-10 dataset.

## Dependencies

### Language and version:

Python 3.6

### Install the following libraries as required:

* tensorflow

* keras

* numpy

* pandas

* matplotlib

* seaborn

To install all required libraries, run the following command from the *ai_project* folder:

```
python -m pip install -r requirements.txt
```

### Dataset:

Download the *CIFAR-10 python version* file from http://www.cs.toronto.edu/~kriz/cifar.html and extract the *cifar-10-batches-py* folder into the *ai_project* folder.

## Instructions for Running

To run the program, navigate to the *ai_project* folder and run the following from command line:

```
python phase2_ai.py
```

### Settings:

The following settings can be modified for different behaviours:

```
# Used (the latest) peviously saved model
USE_LOADED_MODEL = False

# Load weights from the latest saved checkpoint (has lower priority than USE_LOADED_MODEL)
USE_CHECKPOINT = False
```

By default, both flags are set to false. This will make the program train the model from scratch.

Enabling *USE_LOADED_MODEL* will load the latest model that was saved in the *saved_models* folder. (This model is saved after training all epochs.)

Enabling *USE_CHECKPOINT* will load the model with the weights from the latest checkpoint that was saved in the *saved_checkpoints* folder. (This model is from the epoch with the highest validation accuracy.)

## Features

### Transfer Learning with ResNet50:

To run the program, navigate to the *ai_project* folder and run the following from command line:

```
python transfer_learning.py
```

The accuracy is below 50% so far. Further work would be required.

### Deployed Model:

Deploying Python code to Azure Functions, as outlined in the following tutorial: https://docs.microsoft.com/en-us/azure/azure-functions/tutorial-vs-code-serverless-python

Currently incomplete - the code can be found in the *deployment* branch.

### Local Demo:

This is a small script to allow the user to easily classify an image from the image URL. This was primarily created for demo purposes in the absence of a functional web app.

Run the following command from the *ai_project* folder:

```
python classify_image.py
```

This will load the saved model, and prompt the user to enter an image URL for classification (Ctrl+C to terminate). Alternatively, modify the settings for the URL in the script itself.
