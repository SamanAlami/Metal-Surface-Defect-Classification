# Metal Surface Defect Detection with Fine-Tuned MobileNetV2

This project demonstrates the application of transfer learning using a fine-tuned MobileNetV2 model for detecting metal surface defects from the [NEU Metal Surface Defects Dataset](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data/data) on Kaggle. The goal is to classify different types of defects in metal surfaces based on images, using a pretrained MobileNetV2 model to extract features and a newly added fully connected layer for classification.

## Project Overview

The NEU Metal Surface Defects Dataset consists of images of metal surfaces with several types of defects, including:

- **Scratches**
- **Rolled**
- **Pitted**
- **Patches**
- **Inclusion**
- **Crazing**

In this project, we use a **fine-tuned MobileNetV2** model to detect these defects. We freeze the convolutional layers of MobileNetV2, which were pretrained on the ImageNet dataset, and add a new fully connected layer to classify the defects.

## Dataset

The dataset contains images from 6 categories of metal surface defects. The images are of size 200x200 pixels, and the dataset is divided into train, validation, and test sets.

## Approach

1. **Transfer Learning**:
   - The model uses **MobileNetV2**, pretrained on ImageNet.
   - The convolutional layers are frozen to preserve the learned features.
   - A fully connected layer is added and trained for defect classification.

2. **Data Augmentation**:
   - Data augmentation techniques such as **rotation**, **shifting**, and **flipping** are applied to improve generalization and avoid overfitting.

3. **Model Architecture**:
   - **Base Model**: MobileNetV2 (pretrained on ImageNet).
   - **Top Layers**: Global average pooling, followed by a fully connected layer with 128 units and ReLU activation, and a dropout layer to prevent overfitting.
   - **Output Layer**: A softmax layer with 6 units corresponding to the 6 classes.

4. **Training**:
   - The model is trained using the **Adam optimizer** and **categorical cross-entropy loss** for multi-class classification.
   - Training is done for 10 epochs with the validation data used to track performance.

## Results

**Confusion Matrix**

| Predicted \ True | Crazing | Inclusion | Patches | Pitted | Rolled | Scratches |
| ---------------- | ------- | --------- | ------- | ------ | ------ | --------- |
| Crazing          | 12      | 0         | 0       | 0      | 0      | 0         |
| Inclusion        | 0       | 12        | 0       | 0      | 0      | 0         |
| Patches          | 0       | 0         | 12      | 0      | 0      | 0         |
| Pitted           | 0       | 0         | 0       | 12     | 0      | 0         |
| Rolled           | 0       | 0         | 0       | 0      | 12     | 0         |
| Scratches        | 0       | 0         | 0       | 0      | 0      | 12        |

**Model Performance**

The provided confusion matrix demonstrates a high level of accuracy for the model in classifying different types of surface defects:

- **Perfect Classification:** The model correctly identifies all instances of Crazing, Inclusion, Patches, Pitted, Rolled, and Scratches.
- **No Misclassifications:** There are no instances of misclassification between different defect types.

This indicates that the model has learned to distinguish between the various defect types effectively.

## Installation

To run this project, you need to install the following dependencies:

- `torch` (PyTorch)
- `torchvision`
- `matplotlib`
- `seaborn`
- `numpy`
- `scikit-learn` (for confusion matrix)

You can install the dependencies with:

```bash
pip install torch torchvision matplotlib seaborn numpy scikit-learn
```

### Data Directories

Ensure that the dataset is located in the following directories:

- **Training data**: `/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/train`
- **Test data**: `/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test`
- **Validation data**: `/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/valid`

## Acknowledgments

- The [NEU Metal Surface Defects Dataset](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data/data) is provided by the NEU.
- Thanks to the developers of **PyTorch** and **Kaggle** for providing the tools and platform to make this project possible.

## License

This project is licensed under the **MIT License**.

You are free to:

- Use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

As long as you:

- Include the original copyright notice and this permission notice in all copies or substantial portions of the Software.
- Provide appropriate credit to the original author when using this code.

The software is provided "as is", without warranty of any kind.
