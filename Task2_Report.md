# Task 2 Report: Transfer-Learning Image Classifier (CIFAR-10 subset)

## Setup
- Data: CIFAR-10 subset, 5000 train / 1000 validation / 1000 test images (balanced, 10 classes); validation images are disjoint from training images.
- Augmentation (train only): resize 128, random crop (padding 8), horizontal flip, rotation +/-10 degrees; ImageNet normalisation.
- Model: ResNet-18 pretrained on ImageNet. Stage 1 trains only the new 10-way head (5 epochs, Adam lr 1e-3); Stage 2 also fine-tunes layer4 (5 epochs, Adam lr 1e-4).

## Results
| Split | Accuracy | Macro precision | Macro recall | Macro F1 | Loss |
|---|---|---|---|---|---|
| Validation | 81.60% | 0.8191 | 0.8160 | 0.8150 | 0.5701 |
| Test | 82.50% | 0.8278 | 0.8250 | 0.8242 | 0.4966 |

Training curves (figures/task2_curves.png): validation accuracy rises from 48% (epoch 1) to 64% after head-only training, then to 82% after fine-tuning layer4.
Final train accuracy is 91% vs 82% validation, so some overfitting appears during fine-tuning (train loss keeps falling while validation loss flattens); more data, stronger augmentation or early stopping would help.
Errors concentrate on visually similar classes (e.g. cat/dog, deer/horse, airplane/ship); see figures/task2_cm.png.

## Saved model and inference
cifar10_resnet18.pth (weights + class names) and predict.py. Run: python predict.py --image photo.jpg (see README_inference.md).
