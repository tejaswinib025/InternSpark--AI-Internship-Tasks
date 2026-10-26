# Running inference
1. Install: `pip install torch torchvision pillow`
2. Put `cifar10_resnet18.pth` and `predict.py` in one folder.
3. Run: `python predict.py --image path/to/photo.jpg`   (optional: `--model other_checkpoint.pth`). Works on CPU; no GPU needed.
Output example: `Predicted: truck (confidence 97.3%)` plus the top-3 classes.
Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Images are resized to 128x128, so photos of a single, centred object work best.
