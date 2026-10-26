import argparse, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
ap = argparse.ArgumentParser(); ap.add_argument('--image', required=True); ap.add_argument('--model', default='cifar10_resnet18.pth')
a = ap.parse_args()
ck = torch.load(a.model, map_location='cpu'); m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(ck['classes']))
m.load_state_dict(ck['state_dict']); m.eval()
tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
x = tf(Image.open(a.image).convert('RGB')).unsqueeze(0)
with torch.no_grad(): pr = torch.softmax(m(x),1)[0]
top = pr.topk(3)
print(f"Predicted: {ck['classes'][int(top.indices[0])]} (confidence {top.values[0]*100:.1f}%)")
print('Top-3:', ', '.join(f"{ck['classes'][int(i)]} {v*100:.1f}%" for v,i in zip(top.values, top.indices)))
