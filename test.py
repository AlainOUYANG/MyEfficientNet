import os
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet


ARCH = 'efficientnet-b0'
PRETRAINED = True
CLS_NUM = 2
ADVPROP = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

classes = ('cat', 'dog')

model = EfficientNet.from_pretrained(ARCH, num_classes=CLS_NUM, advprop=ADVPROP)
model.load_state_dict(torch.load('./cpt/2021-12-09 00:25:25.366239_checkpoint.pth.tar'))
model.eval()
test_list = os.listdir('./data/test/')

for img_path in test_list:
    img = transform(Image.open('./data/test/'+img_path)).unsqueeze(0)
    output = model(img)
    _, pred = torch.max(output, dim=1)
    print(f'Image: {img_path},\tPredicted: {classes[pred.item()]}')
