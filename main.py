import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visdom import Visdom
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from prettytable import PrettyTable


CHECKPOINT_DIR = './cpt'
ARCH = 'efficientnet-b0'
PRETRAINED = True
CLS_NUM = 2
ADVPROP = True
BATCH_SIZE = 64
NUM_EPOCHS = 2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if PRETRAINED:
    model = EfficientNet.from_pretrained(ARCH, num_classes=CLS_NUM, advprop=ADVPROP)
    print('=> using pre-trained model "{}"'.format(ARCH))
else:
    model = EfficientNet.from_name(ARCH, override_params={'num_classes': CLS_NUM})
    print('=> creating model "{}"'.format(ARCH))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

train_dataset = datasets.ImageFolder('./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.ImageFolder('./data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

viz = Visdom(port=8097)
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss', legend=['loss']))


def train(train_loader, epoch):
    model.train()
    train_loss = 0
    total_num = len(train_loader.dataset)
    print(f'Total number of train images: {total_num}, total number of batches: {len(train_loader)}.')
    for i, (images, target) in enumerate(train_loader):
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 2 == 0:
            s = f'Epoch {epoch + 1}/{NUM_EPOCHS},\tstep {i + 1}/{len(train_loader)},\tloss = {loss.item():.4f}'
            print(s)

        viz.line([loss.item()], [i + (epoch * len(train_loader))], win='train_loss', update='append')
    ave_losss = train_loss / len(train_loader)
    print(f'Training set: Epoch {epoch + 1}/{NUM_EPOCHS}, Average loss: {ave_losss:.4f}')


def val(val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total_num = len(val_loader.dataset)
    print(f'Total number of val images: {total_num}, total number of batches: {len(val_loader)}.')
    y_pred, y_val = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            output = model(images)
            loss = criterion(output, target)
            _, pred = torch.max(output, dim=1)
            correct += torch.sum(pred == target).item()
            val_loss += loss.item()
            y_pred += pred.tolist()
            y_val += target.tolist()
        acc = correct / total_num
        avg_loss = val_loss / len(val_loader)
        print(f'Val set: Average loss: {avg_loss:.4f},\t Accuracy: {correct}/{total_num} ({acc:.2%})')
        metric_results = PrettyTable()
        metric_results.field_names = ['Metric', 'Precision', 'Recall']
        metric_results.add_row(['cat', f'{precision_score(y_val, y_pred):.4f}', f'{recall_score(y_val, y_pred):.4f}'])
        metric_results.add_row(['dog', f'{precision_score(y_pred, y_val):.4f}', f'{recall_score(y_pred, y_val):.4f}'])
        print('***** Classify Metric Test results *****')
        print(metric_results)
        print('***** Multi-Class Confusion Matrix *****')
        cm = PrettyTable()
        cm.field_names = ['confusion', 'pred_cat', 'pred_dog']
        cm.add_row(['real_cat', confusion_matrix(y_val, y_pred)[0, 0], confusion_matrix(y_val, y_pred)[0, 1]])
        cm.add_row(['real_dog', confusion_matrix(y_val, y_pred)[1, 0], confusion_matrix(y_val, y_pred)[1, 1]])
        print(cm)
        print(f'Accuracy score: {accuracy_score(y_val, y_pred):.2%}')


for epoch in range(NUM_EPOCHS):
    train(train_loader, epoch)

print('Finish training!')

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, '%s_checkpoint.pth.tar' % datetime.datetime.now()))

val(val_loader)
