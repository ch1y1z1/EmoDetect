import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import os
import torch.utils.data as data
import tqdm
import pandas as pd


EmoDict = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

# 反转字典
EmoDict_rev = {v: k for k, v in EmoDict.items()}


transform = T.Compose([
    T.Grayscale(1),
    T.Resize(48), 
    T.CenterCrop(48), 
    T.ToTensor(),
    T.Normalize(mean=[.5], std=[.5]) 
])


class EmoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.imgs = []
        for emotion in os.listdir(root):
            emotion_path = os.path.join(root, emotion)
            if os.path.isdir(emotion_path): 
                for img in os.listdir(emotion_path):
                    self.imgs.append(os.path.join(emotion_path, img))
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_name = os.path.basename(os.path.dirname(img_path)) 
        label = EmoDict_rev[label_name] 
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, torch.tensor(label).long()

    def __len__(self):
        return len(self.imgs)


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset_path = './train'
val_dataset_path = './test'

train_dataset = EmoDataset(train_dataset_path, transforms=transform)
val_dataset = EmoDataset(val_dataset_path, transforms=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

model = EmotionNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.01)

# 定义训练过程
def train_model(model, criterion, optimizer, num_epochs=25):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for idx, epoch in enumerate(tqdm.tqdm(range(num_epochs))):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history[phase + '_loss'].append(epoch_loss)
            history[phase + '_acc'].append(epoch_acc.cpu())
        if idx % 5 == 0:
            torch.save(model.state_dict(), 'net_{}.pth'.format(epoch))
    df = pd.DataFrame(history)
    df.to_csv('training_history.csv', index=False)

    return model

# 开始训练模型
model = train_model(model, criterion, optimizer, num_epochs=51)

torch.save(model.state_dict(), 'net.pth')