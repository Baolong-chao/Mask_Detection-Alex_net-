import torch
import torch.nn as nn
from data_process import train_dataloader, test_dataloader
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2)
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2)
        )
        self._fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self._fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self._fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )
        self._fc4 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):
        conv1_output = self._conv1(input)
        conv2_output = self._conv2(conv1_output)
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)
        fc1_output = self._fc1(conv2_output)
        fc2_output = self._fc2(fc1_output)
        fc3_output = self._fc3(fc2_output)
        fc4_output = self._fc4(fc3_output)
        return fc4_output


class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2)
        )
        self._fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self._fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=2)
        )

    def forward(self, input):
        conv1_output = self._conv1(input)
        conv2_output = self._conv2(conv1_output)
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)
        fc1_output = self._fc1(conv2_output)
        fc2_output = self._fc2(fc1_output)
        return fc2_output
'''
'''
def VGG_block(num_convs, in_channels, out_channels):
    # 定义第一层，并转化为 List
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]
    # 通过循环定义其他层
    for i in range(num_convs-1):
        # List每次只能添加一个元素
        # 输入和输出channel均为out_channels
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))

    # 定义池化层
    net.append(nn.MaxPool2d(2, 2))
    # List数据前面加‘*’表示将List拆分为独立的参数
    return nn.Sequential(*net)

def VGG_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(VGG_block(n, in_c, out_c))
    return nn.Sequential(*net)



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature = VGG_net
        #self.fc = nn.Sequential(
        #    nn.Linear(256*16*16, 4096),
        #    nn.ReLU(True),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(True),
        #    nn.Linear(4096, 2)
        #)
        self.pool_shape = 256 * 16 * 16
        self.fc01 = nn.Linear(self.pool_shape, 4096)
        self.fc02 = nn.Linear(4096, 1024)
        self.fc03 = nn.Linear(1024, 2)
        self.relu =nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.feature(x)
        print(x.shape)
        out = torch.reshape(x,[-1,256*16*16])
        print(out.shape)
        out = self.fc01(out)
        out = self.relu(out)
        out = self.fc02(out)
        out = self.relu(out)
        out = self.fc03(out)
        return out
'''
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    since = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        pred = model(X)
        y = y.squeeze_()
        y = y.to(device)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        #for p in list(model.parameters()):
            #torch.sign(p)
        optimizer.step()


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            y = y.squeeze_()
            y = y.to(device)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}% \n")
'''


class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,28,kernel_size=11,stride=4,padding=2), #input(3,125,125) output(28,31,31)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2), #output(28,15,15)
            nn.Conv2d(28,56,kernel_size=5,padding=2),#output(56,15,15)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[56, 7, 7]
            nn.Conv2d(56, 56, kernel_size=3, padding=1),  # output[56, 7, 7]
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 56, kernel_size=3, padding=1),  # output[56, 7, 7]
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 112, kernel_size=3, padding=1),  # output[256, 7, 7]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[112, 3, 3]
        )
        '''
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(112 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        '''
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(112 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
            x = self.features(x)  # 卷积层提取特征
            x = torch.flatten(x, start_dim=1)  # pytorch中tensor通常的排列顺序：[batch,channel,height,width]
            x = self.classifier(x)  # 全连接层分类
            return x




def test_model(model,dataloader,loss_fn,num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    torch.set_grad_enabled(False)

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs - 1))
        print('-'*20)

        model.eval()

        running_loss = 0.0
        running_corrects = 0
        sum_total = 0
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            sum_total += labels.numel()

            outputs = model(inputs)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            # backward + optimize only if in training phase
            print(loss)
            running_loss += loss.item()  #add the loss
            running_corrects += (preds == labels.data).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / sum_total
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        if  epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict()) #deepcpy:copy and stable
            #torch.save(model.state_dict(), "model_"+str(epoch)+".pkl")

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == "__main__" : 
    torch.cuda.empty_cache()
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))


    model = AlexNet().to(device)
    model.eval()
    model.load_state_dict(torch.load('try_model.pkl', map_location='cpu'))  #加载需要检测的参数
    loss_fn = nn.CrossEntropyLoss()

    test_model(model,train_dataloader,loss_fn)
    print("Done!")


