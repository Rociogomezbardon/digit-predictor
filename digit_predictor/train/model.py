import torch.nn as nn

class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        # input 32 x 16 x 3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # input 32 x 16 x 16
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # input 16 x 8 x 16
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # input 16 x 8 x 32
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(8*4*64, 1024)
        self.fc2 = nn.Linear(1024, len(train_data.classes))

        self.maxpool = nn.MaxPool2d(2, 2)
        #self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)

        x = x.view(-1,8*4*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x
