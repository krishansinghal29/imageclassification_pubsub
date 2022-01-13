import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, strides):
            return nn.Sequential(
                nn.ZeroPad2d((0, 2, 0, 2)),
                nn.Conv2d(inp, oup, 3, strides, 0, bias=False),
                nn.BatchNorm2d(oup, eps=1e-04, momentum=0.05, affine=True),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, strides):
            if strides == 2:
                return nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    nn.Conv2d(inp, inp, 3, strides, 0, groups=inp, bias=False),
                    nn.BatchNorm2d(inp, eps=1e-04, momentum=0.05, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup, eps=1e-04, momentum=0.05, affine=True),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, strides, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp, eps=1e-04, momentum=0.05, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup, eps=1e-04, momentum=0.05, affine=True),
                    nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(1, 16, 1),
            conv_dw(16, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            nn.AvgPool2d(7, 7),
        )
        self.fc = nn.Linear(128, 10)
        # self.activation=nn.Softmax()

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        # x = self.activation(x)
        return x