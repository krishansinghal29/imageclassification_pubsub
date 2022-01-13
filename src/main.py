import sys, os
sys.path.append(os.path.dirname(__file__))

import model_cnn
import preprocess_module 
preprocess=preprocess_module.preprocess

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import argparse



def train(data_path):
    # loading train data
    if data_path == None:
        data_train = datasets.FashionMNIST(
            "./data", train=True, transform=preprocess, download=True
        )
    else:
        data_train = datasets.ImageFolder(data_path, transform=preprocess)

    trainloader = DataLoader(data_train, batch_size=32, shuffle=True)

    model = model_cnn.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    num_epochs = 3
    total_step = len(trainloader)
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):

            optimizer.zero_grad()
            inputs, labels = data
            # print(labels)
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
    torch.save(model.state_dict(), "./model_dir/trained_model.pt")

def test(data_path):
    model = model_cnn.Net()
    model.load_state_dict(torch.load("./model_dir/trained_model.pt"))
    # Test the model
    model.eval()
    if data_path == None:
        data_test = datasets.FashionMNIST(
            "./data", train=False, transform=preprocess, download=True
        )
    else:
        data_test = datasets.ImageFolder(data_path, transform=preprocess)
    # print(data_test)

    testloader = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=True)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item() 
            total += float(labels.size(0))
    print("Test Accuracy of the model on the test images: %.2f" % (correct/total))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test or train the model on img files")
    parser.add_argument(
        "mode", help="train or test", choices=["train", "test"]
    )
    parser.add_argument(
        "-m", "--mnist_used", default=True, help="False if mnist data is not used", action='store_false'
    )
    args = parser.parse_args()

    data_path=None
    if args.mode == "train":
        if args.mnist_used==False:
            data_path="./data/data_train"
        train(data_path)
    elif args.mode =="test":
        if args.mnist_used==False:
            data_path="./data/data_test"
        test(data_path)


    
