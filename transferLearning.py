## Xiaowei Zhang 04/06
## This file is for Task 3
import torch
import torchvision
import matplotlib.pyplot as plt
import helper
import torch.optim as optim
import PIL.Image as Image
import cv2
import torch.nn.functional as F
import os

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

def print_layer_frozen_status(model):
    for name, parameter in model.named_parameters():
        status = "Frozen" if not parameter.requires_grad else "Trainable"
        print(f"Layer: {name} - Status: {status}")

def main():
    ## set random seed
    torch.manual_seed(42)
    ## disable cuda acceleration
    torch.backends.cudnn.enabled = False
    ## load the model from saved model file
    model = helper.myNetwork()
    networkStateDict = torch.load("results/model.pth")
    model.load_state_dict(networkStateDict)

    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False

    print(model)

    ## replace the last layer with a new linear layer of three nodes
    model.fc2 = torch.nn.Linear(50, 3)

    ## print out the model
    print(model)

    ## load the optimizer from saved optimizer file
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # optiStateDict = torch.load("results/optimizer.pth")
    # optimizer.load_state_dict(optiStateDict)

    ## set the training set path for the Greek data set
    training_set_path = "data/greek_train"
    # DataLoader for the Greek data set
    trainLoader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( training_set_path,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    examples = enumerate(trainLoader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(batch_idx)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(batch_idx)


    ## print out the frozen status of each layer
    print_layer_frozen_status(model)

    # train the model with the Greek data set till the accuracy is 100%
    loss = 100
    epoch = 0
    correct = 0
    trainLosses = []
    trainCounter = []
    while correct / len(trainLoader.dataset) < 1:
        # helper.train(model, optimizer, trainLoader, trainLosses, trainCounter, epoch, logInterval)
        ## train the model
        epoch += 1
        model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(trainLoader):
            optimizer.zero_grad()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # if batch_idx % 1 == 0:
            correct += pred.eq(target.data.view_as(pred)).sum()
            trainLosses.append(loss.item())
            trainCounter.append((batch_idx*64) + ((epoch-1)*len(trainLoader.dataset)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(trainLoader.dataset),
            100. * batch_idx / len(trainLoader), loss.item() ))
        print(" Accuracy: {}/{} ({:.0f}%)\n ".format(correct, len(trainLoader.dataset), 100. * correct / len(trainLoader.dataset)))


    # print out the training loss
    # fig = plt.figure()
    # plt.plot(trainCounter, trainLosses, color='blue')
    # plt.legend(['Train Loss'], loc='upper right')
    # # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # plt.show()

    ## test the model on custom input
    ## set the test set path for the Greek data set
    test_set_path = "data/greek_test"
    # DataLoader for the Greek data set
    testLoader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( test_set_path,
                                            transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                            GreekTransform(),
                                                                                            torchvision.transforms.Normalize(
                                                                                                (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 1,
        shuffle = True )
    
    ## iterate through the test set and calculate the accuracy
    model.eval()
    test_loss = 0
    correct = 0
    fig = plt.figure()
    for i, (data, target) in enumerate(testLoader):
        # plt.imshow(data[0][0], cmap='gray', interpolation='none')
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        # print("target", target)
        # print("pred", pred)
        if i != 9:
            plt.subplot(3,3,i+1)
            plt.tight_layout()
            plt.imshow(data[0][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(pred.item()))
            plt.xticks([])
            plt.yticks([])
    plt.show()
    test_loss /= len(testLoader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testLoader.dataset),
    100. * correct / len(testLoader.dataset)))


if __name__ == '__main__':
    main()