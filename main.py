### Xiaowei Zhang
## This is the main entry of the program for Task 1A to 1D
import torch
import torchvision
import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchview import draw_graph
import graphviz
import helper

batchSizeTrain = 64
batchSizeTest = 1000
nEpochs = 5
learningRate = 0.01
momentum = 0.5
logInterval = 10


## this is the main function for this project
def main():
    ### Task 1B
    graphviz.set_jupyter_format('png')
    ## set random seed
    torch.manual_seed(42)
    ## disable cuda acceleration
    torch.backends.cudnn.enabled = False

    trainLoader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batchSizeTrain, shuffle=True)
    testLoader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batchSizeTest, shuffle=True)
    # for i, image in enumerate(trainLoader):
    #     if i >= 6:
    #         break
    #     print(i, image)
    examples = enumerate(trainLoader)
    ## example_targets is the ground truth values of the example data
    batch_idx, (example_data, example_targets) = next(examples)


    ## Task 1A
    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

    ## Task 1C
    model = helper.myNetwork()
    # model_graph = draw_graph(model, input_size=example_data.size(), device='meta', save_graph=True)

    ## Task 1D
    optimizer = optim.SGD(model.parameters(), lr=learningRate,
                      momentum=momentum)
    trainLosses = []
    trainCounter = []
    testLosses = []
    testCounter = [i*len(trainLoader.dataset) for i in range(nEpochs + 1)]

    
    helper.test(model, testLoader, testLosses)
    for epoch in range(1, nEpochs+1):
        helper.train(model, optimizer, trainLoader, trainLosses, trainCounter, epoch, logInterval)
        helper.test(model, testLoader, testLosses)

    fig = plt.figure()
    plt.plot(trainCounter, trainLosses, color='blue')
    plt.scatter(testCounter, testLosses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

if __name__ == "__main__":
    main()
