### Xiaowei Zhang
## This is the file for Task 4
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchview import draw_graph
import graphviz
import helper

batchSizeTest = 1000
learningRate = 0.01
momentum = 0.5
logInterval = 10

## this is the network for Task 4
class myNetwork(nn.Module):
    def __init__(self, dropoutRate):
        super(myNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropoutRate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

## this is the main function for task 4
def main():
    graphviz.set_jupyter_format('png')
    ## set random seed
    torch.manual_seed(42)
    ## disable cuda acceleration
    torch.backends.cudnn.enabled = False


    ## Find the training and testing mean and standard deviation
    trainMean = torchvision.datasets.FashionMNIST('data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])).data.float().mean()/255
    trainStd = torchvision.datasets.FashionMNIST('data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                                ])).data.float().std()/255
    testMean = torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                                ])).data.float().mean()/255
    testStd = torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                                ])).data.float().std()/255



    ## Set three parameters of the model to be changed in the grid search
    ## The three parameters are: the dropout rates of the dropout layers, the number of epochs, and the batch size while training
    ## The three parameters are set to be lists of values to be tested
    # dropoutRates = [0.1, 0.2, 0.3, 0.4, 0.5]
    dropoutRates = [0.3, 0.4, 0.5]
    nEpochs = [5, 10, 15, 20, 25]
    batchSizeTrain = [64, 128, 256, 512, 1024]
    bestDropoutRate = 0
    bestEpoch = 0
    bestBatchSize = 0
    bestModel = None
    bestOptimizer = None
    bestTestLoss = float('inf')
    ## iterate through all the combinations of the three parameters and find the best model and save it with the corresponding parameters
    for dropoutRate in dropoutRates:
        for nEpoch in nEpochs:
            for batchSize in batchSizeTrain:
                
                ## import training data from FashionMNIST
                trainLoader = torch.utils.data.DataLoader(
                    torchvision.datasets.FashionMNIST('data', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(), torchvision.
                                            transforms.Normalize((trainMean,), (trainStd,))
                                        ])),
                    batch_size=batchSize, shuffle=True)
                
                testLoader = torch.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (testMean,), (testStd,))
                                            ])),
                batch_size=batchSizeTest, shuffle=True)

                examples = enumerate(trainLoader)

                model = myNetwork(dropoutRate)
                optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)

                trainLosses = []
                trainCounter = []
                testLosses = []
                testCounter = [i*len(trainLoader.dataset) for i in range(nEpoch + 1)]
                for epoch in range(1, nEpoch + 1):
                    ## train the model
                    helper.train(model, optimizer, trainLoader, trainLosses, trainCounter, epoch, logInterval)
                    ## test the model
                    helper.test(model, testLoader, testLosses)

                ## print out the test loss and the corresponding parameters
                # print('Test set: Average loss: {:.4f}, Dropout Rate: {}, Epoch: {}, Batch Size: {}\n'.format(
                #     testLosses[-1], dropoutRate, epoch, batchSize))
        
                with open('bestModel.txt', 'a') as f:
                    f.write('Test set: Average loss: {:.4f}, Dropout Rate: {}, Epoch: {}, Batch Size: {}\n'.format(
                        min(testLosses), dropoutRate, epoch, batchSize))
                f.close()
                ## find the best model and save the parameters
                if min(testLosses) < bestTestLoss:
                    bestDropoutRate = dropoutRate
                    bestEpoch = epoch
                    bestBatchSize = batchSize
                    bestModel = model
                    bestOptimizer = optimizer
                    bestTestLoss = min(testLosses)
    
    ## print out the best model and the corresponding parameters
    with open('bestModel.txt', 'a') as f:
        f.write('Best Model: Average loss: {:.4f}, Dropout Rate: {}, Epoch: {}, Batch Size: {}\n'.format(
            bestTestLoss, bestDropoutRate, bestEpoch, bestBatchSize))
    
    f.close()
    ## save the best model
    torch.save(bestModel.state_dict(), 'fashionModel/bestModel.pt')
    ## save the optimizer
    torch.save(optimizer.state_dict(), 'fashionModel/optimizer.pt')
        
    

if __name__ == "__main__":
    main()
                

