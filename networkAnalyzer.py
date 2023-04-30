## Xiaowei Zhang 04/06
## This file is for Task 2
import torch
import torchvision
import matplotlib.pyplot as plt
import helper
import torch.optim as optim
import PIL.Image as Image
import cv2

def main():
    ## load the model from saved model file
    model = helper.myNetwork()
    networkStateDict = torch.load("results/model.pth")
    model.load_state_dict(networkStateDict)

    ## load the optimizer from saved optimizer file
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optiStateDict = torch.load("results/optimizer.pth")
    optimizer.load_state_dict(optiStateDict)

    ##print out the model
    print(model)

    ##Task 2A
    ##print out the first layer weight's shape
    assert(model.conv1.weight.shape == torch.Size([10, 1, 5, 5]))
    print("The shape of the first convolutional layer is: {}".format(model.conv1.weight.shape))
    ## print out the weight and shape of each filter and visualize the filters with pyplot
    fig = plt.figure()
    for i in range(model.conv1.weight.shape[0]):
        print("The weight of the {}th filter is:\n {}".format(i+1, model.conv1.weight[i, 0]))
        print("The shape of the {}th filter is: {}".format(i+1, model.conv1.weight[i, 0].shape))
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(model.conv1.weight[i, 0].detach().numpy(), cmap='BrBG', interpolation='none')
        plt.title("Filter: {}".format(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    ##Task 2B: apply 10 filters to the first image in the training set

    ## load the training data with batch size = 1, we only need the first batch
    batchSizeTrain = 1
    trainLoader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batchSizeTrain, shuffle=True)
    ##  get the first 1 example
    examples = enumerate(trainLoader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data[0][0])
    ## apply the first 10 filters from the first layer to the first image
    fig = plt.figure()
    with torch.no_grad():
        for j in range(model.conv1.weight.shape[0]):
            plt.subplot(3,4,j+1)
            plt.tight_layout()
            # cv2.filter2D(example_data[0][0].numpy(), -1, model.conv1.weight[i, 0].detach().numpy())
            plt.imshow(cv2.filter2D(example_data[0][0].numpy(), -1, model.conv1.weight[j, 0].detach().numpy()), cmap='gray', interpolation='none')
            plt.xticks([])
            plt.yticks([])
    plt.show()
    
if __name__ == "__main__":
    main()
    print("Done!")