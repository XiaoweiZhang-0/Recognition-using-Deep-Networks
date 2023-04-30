## Xiaowei Zhang
## This file is for Task 1F and 1G
import torch
import torchvision
import matplotlib.pyplot as plt
import helper
import torch.optim as optim
import PIL.Image as Image
import cv2

def main():
    ## Task 1F
    ## load the model from saved model file
    model = helper.myNetwork()
    networkStateDict = torch.load("results/model.pth")
    model.load_state_dict(networkStateDict)

    ## load the optimizer from saved optimizer file
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optiStateDict = torch.load("results/optimizer.pth")
    optimizer.load_state_dict(optiStateDict)

    ## load the test data with batch size = 10, we only need the first batch
    batchSizeTest = 10
    testLoader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batchSizeTest, shuffle=True)
    ## 
    ## get the first 10 examples
    examples = enumerate(testLoader)
    batch_idx, (example_data, example_targets) = next(examples)

    ## set model to evalutation mode
    model.eval()
    # create new plot
    fig = plt.figure()
    # iterate through example data
    for i in range(len(example_data)):
        ## get the output from the model based on input
        output = model(example_data[i])
        ## get the prediction with max probability
        pred = output.data.max(1, keepdim=True)[1]
        
        ## printout all output and its loss
        for j, out in enumerate(output.tolist()[0]):
            print("Digit: {} , Loss: {:.2f} ".format(j, out))
        
        ## print out the prediction and correct label 
        print("Predicted index: {}, Correct label :{}\n".format(pred.item(), example_targets[i].item()))

        if i != 9:
            plt.subplot(3,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(pred.item()))
            plt.xticks([])
            plt.yticks([])
    plt.show()
    
    ## Task 1G
    ## load an image as a tensor
    for i in range(10):
        imagePath = "task1G/out-"+str(i)+".jpg"
        ## open image
        imgColor = cv2.imread(imagePath)
        ## convert the image to grayscale
        imageGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
        ## convert image to tensor
        imageTensor = torchvision.transforms.ToTensor()(imageGray)
        ## reshape the tensor to 1x1x28x28
        imageTensor = imageTensor.reshape(1,1,28,28)
        ## nomalize the tensor
        imageTensor = torchvision.transforms.Normalize((0.1307,), (0.3081,))(imageTensor)
        ## show the tensor as graph
        # plt.imshow(imageTensor[0][0], cmap='gray', interpolation='none')
        # plt.show()
        ## get the output from the model based on input
        output = model(imageTensor)
        ## printout all output and its loss and write it to a file
        for j, out in enumerate(output.tolist()[0]):
                print("Digit: {} , Loss: {:.2f} ".format(j, out))
                with open("task1G/out-"+str(i)+".txt", "a") as f:
                    f.write("Digit: {} , Loss: {:.2f} \n".format(j, out))
                # f.close()
        ## get the prediction with max probability
        pred = output.data.max(1, keepdim=True)[1]
        ## print out the prediction and correct label and write it to a file
        print("Predicted index: {}, Correct label :{}\n".format(pred.item(), i))
        with open("task1G/out-"+str(i)+".txt", "a") as f:
            f.write("Predicted index: {}, Correct label :{}\n".format(pred.item(), i)) 
        f.close()

if __name__ == "__main__":
    main()


