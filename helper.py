import torch
import torch.nn as nn
import torch.nn.functional as F

class myNetwork(nn.Module):
    def __init__(self):
        super(myNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
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

## training function
def train(network, optimizer, trainLoader, trainLosses, trainCounter, epoch, logInterval):
  ## set the network to trainning mode
  network.train()
  ## iterate through the batch and train the model
  for batch_idx, (data, target) in enumerate(trainLoader):
    ## reset optimizer to zero gradients before back propagation
    optimizer.zero_grad()
    ## train this batch for one forward pass and get the predicted output
    output = network(data)
    ## calculate the negative log likelihood loss between output and ground truth
    loss = F.nll_loss(output, target)
    ## back propagation to compute gradient loss wrt network parameters
    loss.backward()
    ## update the model parameters
    optimizer.step()

    if batch_idx % logInterval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(trainLoader.dataset),
        100. * batch_idx / len(trainLoader), loss.item()))
      trainLosses.append(loss.item())
      trainCounter.append(
        (batch_idx*64) + ((epoch-1)*len(trainLoader.dataset)))
      ## Task 1E
      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')

## testing function
def test(network, test_loader, test_losses):
  ## set the network to evaluation mode
  network.eval()
  ## initialize test loss and number of corrected classified cases for this batch
  test_loss = 0
  correct = 0
  ## disable gradient cal inside this block
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      ## aggregate the test loss across this batch
      test_loss += F.nll_loss(output, target, size_average=False).item()
      ## get the output and return the index of predictions with the highest probabilities
      pred = output.data.max(1, keepdim=True)[1]
      ## check if the predicted results equal to the ground truth and sum up all correct ones
      correct += pred.eq(target.data.view_as(pred)).sum()
    ## calculate the test loss for this batch and append it to the testlosses
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
