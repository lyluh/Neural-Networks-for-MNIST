# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
import torch.nn.functional
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        
        alpha0 = 1 / math.sqrt(d)
        alpha1 = 1 / math.sqrt(h)

        self.W0 = Parameter(torch.empty(h, d).uniform_(-alpha0, alpha0))
        self.b0 = Parameter(torch.empty(h).uniform_(-alpha0, alpha0))
        self.W1 = Parameter(torch.empty(k, h).uniform_(-alpha1, alpha1))
        self.b1 = Parameter(torch.empty(k).uniform_(-alpha1, alpha1))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        l0 = relu(torch.matmul(x, self.W0.T) + self.b0)
        return torch.matmul(l0, self.W1.T) + self.b1

class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        alpha0 = 1 / math.sqrt(d)
        alpha1 = 1 / math.sqrt(h0)
        alpha2 = 1 / math.sqrt(h1)

        self.W0 = Parameter(torch.empty(h0, d).uniform_(-alpha0, alpha0))
        self.b0 = Parameter(torch.empty(h0).uniform_(-alpha0, alpha0))
        self.W1 = Parameter(torch.empty(h1, h0).uniform_(-alpha1, alpha1))
        self.b1 = Parameter(torch.empty(h1).uniform_(-alpha1, alpha1))
        self.W2 = Parameter(torch.empty(k, h1).uniform_(-alpha2, alpha2))
        self.b2 = Parameter(torch.empty(k).uniform_(-alpha2, alpha2))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        l0 = relu(torch.matmul(x, self.W0.T) + self.b0)
        l1 = relu(torch.matmul(l0, self.W1.T) + self.b1)
        return torch.matmul(l1, self.W2.T) + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    accuracy = 0
    epoch = 0

    while accuracy < 0.99:
        model.train()
        total_loss = 0
        num_correct = 0
        num_total = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = cross_entropy(pred, y)

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            _, predicted = torch.max(pred, dim=1)
            num_correct += (predicted == y).sum().item()
            num_total += y.size(0)

        accuracy = num_correct / num_total

        print(f'Epoch [{epoch}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}')

        losses.append(total_loss / len(train_loader))
        epoch += 1
    
    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    dataset_train = TensorDataset(x, y)
    dataset_test = TensorDataset(x_test, y_test)

    batch_size = 32
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)
    
    h = 64
    d = 784
    k = 10
    learning_rate = 0.0005

    f1model = F1(h, d, k)
    f1optimizer = Adam(f1model.parameters(), lr=learning_rate)
    f1losses = train(f1model, f1optimizer, train_loader)

    epochs = range(len(f1losses))
    plt.plot(epochs, f1losses, label='F1 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch for F1 model')
    plt.legend()
    plt.show()
    
    f1model.eval()
    f1test_loss = 0
    num_correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            pred = f1model(X)
            loss = cross_entropy(pred, y)
            f1test_loss += loss.item()

            _, predictions = torch.max(pred, dim=1)
            num_correct += (predictions == y).sum().item()

    f1test_loss /= len(test_loader)
    f1accuracy = num_correct / len(dataset_test)
    print(f'F1 Test Loss: {f1test_loss:.4f}, F1 Test Accuracy: {f1accuracy:.4f}')

    total_params_f1 = sum(p.numel() for p in f1model.parameters())
    print(f'Total number of parameters in F1: {total_params_f1}')




    h0 = 32
    h1 = 32
    learning_rate = 0.001

    f2model = F2(h0, h1, d, k)
    f2optimizer = Adam(f2model.parameters(), lr=learning_rate)
    f2losses = train(f2model, f2optimizer, train_loader)

    epochs = range(len(f2losses))
    plt.plot(epochs, f2losses, label='F2 Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch for F2 model')
    plt.legend()
    plt.show()

    f2model.eval()
    f2testloss = 0
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for X, y in train_loader:
            pred = f2model(X)
            loss = cross_entropy(pred, y)
            f2testloss += loss.item()

            _, predictions = torch.max(pred, dim=1)
            num_correct += (predictions == y).sum().item()
            num_total += len(y)

    
    f2testloss /= len(test_loader)
    accuracy = num_correct / num_total
    
    print(f'F2 Test Loss: {f2testloss:.4f}, F2 Test Accuracy: {accuracy:.4f}')

    total_params_f2 = sum(p.numel() for p in f2model.parameters())
    print(f'Total number of parameters in F2: {total_params_f2}')



if __name__ == "__main__":
    main()
