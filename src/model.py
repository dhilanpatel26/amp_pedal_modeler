import torch
import torch.nn as nn
import torch.nn.functional as F


# Q: How does the GuitarAmpSimulator model work?
# A: The GuitarAmpSimulator model is a convolutional neural network (CNN) that takes a 1D input tensor with shape [batch_size, channels, length] and applies two convolutional layers followed by two fully connected layers. The first convolutional layer has 16 output channels, a kernel size of 3, and a stride of 1. The second convolutional layer has 32 output channels, a kernel size of 3, and a stride of 1. Both convolutional layers are followed by ReLU activation functions. The output of the second convolutional layer is flattened to a 1D tensor, which is then passed through a fully connected layer with 1024 output features and a ReLU activation function. Finally, the output is passed through another fully connected layer with 256 output features, which serves as the output of the model.
# Q: What is shape of the input tensor to the GuitarAmpSimulator model?
# A: The input tensor to the GuitarAmpSimulator model should have a shape of [batch_size, 1, 256], where batch_size is the number of input samples and 256 is the length of the input signal.
# Q: What is meant by input samples?
# A: Input samples refer to the individual signals or waveforms that are fed into the model for processing. Each input sample corresponds to a single audio signal or waveform that the model will process and generate an output for.
class GuitarAmpSimulator(nn.Module):
    def __init__(self):
        super(GuitarAmpSimulator, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Define the second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Define a fully connected layer
        self.fc1 = nn.Linear(in_features=32*256, out_features=1024)
        # Define the output layer
        self.fc2 = nn.Linear(in_features=1024, out_features=256)

    def forward(self, x):
        # Apply the first convolutional layer followed by a ReLU activation function
        x = F.relu(self.conv1(x))
        # Apply the second convolutional layer followed by a ReLU activation function
        x = F.relu(self.conv2(x))
        # Flatten the tensor from [batch_size, channels, length] to [batch_size, channels*length]
        x = x.view(x.size(0), -1)
        # Apply the first fully connected layer followed by a ReLU activation function
        x = F.relu(self.fc1(x))
        # Apply the output layer
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    # Create a random tensor with shape (batch_size, channels, length)
    input_signal = torch.randn(10, 1, 256)
    # Create an instance of the GuitarAmpSimulator model
    model = GuitarAmpSimulator()
    # Get the output of the model
    output_signal = model(input_signal)
    # Print the shape of the output tensor
    print(output_signal.shape)  # Should be [10, 256]