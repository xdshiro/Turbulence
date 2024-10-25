




















import numpy as np
import torch

def conv2d(input, kernel, stride=1, padding=0):
    # Add padding to the input
    input_padded = np.pad(input, [(padding, padding), (padding, padding)], mode='constant')

    # Get dimensions of input and kernel
    input_h, input_w = input_padded.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    # Initialize output
    output = np.zeros((output_h, output_w))

    # Perform convolution
    for i in range(0, output_h):
        for j in range(0, output_w):
            # Extract the region of interest from the input
            region = input_padded[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)

    return output

# Example usage
input_image = np.array([
    [1, 2, 0, 1],
    [4, 3, 2, 1],
    [1, 2, 3, 0],
    [0, 1, 2, 4]
])

kernel = np.array([
    [1, 0],
    [0, -1]
])

output = conv2d(input_image, kernel, stride=2, padding=1)
print(output)

def conv2d_simple(input0, kernel, pad, stride=2):
	input = torch.nn.functional.pad(input0, (pad, pad, pad, pad))

	# Get dimensions of input and kernel
	input_h, input_w = input.shape
	kernel_h, kernel_w = kernel.shape
	
	# Calculate output dimensions
	output_h = (input_h - kernel_h) // stride + 1
	output_w = (input_w - kernel_w) // stride + 1
	
	# Initialize output
	output = torch.zeros((output_h, output_w))
	
	# Perform convolution
	for i in range(0, output_h):
		for j in range(0, output_w):
			# Extract the region of interest from the input
			region = input[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
			# Perform element-wise multiplication and sum
			output[i, j] = torch.sum(region * kernel)
	
	return output

input_image = torch.Tensor([
    [1, 2, 0, 1],
    [4, 3, 2, 1],
    [1, 2, 3, 0],
    [0, 1, 2, 4]
])

kernel = torch.Tensor([
    [1, 0],
    [0, -1]
])
output = conv2d_simple(input_image, kernel, pad=1)
print(output)