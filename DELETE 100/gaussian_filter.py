import torch
import torch.nn.functional as F


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
	"""Create a 2D Gaussian kernel."""
	x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
	print(x)
	y = x.view(-1, 1)
	kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
	kernel = kernel / kernel.sum()  # Normalize the kernel
	return kernel


def gaussian_filter(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
	"""Apply a 2D Gaussian filter to the input tensor."""
	kernel = gaussian_kernel(kernel_size, sigma).to(input.device)
	# Reshape kernel to match input dimensions for convolution
	kernel = kernel.view(1, 1, kernel_size, kernel_size)
	
	# Perform 2D convolution with Gaussian kernel
	filtered = F.conv2d(input, kernel, padding=kernel_size // 2)
	return filtered


def gaussian_kernel2(size: int, sigma: float) -> torch.Tensor:
	"""Create a 2D Gaussian kernel."""
	# xy = torch.linspace(-(size - 1) / 2, (size - 1) /2, size)
	# xy = torch.arange(-(size - 1) / 2, (size - 1) /2 + 1)
	# x, y = torch.meshgrid(xy, xy, indexing='ij')
	x = torch.linspace(-(size - 1) / 2, (size - 1) /2, size)
	y = x.view(-1, 1)
	kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
	kernel = kernel / torch.sum(kernel)  # Normalize the kernel
	return kernel


def gaussian_filter2(input: torch.Tensor, kernel) -> torch.Tensor:
	"""Apply a 2D Gaussian filter to the input tensor."""
	kernel = kernel.unsqueeze(0).unsqueeze(0)
	# Reshape kernel to match input dimensions for convolution
	
	
	# Perform 2D convolution with Gaussian kernel
	filtered = torch.nn.functional.conv2d(input, kernel, padding=1)
	return filtered


# Example usage
if __name__ == "__main__":
	k1 = gaussian_kernel(3, 1)
	k2 = gaussian_kernel2(3, 1)
	print(k2, k1)
	input_image = torch.tensor(
		[[1.0, 2.0, 0.0, 1.0],
		 [4.0, 3.0, 2.0, 1.0],
		 [1.0, 2.0, 3.0, 0.0],
		 [0.0, 1.0, 2.0, 4.0]
	]).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 4, 4)
	print(gaussian_filter2(input_image, k2))

	
	# Create a sample 4x4 input image (batch size = 1, channels = 1)
	input_image = torch.tensor([
		[[1.0, 2.0, 0.0, 1.0],
		 [4.0, 3.0, 2.0, 1.0],
		 [1.0, 2.0, 3.0, 0.0],
		 [0.0, 1.0, 2.0, 4.0]]
	]).unsqueeze(0)  # Shape: (1, 1, 4, 4)
	
	# Apply Gaussian filter
	kernel_size = 3  # Kernel size 3x3
	sigma = 1.0  # Standard deviation of the Gaussian
	
	output = gaussian_filter(input_image, kernel_size, sigma)
	print(output)
