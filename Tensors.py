import torch


# Scalar
scalar = torch.tensor(7)

# Get tensor back as python int
scalar.item()

# Vector (ndim is the dimention, shape is the # of elements)
vector = torch.tensor([7, 7])
vector
vector.ndim

vector.shape

# MATRIX (2 dimentional tensor)
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])

MATRIX
MATRIX.shape

#%%
# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 5, 4]]])

TENSOR.size

#%%
# Random Tensors
# Create a random tensor of size (3, 4)


random_tensor = torch.rand(5, 10, 10)
random_tensor

#%%
# Create random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size = (224, 223, 3)) #height, width, color channel
random_image_size_tensor.shape, random_image_size_tensor.ndim

#%%
# Create a tensor with all 0s
zero = torch.zeros(3, 4)
zero

#create a tensor of all 1
ones = torch.ones(3, 4)
ones 
ones.dtype
#%%
#Createing a range of tensors and tensors-like
#Use torch.range()
# 2 through 90 tensor
two_through_ninety = torch.arange(start = 2, end = 92, step = 2)
print(two_through_ninety)

torch.zeros_like(input = two_through_ninety)

#%%
# Tensor Datatypes
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype = None,
                               device = None,
                               requires_grad = False)
float_32_tensor

float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor
#%%
# Getting information from Tensors

some_tensor = torch.rand(3, 4)
some_tensor
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device of tensor: {some_tensor.device}")

#%%

# Manipulating Tensors (tensor operations)

tensor = torch.tensor([1, 2, 3])
tensor - 10

# Element wise Multiplication

print(f'Equals: {tensor * tensor}')

#matrix Multiplication
torch.matmul(tensor, tensor)

#%%

# Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

#.T needed to transpose one of the tensors so they have matching inner dimensions
torch.matmul(tensor_A, tensor_B.T)

#%%

# Tensor aggregation

x = torch.arange(0, 100, 10)
torch.mean(x.type.float32)