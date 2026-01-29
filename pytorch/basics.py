import torch
import numpy as np


#   =========== Autograd ============
#   =================================

## Tensors
x = torch.tensor([[2,2], [2,3]], requires_grad=True, dtype=torch.float16)
w = torch.tensor([[1,2], [2,2]], requires_grad=True, dtype=torch.float16)
b = torch.tensor(3, requires_grad=True, dtype=torch.float16)

print(x)
print(w)

# Computational graph
y = w * x + b  # y = 2 * x + 3

# gradients
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)



##########################
##########################


##tensors
a = torch.randn(10, 3)
b = torch.randn(10, 2)

