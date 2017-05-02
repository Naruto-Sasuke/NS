npy4th = require 'npy4th'
require 'lib/utils'
require 'nn'
require 'torch'
-- read a .npy file into a torch tensor
--array = npy4th.loadnpy('array.npy')

local input = torch.Tensor(2,3,3)
input[1]:fill(1)
input[2]:fill(2)
print(input)

local conv = torch.Tensor(input)
conv = input:view(1,conv:size(1),conv:size(2),conv:size(3))
print(conv)

local net = nn.Sequential()
local modual = nn.SpatialConvolution(2,2,3,3):noBias()
net:add(modual)


net:get(1).weight[1] = conv:clone()
print('*****weight')
print(net:get(1).weight)

print(net:forward(input))


-- data = {}
-- v = {}
-- for i = 1,3 do
--     data[i] = torch.Tensor(4,5)
--     v[i] = torch.Tensor(2,12)
-- end
-- torch.save('data.dat',{data,v})

-- data = torch.load('data.dat')

-- print(#data)
-- for i = 1,#data do
--     print(data[i])
--     print('\n')
-- end
-- print(v)

-- save a torch.*Tensor into a .npy file
-- a = {}
-- a[1] = torch.randn(3,4)
-- print(a[1])
-- --myTensor = torch.Tensor({1,2,3})
-- npy4th.savenpy('tensor.npy', a[1])

-- append a torch.*Tensor to an existing .npy file
-- myTensor2 = torch.Tensor({4,5,6})
-- npy4th.savenpy('tensor.npy', myTensor2, 'a')

-- array,arr1 = npy4th.loadnpy('tensor.npy')
-- print(array)
-- print(arr1)