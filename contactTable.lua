-- First, I have to predict the possibility of this work here.
-- Almost 100% failed. But, I would rather have a try than give up directly.
-- I want to use padding(-1,x,x,x) to cut the spatial size of input.
-- Then I can get the number of w*h patches as inputs of contactTable network.

-- Secondly, I need to backword gradients. It's tough. As When I get all w*h's 
-- gradients, how can I manage to combine as gradInput?
-- If autograd can help, I need to try autograd directly taken 
require 'nn'
require 'cutorch'
require 'torch'

local input = torch.Tensor({{1,2,2,3}})
input = input:view(1,1,4):repeatTensor(1,3,1):repeatTensor(3,1,1)
print(input)

local mlp = nn.Sequential()

local contb = nn.ConcatTable()

local p = nn.SpatialZeroPadding(0,-1,0,0)
local q = nn.SpatialZeroPadding(-1,0,0,0)
contb:add(p)
contb:add(q)

mlp:add(contb)
print(mlp:forward(input))




mlp = nn.Sequential()       -- Create a network that takes a Tensor as input
mlp:add(nn.SplitTable(2))
c = nn.ParallelTable()      -- The two Tensor slices go through two different Linear
c:add(nn.Linear(10, 3))     -- Layers in Parallel
c:add(nn.Linear(10, 7))
mlp:add(c)                  -- Outputing a table with 2 elements
p = nn.ParallelTable()      -- These tables go through two more linear layers separately
p:add(nn.Linear(3, 2))
p:add(nn.Linear(7, 1))
mlp:add(p)
mlp:add(nn.JoinTable(1))    -- Finally, the tables are joined together and output.

pred = mlp:forward(torch.randn(10, 2))
print(pred)

for i = 1, 100 do           -- A few steps of training such a network..
   x = torch.ones(10, 2)
   y = torch.Tensor(3)
   y:copy(x:select(2, 1):narrow(1, 1, 3))
   pred = mlp:forward(x)

   criterion = nn.MSECriterion()
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(0.05)

   print(err)
end