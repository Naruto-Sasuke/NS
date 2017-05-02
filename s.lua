require "nn"
require "torch"
require 'lib/MaxCoord'


local input = torch.randn(2,5,5):mul(4):floor()
print('input:')
print(input)
local net = nn.Sequential()
local mlp = nn.SpatialConvolution(2,3,3,3):noBias()

local w = torch.randn(3,2,3,3):mul(2):floor()
w1 = w:mul(1/(torch.norm(w,2)+1e-8))
print('w:')
print(w1)
mlp.weight = w1:clone()
net:add(mlp)
print('before MaxCoord..')
local res = net:forward(input)
print(res)
net:add(nn.MaxCoord())

print(net)
print('Kbar:')
local Kbar  = net:forward(input)
print(Kbar)

-- FullConvolution
local bconv = nn.SpatialFullConvolution(3,2,3,3):noBias()
bconv.weight = w:clone()

print(bconv)

print(('backconv'))
print(bconv:forward(Kbar))
print('input')
print(input)