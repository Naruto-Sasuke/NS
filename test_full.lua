require 'nn'
require 'torch'
require 'cutorch'
require 'lib/utils'

local input = torch.Tensor(2,8,9)
local padding = 3
print(getAverageTemplate(input, padding))

