require 'lib/utils'
require 'nn'
require 'torch'
-- This script is a test for a new way to calculate gram value.
-- Only for those whose height equals width works.


local width = 5
local stride = 3
local index = 9
for index = 1,25 do
    local idx = index_to_count(index, stride, width)
    print('index: '..index..', idx: '..idx)
end