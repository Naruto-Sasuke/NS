require 'io'    
require 'lib/utils'
local nS = 10
local clu_nums = {50,34}
local strs = ''
print(type(clu_nums[1]))
for i = 1, #clu_nums do
    --print(tostring(clu_nums[i]))
    strs = strs..','..tostring(clu_nums[i])
end
strs = string.sub(strs,2)
print(strs)
command = string.format("python 'test_arg.py' --cluster_num %s --style_layers %d", strs, nS)
os.execute(command)