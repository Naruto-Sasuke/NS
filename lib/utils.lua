require 'torch'
--[[
	This functon works as the extractor of cnn features.
	Return:
		cnn_features is a table with each index indicating the total points data.
		Attention: cnn_features[1] represents the data of first layer which
				   includes #images images as its value: cnn_features[1][1],
				   cnn_features[1][2],...,cnn_features[1][#images]
		Size:	   C*H*W
]]
function extract_features(cnn,images,extract_layers,params)
  local cnn_features = {}
  local next_layer_idx = 1
  local net = nn.Sequential()
  local isGPU = params.gpu

  for i = 1, #cnn do
    if next_layer_idx <= #extract_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if isGPU >= 0 then
          if params.backend ~= 'clnn' then
            avg_pool_layer:cuda()
          else
            avg_pool_layer:cl()
          end
        end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      -- extract_features here
      if name == extract_layers[next_layer_idx] then
        cnn_features[next_layer_idx] = {}
        for idx = 1, #images do
          local features = net:forward(images[idx]):clone()
            table.insert(cnn_features[next_layer_idx],features)
        end
        next_layer_idx = next_layer_idx + 1
      end
    end
  end
  return cnn_features
end


function ex_tensor(input, pad, method)
-- mirror and zero
	assert((pad-1)%2 == 0, 'pad should be odd number!')
	local padding = (pad-1)/2
	local method = method or 'mirror'
	local k = input:size()
	local output = torch.Tensor(k[1], k[2]+padding*2, k[3]+padding*2):typeAs(input):zero()
	output[{{},{padding+1, -padding-1},{padding+1, -padding-1}}] = input:clone()
	if method == 'mirror' then
		for i = 1, padding do
			output[{{},{i},{padding+1, -padding-1}}] = output[{{},{padding*2+1-i},{padding+1, -padding-1}}]  -- up
			output[{{},{-i},{padding+1, -padding-1}}] = output[{{},{-padding*2-1+i},{padding+1, -padding-1}}]  --down
			output[{{},{padding+1, -padding-1},{i}}] = output[{{},{padding+1, -padding-1},{padding*2+1-i}}]  --left
			output[{{},{padding+1, -padding-1},{-i}}] = output[{{},{padding+1, -padding-1},{-padding*2-1+i}}]  --right
		end
		for i = 1, padding do
			output[{{},{1, padding},{i}}] = output[{{},{1, padding},{padding*2+1-i}}] --left_up
			output[{{},{-padding,-1},{i}}] = output[{{},{-padding,-1},{padding*2+1-i}}]  --left_down
			output[{{},{1, padding},{-i}}] = output[{{},{1, padding},{-padding*2-1+i}}] --right_up
			output[{{},{-padding,-1},{-i}}] = output[{{},{-padding, -1},{-padding*2-1+i}}]  --right_down
		end
	else
		-- done
	end
    return output
end

--[[
  kernels : NUM*DIMENSION 
  point:    DIMENSION
  Return:
    list of indices in the goal of maximizing similarity.
  Note: sq[i] is a tensor, meaning the i-th index.
        You can get the index by sq[i][1].
]]
function find_nearest_cluster(kernels, point, method, normalized)
  local method = method or 'Euclidean'
  local kernels = kernels:double()
  local point = point:double()
  local normalized = normalized or true
  assert((#kernels)[2] == (#point)[1],'kernels dimension should be the same as that of point!')
  if normalized then  -- l2 , each line is a sample
    kernels:cdiv(torch.norm(kernels, 2, 2):expandAs(kernels) + 1e-8)
    point:div(torch.norm(point,2) + 1e-8)
  end
  local sq = torch.Tensor()
  if method == 'Euclidean' then
    local point_ex = point:view(1,-1):expandAs(kernels):clone()
    local pow2dist = torch.sum(torch.pow(kernels:add(-1,point_ex),2),2)
    _, sq = torch.sort(pow2dist,1)  -- along the row to get the sorted indices
    sq = sq:squeeze()
  elseif method == 'Cosine'  then
    local num, C = kernels:size()
    local net = nn.Sequential()
    local modle = nn.Cosine(C, num)
    modle.weight = kernels:clone():double()
    net:add(modle)
    local result = net:forward(point:squeeze():double())
    _,sq = torch.sort(result,1, true)  -- descending order
  end
  return sq
end


function gramMatrix(input)
    local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
    local vecInput = input:view(N,C,H*W)

    --知识点：
    -- 1. 块相乘
    -- 2. 对其中某两维进行转置
    -- 3. gram矩阵确实是这样实现的！！
    local gramMatrix = torch.bmm(vecInput, vecInput:transpose(2,3))
    print('vecInput')
    print(vecInput)
    local output = gramMatrix / H / W
    return output
end

function single_index2_double(index,width)
-- This function is mainly change the single index to double index in Lua
  local i = math.floor((index-1)/width)+1
  local j = index - (i-1)*width
  return i,j
end

function double_index2_single(i,j,width)
  return (i-1)*width+j
end

--[[Count the number of samples, sampling from 1, ending with index.
    cnt: the accumuting number of samples
 ]]
function index_to_count(index, stride, width)
  local count_inline = math.floor((width-1)/stride)+1
  local i = math.floor((index-1)/width)+1
  local j = index - (i-1)*width
  local idx_cur = math.floor((i-1)/stride)+1
  local cnt = (idx_cur-1)*count_inline + math.floor((j-1)/stride) + 1
  if (i-1)%stride == 0 then
    return cnt
  else
    cnt = idx_cur*count_inline
  end
  return cnt
end

-- input should be extended
--[[
-- for example:
local input = torch.Tensor(2,8,9)
local padding = 3
print(getAverageTemplate(input, padding))

(1,.,.) =
  1  2  3  3  3  3  3  2  1
  2  4  6  6  6  6  6  4  2
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  2  4  6  6  6  6  6  4  2
  1  2  3  3  3  3  3  2  1

(2,.,.) =
  1  2  3  3  3  3  3  2  1
  2  4  6  6  6  6  6  4  2
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  2  4  6  6  6  6  6  4  2
  1  2  3  3  3  3  3  2  1
[torch.DoubleTensor of size 2x8x9]
]]
function getAverageTemplate(input, padding)
  local function refleat2DSquareTensor(tensor, dim)
    local t = tensor:clone()
    local flipIdx = math.ceil(t:size(1)/2)
    if dim == 1 then
      for i = 1, flipIdx -1 do
        local tmp = t[{ {i},{} }]:clone()
        t[{ {i},{} }] = t[{ {-i},{} }]:clone()
        t[{ {-i},{} }] = tmp:clone()
      end
    else
        for i = 1, flipIdx -1 do
        local tmp = t[{ {},{i} }]:clone()
        t[{ {},{i} }] = t[{ {},{-i} }]:clone()
        t[{ {},{-i} }] = tmp:clone()
      end
    end
    return t
  end
  local sz = input:size()
  local c = sz[1]
  local h = sz[2]
  local w = sz[3]
  local output = torch.zeros(h, w):typeAs(input)
  local final = padding*padding
  -- First, construct letTen.
  local i = 0
  local letTen = torch.Tensor(padding):apply(function()i = i+1 return i end):repeatTensor(padding,1)
  for i = 2, padding do
    letTen[i] = letTen[1] * i
  end
  -- Second, fill the center with padding*padding.
  output[{{padding+1, -padding-1},{padding+1, -padding-1} }]:fill(final)

  -- Third, fill four corners with leTen and its transformations.
  local flip1letTen = refleat2DSquareTensor(letTen,1)
  local flip2letTen = refleat2DSquareTensor(letTen,2)
  local flip3letTen = refleat2DSquareTensor(flip2letTen,1)
  output[{{1, padding},{1,padding} }] = letTen:clone()  --left_up
  output[{{-padding, -1},{1,padding} }] = flip1letTen:clone() --left_down
  output[{{1, padding},{-padding, -1} }] = flip2letTen:clone()  -- right_up
  output[{{-padding, -1},{-padding,-1} }] = flip3letTen:clone()  --right_down

  --Fourth, fill four white spaces
  local colTen = output[{ {1, padding},{padding} }]:clone()
  local flipcolTen = output[{ {-padding, -1},{padding} }]:clone()
  local rowTen = output[{ {padding},{1, padding} }]:clone()
  local fliprowTen = output[{ {padding},{-padding, -1} }]:clone()
  output[{ {1, padding},{padding+1, -padding-1} }] = colTen:repeatTensor(1, w - 2*padding):clone() --up_center
  output[{ {padding+1, -padding-1},{1, padding} }] = rowTen:repeatTensor(h - 2*padding, 1):clone() --left_center
  output[{ {padding, -padding-1},{-padding, -1} }] = fliprowTen:repeatTensor(h - 2*padding+1, 1):clone() --right_center
  output[{ {-padding, -1},{padding, -padding-1} }] = flipcolTen:repeatTensor(1, w - 2*padding+1):clone() --down_center

  --fifth, expand tensor to c channels
  output = output:repeatTensor(c,1,1)
  return output
end


