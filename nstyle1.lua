-- This script is based on ORIGINAL_VERSION
-- Here I' ll make stride method work.
-- ******************* GUARANTEE VERSION ******
-- Does not work at all!

require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'lib/utils'

require 'loadcaffe'
require 'io'



local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/picasso_selfport1907.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 256, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'adam', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 5)
cmd:option('-output_image', 'output/out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'cudnn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

-- Added by Yan
cmd:option('-cluster_nums',{200})
cmd:option('-cluster_ratio',{0.03},'Prior to cluster_num,representing the ratio of cluster number to all pixels in certain style layer')
cmd:option('-cluster_method','kmeans','kmenas|others')
cmd:option('-padding',3,'padding')
cmd:option('-data_generate',1,'Whether to regenerate new data')
cmd:option('-stride',3,'stride of calculate grams in features')





cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu4_1', 'layers for style')
--cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

local function main(params)
  if params.gpu >= 0 then
	if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
  
  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end

  -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end
  

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      content_image_caffe = content_image_caffe:cuda()
      for i = 1, #style_images_caffe do
        style_images_caffe[i] = style_images_caffe[i]:cuda()
      end
    else
      content_image_caffe = content_image_caffe:cl()
      for i = 1, #style_images_caffe do
        style_images_caffe[i] = style_images_caffe[i]:cl()
      end
    end
  end

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
-- Added 1 by Yan
  local nS = #style_layers
  local nP = #style_images_caffe
	local pad = params.padding
-- kernels_tb[1][1] means the first layer and the first kernel,
-- its cluster gram is gram_tb[1][1].
	local kernels_tb = {}
  local grams_tb = {}
	local counts_tb = {}
	local labels_tb = {}
  local pointed_tb = {}  -- 0 means the it's a non-pointed cluster

	local cnn_features = extract_features(cnn, style_images_caffe, style_layers,params)
  local clu_nums_tb = {}

  if params.cluster_ratio ~= 'nil' then
    local tmp = 0
    for i = 1, nS do
      tmp = 0
      for j = 1, nP do
        tmp = tmp + cnn_features[i][j]:size(2) * cnn_features[i][j]:size(3)
      end
      clu_nums_tb[i] = math.floor(tmp*params.cluster_ratio[i])
    end
  else
    clu_nums_tb = params.cluster_num
  end

  if params.data_generate == 0 then
    -- diretely load data that has generated
		local data = torch.load('data/cluster_data.dat')
		if #data == 5 then
			print('Loading data...')
			kernels_tb = data[1]
      grams_tb = data[2]
			counts_tb = data[3]
			labels_tb = data[4]
      pointed_tb = data[5]
		else
			error('Data has been damaged, Use --data_generate 1 to generate new data')
			return
		end
  else
    -- write important data into disk, so that python script can read.
    npy4th = require 'npy4th'
    for i = 1, nS do
      for j = 1, nP do
        npy4th.savenpy(string.format('data/s%d_p%d.npy',i,j),cnn_features[i][j])  
      end
		end
    -- deal with clu_nums_tb parsing into python
    local clu_nums_str = ''
    for i = 1,#clu_nums_tb do 
      clu_nums_str = clu_nums_str..','..tostring(clu_nums_tb[i])
    end
    clu_nums_str = string.sub(clu_nums_str,2)
    print(clu_nums_str)
    command = string.format("python 'test.py' --cluster_nums %s --style_layers %d --image_num %d --cluster_method %s", clu_nums_str, nS, nP, params.cluster_method)
    os.execute(command)
		-- Load data that processed by python sklearn
		for i = 1,nS do
			-- No use now, kernels are just for debuging here
			local kernels = npy4th.loadnpy(string.format('data/kernels_%d.npy',i))
			local counts = npy4th.loadnpy(string.format('data/counts_%d.npy',i))		
			local labels = npy4th.loadnpy(string.format('data/labels_%d.npy',i))
      local pointed = torch.Tensor(clu_nums_tb[i]):zero()

			-- calculate the gram value of each points
			local cur_pic_idx = 1 -- Index of the first pixel of current image in all image pixels
			local channel = cnn_features[i][1]:size(1)
			local grams = torch.Tensor(clu_nums_tb[i], channel, channel):zero():cuda() 
			local clu_counts = torch.Tensor(clu_nums_tb[i]):zero()  -- for debuging
			local ex_features = nil
			for p_idx = 1,nP do
				ex_features = ex_tensor(cnn_features[i][p_idx], pad)
				local pic = cnn_features[i][p_idx]
				local height = pic:size(2)  -- Just for debuging
				local width = pic:size(3) 
				local half_pad = math.floor(pad/2)

				for bias = 1, height*width do
					local m, n = single_index2_double(bias, width)
					print('m:'..m..',n:'..n)
					local m_in_ex = m + half_pad
					local n_in_ex = n + half_pad
					local mn_patch = ex_features[{{},{m_in_ex - half_pad, m_in_ex + half_pad}, {n_in_ex - half_pad, n_in_ex + half_pad}}]:clone()

					local gram_value = GramMatrix():cuda():forward(mn_patch):clone()
					-- find the (m,n) point belongs to which cluster
					local cluster = labels[cur_pic_idx + bias - 1]
          pointed[cluster] = 1
					--print(string.format('style:%d\tpic:%d\tm:%d\tn:%d\tcluster:%d\n',i,p_idx,m,n,cluster))

					grams[cluster]:add(gram_value)

					clu_counts[cluster] = clu_counts[cluster]+1
				end
				cur_pic_idx = cur_pic_idx + height*width
			end -- end of 1:nP

			for k = 1,clu_nums_tb[i] do
				assert(counts[k] == clu_counts[k],'Two ways of counting the number of points in clusters are not equal!')
			end
			-- calculate the average gram value for each cluster
			grams = torch.cdiv(grams, counts:view(clu_nums_tb[i],1,1):expand(clu_nums_tb[i],channel,channel):cuda())

			table.insert(kernels_tb, kernels)
			table.insert(grams_tb, grams)
			table.insert(counts_tb, counts)
			table.insert(labels_tb, labels)
      table.insert(pointed_tb,pointed)

		end -- end of 1:nS
		
		-- write important clustered data into disk for later use.
		torch.save('data/cluster_data.dat',{kernels_tb,  grams_tb, counts_tb, labels_tb, pointed_tb})		
  end -- end of params.data_generate
  -- clear garbage
	local counts_tb = nil
	local labels_tb = nil
  collectgarbage()
-- End of YAN 1

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
  end
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer:cuda()
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(content_image_caffe):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        loss_module:cuda()
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local norm = params.normalize_gradients
        -- Target is calculated in StyleLoss inside!
        -- It seems No good ways to use the style_blend_weights Now!
        -- Maybe style_blend_weights can be add into calculating grams_tb.
        -- It is similar to original version and it does make sense.
        -- params in: grams_tb[i] and kernels_tb[i]
        local loss_module = nn.StyleLoss(params.style_weight, grams_tb[next_style_idx], kernels_tb[next_style_idx], pointed_tb[next_style_idx], params, norm):float()
        loss_module:cuda()
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  
  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      img = img:cuda()
    else
      img = img:cl()
    end
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
      end

      -- Maybe perform postprocessing for color-independent style transfer
      if params.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end

      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this function many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end
  

function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.rgb2yuv(content)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, grams_value, kernels, pointed, params, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.grams_value = grams_value
  self.kernels = kernels
  self.stride = params.stride
  self.padding = params.padding
  self.pointed = pointed
  self.loss = 0
  self.target = nil --(H*W)*C*C target grams_value
  self.ex_input = nil
  self.c = 0
  self.h = 0
  self.w = 0
	self.point_num = 0 --number of sample points
  
  self.gram = GramMatrix()
  self.G = nil  --(H*W)*C*C  input grams_value
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)

  -- reshape C*H*W to C*(H*W) and calculate grams for each pixels
  -- note: torch and numpy stores tensors in C style, so there is no big problem.
  self.c = (#input)[1]
  self.h = (#input)[2]
  self.w = (#input)[3]
  local zero_grams = torch.Tensor(self.grams_value:size()):typeAs(input):zero()
  local half_pad = math.floor(self.padding/2)
  self.ex_input = ex_tensor(input, self.padding)

	-- When using stride, the sizes of target and G should be changed.
	self.point_num = (math.floor((self.w-1)/self.stride)+1)*(math.floor((self.h-1)/self.stride)+1)
  self.target = torch.Tensor(self.point_num, self.c, self.c):typeAs(input):zero()
  self.G = torch.Tensor(self.target:size()):zero():cuda()
 -- local file  = io.open('log.txt','r')

  --for bias = 1, self.h*self.w, self.stride do
  for bias_i = 1, self.h, self.stride do
    for bias_j = 1, self.w, self.stride do
			local bias = double_index2_single(bias_i,bias_j,self.w)

			-- local m, n = single_index2_double(bias, self.w)
			print('bias_i:'..bias_i..',bias_j:'..bias_j)
			local m_in_ex = bias_i + half_pad
			local n_in_ex = bias_j + half_pad
			local ex_input = self.ex_input
			local point = ex_input[{{},{m_in_ex},{n_in_ex}}]:clone()
			local mn_patch = ex_input[{{},{m_in_ex - half_pad, m_in_ex + half_pad}, {n_in_ex - half_pad, n_in_ex + half_pad}}]:clone()
      local sample_index = index_to_count(bias, self.stride, self.w)
			self.target[{{sample_index},{},{}}] = self.gram:forward(mn_patch):clone()
			-- Just for debuging
			local f2 = io.open('log2.txt','a+')
			f2:write(bias..':\n')
			for i = 1, ex_input:size(1) do
				local e = point[{{i},{},{}}]:squeeze()
				--print(torch.type(e))
				f2:write(math.floor(e)..'\t')
			end
			f2:write('*********************\n\n\n')
			f2:close()

			-- find the (m,n) point belongs to which cluster
			-- local cluster = labels[cur_pic_idx + bias - 1]
			-- We use distance to measure it.
			local order = find_nearest_cluster(self.kernels, point)
			--check selected cluster not non-pointed.
			-- How to optimize?
			local clu_gram = torch.Tensor():cuda()
			print('self.pointed')
			print(order)
			-- print(self.pointed)
			for i = 1,(#order)[1] do
				if self.pointed[i] == 1 then
						local f3 = io.open('log3.txt','a+')
						f3:write(bias..': '..order[i][1]..'\n')
						if bias == self.w*self.h then
							f3:write('*************\n')
						end
						f3:close()
						print(string.format('cluster:%d',order[i][1]))
						clu_gram = self.grams_value[{{order[i][1]},{},{}}]:clone():cuda()
						print('a')
						break
				else
						assert(1==2,'Try to set bigger cluster number.')
				end
			end
			self.G[sample_index] = clu_gram

    end
  end
 -- file:close()
  -- calculate Loss between target_gram and input_gram
  self.G:div(input:nElement())
  self.target:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

--[[The main difficulty lies in here: how to transfer the grad_patch
    to gradInput. Region of size PAD*PAD should be squeeze backward to 1*1.
    So that, We get back_grad_patch:(H*W)*C*1, reshape and permute it to C*H*W.
  ]]
function StyleLoss:updateGradInput(input, gradOutput)
  self.gradInput = torch.Tensor(input:size()):typeAs(input):zero() --C*H*W
  local ex_gradInput = ex_tensor(self.gradInput, self.padding)
  local grad_patch = torch.Tensor(self.point_num, self.c, self.padding, self.padding):cuda()--(H*W)*C*PAD*PAD
  local dG = self.crit:backward(self.G, self.target) --(h*W)*C*C
  dG:div(input:nElement())
  --self.ex_input = ex_tensor(input, self.padding)  -- maybe make it self.ex_input if memory allows
  local ex_input = self.ex_input
  local half_pad = math.floor(self.padding/2)
	local bias = 0
-- for bias = 1, self.h*self.w, self.stride do
	for bias_i = 1,self.h, self.stride do
		for bias_j = 1, self.w, self.stride do
			bias = double_index2_single(bias_i, bias_j, self.w)
		--	local m, n = single_index2_double(bias, self.w)
		--  print('m:'..m..',n:'..n)
			local m_in_ex = bias_i + half_pad
			local n_in_ex = bias_j + half_pad
			local point = ex_input[{{},{m_in_ex},{n_in_ex}}]
			local mn_patch = ex_input[{{},{m_in_ex - half_pad, m_in_ex + half_pad}, {n_in_ex - half_pad, n_in_ex + half_pad}}]:contiguous()
		--	local dG_bias = dG[{ {bias},{},{} }]:squeeze():clone()
      local sample_index = index_to_count(bias, self.stride, self.w)
			grad_patch[{{sample_index},{},{},{}}] = self.gram:backward(mn_patch, dG[sample_index]):clone() -- C*PAD*PAD
      print('ex_h:'..(#self.ex_input)[2]..' ex:w'..(#self.ex_input)[3])
      print('m_in_ex:'..m_in_ex..' n_in_ex:'..n_in_ex..' sample_index:'..sample_index)
			ex_gradInput[{{},{m_in_ex - half_pad, m_in_ex + half_pad}, {n_in_ex - half_pad, n_in_ex + half_pad}}] = grad_patch[{{sample_index},{},{},{}}]:clone()
		end
  end
  local x = index_to_count(bias,self.stride,self.w)
  print('x:'..x)
	assert(x == self.point_num,'Compututaion Wrong!')
  -- crop ex_gradInput to gradInput
  self.gradInput = ex_gradInput[{{},{1+half_pad, -1-half_pad},{1+half_pad, -1-half_pad}}]:clone()


  -- **AVERAGE method**,
  -- It shouldn't work. Maybe later to find a better way.
  -- Using stride x and y axies, or figure out the different weights contributed by each local pixels
  -- and calculate it back.
  -- local dG_reshape = torch.reshape(dG, torch.LongStorage(self.h, self.w, self.c, self.pading, self.padding))
  -- local grad_patch_mean = torch.mean(grad_patch:reshape(self.h*self.w, self.c, self.padding*self.padding), 3):clone()
  -- local grad_patch_reshape = grad_patch_mean:squeeze():permute(2,1):reshape(self.c, self.h, self.w):clone()  -- C*H*W

	-- Since it is no overlapping now, so just assign each point to its corresponding position.
	-- TODO


 -- self.gradInput = grad_patch_reshape:clone()

 -- self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local params = cmd:parse(arg)
main(params)
