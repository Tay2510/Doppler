
-- Doppler lua/torch script
-- Modifed by Chao-Ming Yen from Kai Sheng Tai's github repo "neuralart"
-- =====================================================================
--torch.setdefaulttensortype('torch.FloatTensor')

-- some globals

model = ""
img_content = ""
img_style = ""
timer = torch.Timer()

style_weights = {
        ['conv1_1'] = 1,
        ['conv2_1'] = 1,
        ['conv3_1'] = 1,
        ['conv4_1'] = 1,
        ['conv5_1'] = 1,
    }

    content_weights = {
        ['conv4_2'] = 1,
}

function loadModel(weights_path)
    print('[Torch]: loading weight tensor...')
    timer:reset()
    local weights = torch.load(weights_path)
    print('[Torch]: loading weight tensor takes '.. timer:time().real .. ' seconds')

    print('[Torch]: creating model...')
    timer:reset()

    -- create model template
    model = nn.Sequential()
        :add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialMaxPooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialMaxPooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialMaxPooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialMaxPooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true))

    -- load weights

    for i, module in ipairs(model:listModules()) do
        if module.weight then module.weight:copy(weights[i][1]) end
        if module.bias then module.bias:copy(weights[i][2]) end
    end

    print('[Torch]: Create model takes ' .. timer:time().real .. ' seconds')

    --collectgarbage()
    model:float()

    -- load imageContent
    img_content = torch.FloatTensor(3,256,256)
    img_content = preprocess(img_content, 256)

    print('[Torch]: Forward content image...')
    timer:reset()
    model(img_content)
    print('[Torch]: Forward content takes ' .. timer:time().real .. ' seconds')
end

-- Utilities for modules
---------------------------------------------------------------
function nn.Module:name(name)
    self._name = name
    return self
end

function nn.Module:findByName(name)
    if self._name == name then return self end
    if self.modules ~= nil then
        for i = 1, #self.modules do
            local module = self.modules[i]:findByName(name)
            if module ~= nil then return module end
        end
    end
end

-- returns a list of modules
function nn.Module:listModules()
   local function tinsert(to, from)
        if torch.typename(from) == nil then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end

   end
   -- include self first
   local modules = {self}
   if self.modules then
      for i=1,#self.modules do
         local modulas = self.modules[i]:listModules()
         if modulas then
            tinsert(modules,modulas)
         end
      end
   end
   return modules
end
---------------------------------------------------------------


--
-- Convenience functions to replicate Caffe preprocessing
---------------------------------------------------------------
local means = { 104, 117, 123 }

function preprocess(img, scale)
    -- handle monochrome images
    if img:size(1) == 1 then
        local copy = torch.FloatTensor(3, img:size(2), img:size(3))
        copy[1] = img[1]
        copy[2] = img[1]
        copy[3] = img[1]
        img = copy
    elseif img:size(1) == 4 then
        img = img[{{1,3},{},{}}]
    end

    local w, h = img:size(3), img:size(2)
    if scale then
        if w < h then
            img = image.scale(img, scale * w / h, scale)
        else
            img = image.scale(img, scale, scale * h / w)
        end
    end

    -- reverse channels
    local copy = torch.FloatTensor(img:size())
    copy[1] = img[3]
    copy[2] = img[2]
    copy[3] = img[1]
    img = copy

    img:mul(255)
    for i = 1, 3 do
        img[i]:add(-means[i])
    end

    return img:view(1, 3, img:size(2), img:size(3))
end

function depreprocess(img)
    local copy = torch.FloatTensor(3, img:size(3), img:size(4)):copy(img)
    for i = 1, 3 do
        copy[i]:add(means[i])
    end
    copy:div(255)

    -- reverse channels
    local copy2 = torch.FloatTensor(copy:size())
    copy2[1] = copy[3]
    copy2[2] = copy[2]
    copy2[3] = copy[1]
    copy2:clamp(0, 1)
    return copy2
end
---------------------------------------------------------------

--
-- Cost functions
---------------------------------------------------------------
-- compute the Gramian matrix for input
function gram(input)
    local k = input:size(2)
    local flat = input:view(k, -1)
    local gram = torch.mm(flat, flat:t())
    return gram
end

function collect_activations(model, activation_layers, gram_layers)
    local activations, grams = {}, {}
    for i, module in ipairs(model.modules) do
        local name = module._name
        if name then
            if activation_layers[name] then
                local activation = module.output.new()
                activation:resize(module.output:nElement())
                activation:copy(module.output)
                activations[name] = activation
            end

            if gram_layers[name] then
                grams[name] = gram(module.output):view(-1)
            end
        end
    end
    return activations, grams
end

--
-- gradient computation functions
---------------------------------------------------------------
local euclidean = nn.MSECriterion()
euclidean.sizeAverage = false
if opt.cpu then
    euclidean:float()
else
    euclidean:cuda()
end

function style_grad(gen, orig_gram)
    local k = gen:size(2)
    local size = gen:nElement()
    local size_sq = size * size
    local gen_gram = gram(gen)
    local gen_gram_flat = gen_gram:view(-1)
    local loss = euclidean:forward(gen_gram_flat, orig_gram)
    local grad = euclidean:backward(gen_gram_flat, orig_gram)
                          :view(gen_gram:size())

    -- normalization helps improve the appearance of the generated image
    local norm = size_sq
    if opt.model == 'inception' then
        norm = torch.abs(grad):mean() * size_sq
    else
        norm = size_sq
    end
    if norm > 0 then
        loss = loss / norm
        grad:div(norm)
    end
    grad = torch.mm(grad, gen:view(k, -1)):view(gen:size())
    return loss, grad
end

function content_grad(gen, orig)
    local gen_flat = gen:view(-1)
    local loss = euclidean:forward(gen_flat, orig)
    local grad = euclidean:backward(gen_flat, orig):view(gen:size())
    if opt.model == 'inception' then
        local norm = torch.abs(grad):mean()
        if norm > 0 then
            loss = loss / norm
            grad:div(norm)
        end
    end
    return loss, grad
end

-- total variation gradient
function total_var_grad(gen)
    local x_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {1, -2}, {2, -1}}]
    local y_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {2, -1}, {1, -2}}]
    local grad = gen.new():resize(gen:size()):zero()
    grad[{{}, {}, {1, -2}, {1, -2}}]:add(x_diff):add(y_diff)
    grad[{{}, {}, {1, -2}, {2, -1}}]:add(-1, x_diff)
    grad[{{}, {}, {2, -1} ,{1, -2}}]:add(-1, y_diff)
    return grad
end
---------------------------------------------------------------



--[[
function classifyExample(tensorInput)
    v = model(tensorInput)
	print(v[1])
	return v[1]
end
]]
