

--torch.setdefaulttensortype('torch.FloatTensor')

-- some globals
model = ""
timer = torch.Timer()

function loadModel(weights_path)
    print('Torch: loading weight tensor...')
    timer:reset()
    local weights = torch.load(weights_path)
    print(timer:time().real .. ' seconds')

    print('Torch: creating model...')
    timer:reset()

    -- create model template
    model = nn.Sequential()
        :add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv1_1'))
        :add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv1_2'))
        :add(nn.SpatialAveragePooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv2_1'))
        :add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv2_2'))
        :add(nn.SpatialAveragePooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv3_1'))
        :add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv3_2'))
        :add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv3_3'))
        :add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv3_4'))
        :add(nn.SpatialAveragePooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv4_1'))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv4_2'))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv4_3'))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv4_4'))
        :add(nn.SpatialAveragePooling(2, 2, 2, 2))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv5_1'))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv5_2'))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv5_3'))
        :add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
        :add(nn.ReLU(true):name('conv5_4'))

    -- load weights

    for i, module in ipairs(model:listModules()) do
        if module.weight then module.weight:copy(weights[i][1]) end
        if module.bias then module.bias:copy(weights[i][2]) end
    end

    print(timer:time().real .. ' seconds')

    collectgarbage()
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


-- add missing function in torch-ios
---------------------------------------------------------------
local SpatialAveragePooling, parent = torch.class('nn.SpatialAveragePooling', 'nn.Module')

function SpatialAveragePooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.padW = padW or 0
   self.padH = padH or 0
   self.ceil_mode = false
   self.count_include_pad = true
   self.divide = true
end

function SpatialAveragePooling:ceil()
   self.ceil_mode = true
   return self
end

function SpatialAveragePooling:floor()
   self.ceil_mode = false
   return self
end

function SpatialAveragePooling:setCountIncludePad()
   self.count_include_pad = true
   return self
end

function SpatialAveragePooling:setCountExcludePad()
   self.count_include_pad = false
   return self
end

local function backwardCompatible(self)
   if self.ceil_mode == nil then
      self.ceil_mode = false
      self.count_include_pad = true
      self.padH = 0
      self.padW = 0
   end
end

function SpatialAveragePooling:updateOutput(input)
   backwardCompatible(self)
   input.THNN.SpatialAveragePooling_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      self.ceil_mode,
      self.count_include_pad
   )
   -- for backward compatibility with saved models
   -- which are not supposed to have "divide" field
   if not self.divide then
     self.output:mul(self.kW*self.kH)
   end
   return self.output
end

function SpatialAveragePooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.THNN.SpatialAveragePooling_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH,
         self.ceil_mode,
         self.count_include_pad
      )
      -- for backward compatibility
      if not self.divide then
         self.gradInput:mul(self.kW*self.kH)
      end
      return self.gradInput
   end
end
---------------------------------------------------------------

--[[
function classifyExample(tensorInput)
    v = model(tensorInput)
	print(v[1])
	return v[1]
end
]]
