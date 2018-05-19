--[[

Copyright 2018 Alexis Munsayac (@lxsmnsyc)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

]]--
local Neuron = require "neuron"
local json = require "../lib/json"

local exp = math.exp

-- our Activation functions
local ACT = {
  sigmoid = function(x, d)
    local fx = 1/(1 + exp(-x))
    if(not d) then
     return fx 
    end
    return fx * (1 - fx)
  end,
  tanh = function (x, d)
    if(d) then
      return 1 - tanh(x)^2
    end
    return tanh(x)
  end,
  identity = function (x, d)
    return d and 1 or x
  end,
  hlim = function (x, d)
    return d and 1 or ( (x > 0.0) and 1 or 0)
  end,
  relu = function (x, d)
    if d then
      return (x > 0.0) and 1 or 0
    end
    return (x > 0.0) and x or 0
  end
}
-- for generating neuron initial weight
local function getWeightInitial(net)
  return math.sqrt(6/(#net[1] + #net[#net]))
end

local Network = {}
Network.__index = Network

function Network.new(scheme, config)
  local obj = {}
  
  config = config or {}
  for x = 1, #scheme do
    -- build layers
    obj[x] = {}
    for y = 1, scheme[x] do
      -- add neuron to layer
      config[x] = config[x] or {}
      obj[x][y] = Neuron.new(config[x])
    end
  end
  
  -- Configure Global weight
  
  for x = 2, #scheme do
    for y = 1, scheme[x] do
      for i = 1, scheme[x - 1] do
        obj[x][y]:addInput(math.random() * getWeightInitial(obj) - getWeightInitial(obj))
      end
    end
  end
  
  setmetatable(obj, Network)
  return obj
end

function Network:activate(input)
  local input2
  local input2d
  for x = 1, #self do
    input2 = {}
    input2d = {}
    for y = 1, #self[x] do
      local pred, z = self[x][y]:activate(input)
      input2[y], input2d[y] = pred, z
    end
    input = input2
  end
  return input2, input2d
end

function Network:propagate(input, output)
  local inputs = {}
  
  local outputs = {}
  local doutputs = {}
  
  for x = 1, #self do
    inputs[x] = {}
  end
  for i = 1, #input do
    inputs[1][i] = input[i]
  end
  
  for x = 1, #self do
    outputs[x] = {}
    doutputs[x] = {}
    for y = 1, #self[x] do
      outputs[x][y], doutputs[x][y] = self[x][y]:activate(inputs[x])
      if(x + 1 <= #self) then
        inputs[x + 1][y] = outputs[x][y]
      end
    end
  end
  
  -- get error factor of outputs
  
  
  local errors = {}
  errors[#self] = {}
  for i = 1, #self[#self] do
    errors[#self][i] = (output[i] - outputs[#self][i])
  end
  
  -- get deltas
  local deltas = {}
  deltas[#self] = {}
  
  local t_err = 0
  for i = 1, #self[#self] do
    deltas[#self][i] = outputs[#self][i]*errors[#self][i]
  end
  
  -- get errors and deltas backwards
  
  for x = #self - 1, 1, -1 do
    errors[x] = {}
    deltas[x] = {}
    for y = 1, #self[x] do
      errors[x][y] = 0
      for i = 1, #self[x + 1] do
        errors[x][y] = errors[x][y] + deltas[x + 1][i]*self[x + 1][i].inputs[y].weight
      end
      deltas[x][y] = outputs[x][y]*errors[x][y]
    end
  end
  
  for x = #self, 1, -1 do
    for y = 1, #self[x] do
      self[x][y].bias = self[x][y].bias + self[x][y].rate*deltas[x][y]
      
      for i = 1, #self[x][y].inputs do
        self[x][y].inputs[i].weight = self[x][y].inputs[i].weight + self[x][y].rate * inputs[x][i] * deltas[x][y]
      end
    end
  end
end




local function table_to_string(tbl)
    local result = "{"
    for k, v in pairs(tbl) do
        -- Check the key type (ignore any numerical keys - assume its an array)
        if type(k) == "string" then
            result = result.."[\""..k.."\"]".."="
        end

        -- Check the value type
        if type(v) == "table" then
            result = result..table_to_string(v)
        elseif type(v) == "boolean" then
            result = result..tostring(v)
        else
            result = result.."\""..v.."\""
        end
        result = result..","
    end
    -- Remove leading commas from the result
    if result ~= "" then
        result = result:sub(1, result:len()-1)
    end
    return result.."}"
end

function Network:save()
  return json.encode(self)
end


function Network.load(data)
  local obj = json.decode(data)
  setmetatable(obj, Network)
  
  for x = 1, #obj do
    for y = 1, #obj[x] do
      setmetatable(obj[x][y], Neuron)
    end
  end
  return obj
end
return Network