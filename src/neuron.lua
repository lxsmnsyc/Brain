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

local json = require "json"

local rand, exp, tanh = math.random, math.exp, math.tanh

function random()
  return rand() * 0.2
end

local CONST = {
  learningRate = 0.2,
}

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
    return d and 1 or ( (x > 0) and 1 or 0)
  end,
  relu = function (x, d)
    if d then
      return (x > 0) and 1 or 0
    end
    return (x > 0) and x or 0
  end
}

local Neuron = {}
Neuron.__index = Neuron

function Neuron.new(data)
  data = data or {}
  local obj = {
    bias = data.bias or random(),
    rate = data.rate or CONST.learningRate,
    activation = data.activation or "sigmoid",
    inputs = {}
  }
  setmetatable(obj, Neuron)
  return obj
end

local function getWeightInitial(n)
  return math.sqrt(6/(n + 1))
end

function Neuron:addInput(weight)
  local i = self.inputs
  i[#i + 1] = {}
  
  -- adjust all weights
  for x = 1, #i do
    i[x].weight = random() * getWeightInitial(#i) - getWeightInitial(#i)
  end
end

function Neuron:activate(inputs)
  local z = 0
  
  for i = 1, #inputs do
    if self.inputs[i] == nil then
      self:addInput()
    end
    
    z = z + self.inputs[i].weight*inputs[i]
  end
  z = z + self.bias
  return ACT[self.activation](z, false), z
end

function Neuron:propagate(inputs, output)
  local prediction, z = self:activate(inputs)
  
  local err = (prediction - output)^2
  
  local delta = err * ACT[self.activation](z, true)
  
  for i = 1, #self.inputs do
    self.inputs[i].weight = self.inputs[i].weight - self.rate * dcost_dz * inputs[i]
  end
  
  self.bias = self.bias - self.rate * dcost_dz
end


function Neuron:save()
  return json.encode(self)
end


function Neuron.load(data)
  local obj = json.decode(data)
  setmetatable(obj, Neuron)
  return obj
end
return Neuron