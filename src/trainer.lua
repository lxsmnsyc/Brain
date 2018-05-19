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
local Network = require "network"

local Trainer = {}
Trainer.__index = Trainer 

function Trainer.new(network, data, precision)
  local obj = {
    network = network,
    data = data,
    precision = precision
  }
  setmetatable(obj, Trainer)
  return obj
end

function Trainer:train()
  
  local iterations = 0
  while(true) do
    -- check if current activation is now approaching the precision of outputs
    
    local total = 0
    local count = 0
    for i = 1, #self.data do
      local outputs = self.network:activate(self.data[i].inputs)
	--print(table.unpack(self.data[i].inputs))
     -- print(table.unpack(outputs))
      for j = 1, #outputs do
        -- approximate value
        local actual = self.data[i].outputs[j]
        local prediction = outputs[j]
        
        total = total + math.abs(prediction - actual)
        count = count + 1
      end
    end
    
    
    
    if(total/count <= self.precision) then break end
    
    iterations = iterations + 1
    
   -- print(iterations)
    
    for i = 1, #self.data do
      self.network:propagate(self.data[i].inputs, self.data[i].outputs)
    end
  end
  return iterations
end

return Trainer