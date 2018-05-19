local Brain = require "..brain"

-- Create the network with a hidden layer with 3 neurons and an output layer with 1 neuron 
local xor = Brain.Network.new({3, 1})

-- Create a Trainer with the provided inputs and target outputs, with an average error of 5%
local xorTrainer = Brain.Trainer.new(xor,{
  {inputs = {0, 0}, outputs = {0}},
  {inputs = {0, 1}, outputs = {1}},
  {inputs = {1, 0}, outputs = {1}},
  {inputs = {1, 1}, outputs = {0}},
}, 0.05)

-- Train ( or propagate) the network
-- Prints the number of iterations
print(xorTrainer:train())


-- Activate the network by testing out the outputs 
print(xor:activate({0, 0})) -- 
print(xor:activate({0, 1})) -- 
print(xor:activate({1, 0})) --
print(xor:activate({1, 1})) -- 

-- How about saving the state of our Network 
print(xor:save())