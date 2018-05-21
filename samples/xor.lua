package.path = ";C:\\Users\\Alexis\\Downloads\\ZeroBraneStudio\\myprograms\\Brain\\?.lua"..package.path

local Brain = require "brain"

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

-- total iterations: 3808
print(xor:activate({0, 0})[1]) -- 0.063153811543313 
print(xor:activate({0, 1})[1]) -- 0.98354198210035
print(xor:activate({1, 0})[1]) -- 0.98347553558534
print(xor:activate({1, 1})[1]) -- 0.10383346250412

-- How about saving the state of our Network 
print(xor:save())

-- Now just load that thing again

xor = Brain.Network.load("[[{\"bias\":5.3522729772215,\"inputs\":[{\"weight\":-7.7988682187362},{\"weight\":-7.8065557100085}],\"activation\":\"sigmoid\",\"rate\":0.2},{\"bias\":7.1335769876748,\"inputs\":[{\"weight\":-9.570430268427},{\"weight\":-9.5778519512253}],\"activation\":\"sigmoid\",\"rate\":0.2},{\"bias\":110.82004178712,\"inputs\":[{\"weight\":-56.357684671178},{\"weight\":-54.289938877402}],\"activation\":\"sigmoid\",\"rate\":0.2}],[{\"bias\":-10.27463310763,\"inputs\":[{\"weight\":-3.4509875601262},{\"weight\":-3.943722716831},{\"weight\":14.952981918006}],\"activation\":\"sigmoid\",\"rate\":0.2}]]")


-- Stil the same! (w/ slight changes, but still!)
print(xor:activate({0, 0})[1]) -- 0.063153811543313 -> 0.063153811543275 
print(xor:activate({0, 1})[1]) -- 0.98354198210035 -> 0.98354198210034
print(xor:activate({1, 0})[1]) -- 0.98347553558534 -> 0.98347553558533
print(xor:activate({1, 1})[1]) -- 0.10383346250412 -> 0.10383346250469
