require 'torch'
require 'nn'
require 'optim'
require 'math'


local graph = {}
for line in io.lines("../BlogCatalog-dataset/datatemp/edges.csv") do
	local u,v = line:match("([^,]+),([^,]+)")
	graph[#graph+1] = {u=torch.Tensor{u}, v=torch.Tensor{u}, w=torch.Tensor{1}}
end


local vocab = {}
for line in io.lines("../BlogCatalog-dataset/datatemp/nodes.csv") do
	local u = line
	vocab[#vocab+1] = {u=u}
end


vocab_size = #vocab
node_embed_size = 10
learning_rate = 0.01
max_epochs = 100
batch_size = 10

-- local train_data = {}
-- for i=1, #graph do
-- 	--features = {torch.Tensor{graph[i]['u']},torch.Tensor{graph[i]['v']}}
-- 	features = {graph[i]['u'],graph[i]['v']}
-- 	label = graph[i]['w']
-- 	train_data[#train_data+1] = {features,label}
-- end

local train_data = {}
local batch_count = 0
-- local num_batch = math.floor(#graph/batch_size)
-- 
-- if #graph % batch_size ~=0 then
-- 	num_batch = num_batch + 1
-- end

local i = 1
while i <= #graph do
	batch = {}
	for k=1, batch_size do
		if i>#graph then
			break
		end
		features = {graph[i]['u'],graph[i]['v']}
 		label = graph[i]['w']
 		batch[#batch+1] = {features,label}
 		i = i + 1
 	end
 	train_data[#train_data+1] = {batch}
end

function train_data:size() return #train_data end


node_lookup = nn.LookupTable(vocab_size, node_embed_size)

model = nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(node_lookup)
model.modules[1]:add(node_lookup:clone('weight', 'bias', 'gradWeight', 'gradBias'))
model:add(nn.CosineDistance())
model:add(nn.Sigmoid())

criterion = nn.MSECriterion()

print(model:forward({torch.Tensor{1},torch.Tensor{2}}))
--require 'os'
--local loss = 0
--local output = model:forward({torch.Tensor{1},torch.Tensor{2}})
--		loss = loss + criterion:forward(output, torch.Tensor{0.5})
--		local grads = criterion:backward(output, torch.Tensor{0.5})
--		print(output)
--		print(grads)
--		model:backward({torch.Tensor{1},torch.Tensor{2}}, grads)
--		os.exit(0)






trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_epochs

print('Node Lookup before learning')
print(node_lookup.weight)

params, grad_params = model:getParameters()
feval = function(x)
	-- Get new params
	params:copy(x)

	-- Reset gradients (buffers)
	grad_params:zero()

	-- loss is average of all criterions
	local loss = 0
	for i = 1, #train_data do
		local output = model:forward(train_data[i][1])
		loss = loss + criterion:forward(output, train_data[i][2])
		local grads = criterion:backward(output, train_data[i][2])
		print(output)
		model:backward(train_data[i][1], grads)
	end
	grad_params:div(#train_data)

	-- L2 regularization
	loss = loss -- + 0.5 * l2_reg * (params:norm() ^ 2)

	return loss, grad_params
end

optim_state = {learningRate = learning_rate}
print('# StochasticGradient: training')
local l = 0
for epoch = 1, max_epochs do
	
	local _, loss = optim.sgd(feval, params, optim_state)
	l = loss[1]
	print('# current error = '..l)
end


print('\nNode Lookup after learning')
print(node_lookup.weight)

