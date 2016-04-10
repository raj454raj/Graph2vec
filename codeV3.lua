require 'torch'
require 'nn'
require 'optim'
require 'math'
require 'os'
require 'sys'

local graph = {}
for line in io.lines("../BlogCatalog-dataset/datatemp/edges.csv") do
	local u,v = line:match("([^,]+),([^,]+)")
	graph[#graph+1] = {u=torch.Tensor{u}, v=torch.Tensor{u}}
end

local vocab = {}
for line in io.lines("../BlogCatalog-dataset/datatemp/nodes.csv") do
	local u = line
	vocab[#vocab+1] = {u=torch.Tensor{u}}
end


Out_Degree = {}
for i=1, #vocab do
	Out_Degree[i] = 0
end

for i=1, #graph do
	if i==1 then
		prev_u = graph[i]['u'][1]
		count = 1
	else
		if graph[i]['u'][1] == prev_u then
			count = count + 1
		else
			Out_Degree[prev_u] = count
			prev_u = graph[i]['u'][1]
			count = 1
		end
	end
end
Out_Degree[prev_u] = count


for i=1, #graph do
	graph[i]['w'] = torch.Tensor{1/#graph}
	graph[i]['s'] = torch.Tensor{1/Out_Degree[graph[i]['u'][1]]}
end



vocab_size = #vocab
node_embed_size = 5
learning_rate = 0.01
max_epochs = 5
batch_size = 10


-- #############################################################################################################

train_data = {}
batch_count = 0

local i = 1
while i <= #graph do
	batch = {}
	for k=1, batch_size do
		if i>#graph then
			break
		end
		features = {graph[i]['u'],graph[i]['v']}
 		label = graph[i]['w']
 		batch[k] = {features,label}
 		i = i + 1
 	end
 	train_data[#train_data+1] = batch
end

function train_data:size() return #train_data end


node_lookup = nn.LookupTable(vocab_size, node_embed_size)

model = nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(node_lookup)
model.modules[1]:add(node_lookup:clone('weight', 'bias', 'gradWeight', 'gradBias'))
model:add(nn.DotProduct())
model:add(nn.Sigmoid())

criterion = nn.MSECriterion()



edge_batch = {}
function edge_batch:size() return #edge_batch end

params, grad_params = model:getParameters()

feval = function(x)
	-- Get new params
	params:copy(x)

	-- Reset gradients (buffers)
	grad_params:zero()

	-- loss is average of all criterions
	local loss = 0
	for i = 1, #edge_batch do
		local output = model:forward(edge_batch[i][1])
		loss = loss + criterion:forward(output, edge_batch[i][2])
		local grads = criterion:backward(output, edge_batch[i][2])
		model:backward(edge_batch[i][1], grads)
	end
	grad_params:div(#edge_batch)

	return loss, grad_params
end

optim_state = {learningRate = learning_rate}
print('# StochasticGradient: training')
--local l = 0
--for epoch = 1, max_epochs do
--	for i=1, #train_data do
--		edge_batch = train_data[i]
--		local _, loss = optim.sgd(feval, params, optim_state)
--		l = loss[1]
--		print('# current error = '..l)
--	end
--end


-- print('\nNode Lookup after learning')
-- print(node_lookup.weight)

-- #############################################################################################################--
--
function build_table()
    local start = sys.clock()
    local total_count_pow = 0
    print("Building a table of unigram frequencies... ")
    for i = 1, vocab_size do
    	total_count_pow = total_count_pow + Out_Degree[i]^0.75
    end

    -- Negative samples lookup table
    negative_samples = torch.IntTensor(1e6)
    local word_index = 1
    local word_prob = Out_Degree[word_index]^0.75 / total_count_pow
    for idx = 1, 1e6 do
        negative_samples[idx] = word_index
        if idx / vocab_size > word_prob then
            word_index = word_index + 1
            if word_index > vocab_size then
            	word_index = word_index - 1
        	end
	    	word_prob = word_prob + Out_Degree[word_index]^0.75 / total_count_pow
        end
    end
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
end

-- Building neural network
node_lookup2C = nn.LookupTable(vocab_size, node_embed_size)
node_lookup2R = nn.LookupTable(vocab_size, node_embed_size)

model = nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(node_lookup2C)
model.modules[1]:add(node_lookup2R)
model:add(nn.DotProduct())
model:add(nn.LogSigmoid())

criterion = nn.MSECriterion()

