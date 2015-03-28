train_cfg = [
'iter = mnist' char(10)...
'    path_img = "./data/train-images-idx3-ubyte"' char(10)...
'    path_label = "./data/train-labels-idx1-ubyte"' char(10) ...
'    shuffle = 1' char(10)...
'iter = end' char(10)...
'input_shape = 1,1,784' char(10)...
'batch_size = 100' char(10)];


eval_cfg = [
'iter = mnist' char(10)...
'    path_img = "./data/t10k-images-idx3-ubyte"' char(10)...
'    path_label = "./data/t10k-labels-idx1-ubyte"' char(10)...
'iter = end' char(10)...
'input_shape = 1,1,784' char(10)...
'batch_size = 100' char(10)...
];


cfg = [
'netconfig=start' char(10)...
'layer[+1:fc1] = fullc:fc1' char(10)...
'  nhidden = 100' char(10)...
'  init_sigma = 0.01' char(10)...
'layer[+1:sg1] = sigmoid:se1' char(10)...
'layer[sg1->fc2] = fullc:fc2' char(10)...
'  nhidden = 10' char(10)...
'  init_sigma = 0.01' char(10)...
'layer[+0] = softmax' char(10)...
'netconfig=end' char(10)...
'input_shape = 1,1,784' char(10)...
'batch_size = 100' char(10)...
'random_type = xavier' char(10)...
'metric[label]=error' char(10)...
'eta=0.1' char(10)...
'momentum=0.9' char(10)...
];

train = DataIter(train_cfg);
eval = DataIter(eval_cfg);

net = Net('gpu', cfg)
net.init_model()

% train 1 epoch
train.before_first();
while train.next() == 1,
  net.update(train);
end
net.evaluate(eval, 'eval')
train.before_first();
eval.before_first();

w1 = net.get_weight('fc1', 'wmat');
b1 = net.get_weight('fc1', 'bias');
w2 = net.get_weight('fc2', 'wmat');
b2 = net.get_weight('fc2', 'bias');

% train second epoch

while train.next() == 1,
  d = train.get_data();
  l = train.get_label();
  net.update(d, l);
end
net.evaluate(eval, 'eval')
eval.before_first();

% reset weight
net.set_weight(w1, 'fc1', 'wmat');
net.set_weight(b1, 'fc1', 'bias');
net.set_weight(w2, 'fc2', 'wmat');
net.set_weight(b2, 'fc2', 'bias');
net.evaluate(eval, 'eval')

delete(net);
delete(train);
delete(eval);
