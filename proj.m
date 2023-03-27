% author: Jingyu Ren
% Laurentian University
% Student ID: 0421763
% Machine Learning Project
clc;clear;

% import preprocessed data
opts = delimitedTextImportOptions("NumVariables", 4);
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["x1", "x2", "x3", "Y"];
opts.VariableTypes = ["double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
umap_train_cc = readtable("C:\Laurentian\Winter 2023\Machine Learning\Project\umap_train_cc.csv", opts);
umap_test_cc = readtable("C:\Laurentian\Winter 2023\Machine Learning\Project\umap_test_balanced_cc.csv", opts);
umap_train_cc = table2array(umap_train_cc); % training data
umap_test_cc = table2array(umap_test_cc); % real test data
clear opts

% splite the data
rng('shuffle')
rand_indices = randperm(size(umap_train_cc, 1));  % Generate a random permutation of the rows of the data matrix
% Divide the random permutation into tow parts in a 80:20 ratio
input_indices = rand_indices(1:round(0.8 * length(rand_indices)));
validate_indices = rand_indices(round(0.8 * length(rand_indices)) + 1: end);
input_data = umap_train_cc(input_indices, :);
validate_data = umap_train_cc(validate_indices, :);
% Extract input, target, validation, and test data from the original data matrix
inputs = input_data(:,1:size(input_data,2)-1);
targets = input_data(:,size(input_data,2));
valid = validate_data(:,1:size(validate_data,2)-1);
validtargets = validate_data(:,size(validate_data,2));
test = umap_test_cc(:,1:size(umap_test_cc,2)-1);
testtargets = umap_test_cc(:,size(umap_test_cc,2));
clear input_indices validate_indices test_indices input_data validate_data test_data

% Initialize MLP network
nIterations=10;
eta=0.0001;
nhidden=13;

% learning
tic
% omlp1 = mlp(nhidden,'threshold', false, 'sequential');  % sequential
% omlp1 = omlp1.fit(inputs, targets, eta, nIterations);
omlp = mlp(nhidden,'threshold', true, 'sequential');  % sequential & earlystopping
omlp = omlp.fit(inputs, targets,eta,nIterations,0.9,valid,validtargets);
% omlp3 = mlp(nhidden, 'threshold', false, 'batchtraining');  % batchtraining
% omlp3 = omlp3.fit(inputs, targets,eta,nIterations);
% omlp4 = mlp(nhidden,'threshold', true, 'batchtraining');  % batchtraining & earlystopping
% omlp4 = omlp4.fit(inputs, targets,eta,nIterations,0.9,valid,validtargets);
% omlp = mlp(nhidden,'threshold', false, 'minibatches', 2);  % minibatches
% omlp = omlp.fit(inputs, targets,eta,nIterations);
% omlp = mlp(nhidden,'threshold', true, 'minibatches', 2);  % minibatches & earlystopping
% omlp = omlp.fit(inputs, targets, eta, nIterations, 0.9, valid, validtargets);
toc

% evaluate the model
[error, output] = omlp.test(test, testtargets);
fprintf("Test Error: %.5f\n", error);

C = confusionmat(testtargets,output);
TN = C(1,1);
FP = C(1,2);
FN = C(2,1);
TP = C(2,2);
Accuracy = (TN+TP)/(FN+FP+TN+TP);
Precision = TP/(FP+TP);
Recall = TP/(FN+TP);
F_measure = 2*(Recall*Precision)/(Recall+Precision);

fprintf('Accuracy: %.3f\n',Accuracy);
fprintf('Precision: %.3f\n',Precision);
fprintf('Recall: %.3f\n',Recall);
fprintf('F-measure: %.3f\n',F_measure);

figure
hold off
confusionchart(testtargets,output)
% cm = confusionchart(testtargets,output, 'ColumnSummary','column-normalized', 'RowSummary','row-normalized');
% plotconfusion(testtargets, output)

% plot plot RMS vs number of iterations
niter = length(omlp.train_error);
x=linspace(1,niter,niter);
figure
plot(x,omlp.train_error)
grid on
title('Root Mean Square Error')
xlabel('Iteration')
ylabel('Error')

