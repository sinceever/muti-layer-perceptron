% author: Jingyu Ren
% Laurentian University
% Student ID: 0421763
% Multiple layer perceptron

classdef mlp
    % A Multi-Layer Perceptron
    properties
        ndata  % number of input samples #1400
        nin  % number of input features #12
        nhidden  % number of hidden nodes
        nout  % number of output features #1
        weights1 % Wij (L+1)*M bias=-1
        weights2 % Wjk (M+1)*N bias=-1
        hidden  % hidden layer's activation
        outputs % output
        train_error % Matrix of train_error for each iteration

        nt_hidden  % node type of hidden layer 'linear' 'sigmoid' 'relu' 'tanh'
        nt_output  % node type of output layer 'linear' 'sign' 'soft-max' 'threshold'
        earlystop  % switch to use earlystopping varification
        batch_method  % 'sequential' 'batchtraining' 'minibatches'
        batch_size  % hyperparameter of batch size when using minibatches 2 to 1024
        momentum  % pick up Momentum
        updatew1  % update weights1 in hidden layer 
        updatew2  % update weights2 in ouput layer

        %weight_record % 2.f Record of some weights updates for each iteration
    end
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constructor             %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % omlp = mlp(inputs, targets, nhidden);
        function obj = mlp(nhidden, nt_output, earlystop, batch_method, batch_size)
            % Set up network size
            % ndata:number of samples in training dataset
            % nin:number of nodes in input layer
            arguments
                nhidden=5
                nt_output = 'linear'
                earlystop = true
                batch_method = 'batchtraining'
                batch_size = 128
            end
            if batch_size <= 1024 && batch_size >= 2
                obj.batch_size = batch_size;
            else
                disp('Batch size should be a value between 2 and 1024')
                return
            end
            obj.batch_method = batch_method;
            obj.earlystop = earlystop;
            obj.nt_output = nt_output;
            obj.nhidden = nhidden;  % number of nodes in hidden layer
            % Recording data to plot
            obj.train_error = [];
            %obj.weight_record = [obj.weights1(1,1); obj.weights1(9,5); obj.weights1(13,3); obj.weights2(2,1);obj.weights2(4,1)];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Fit() entrance function %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = fit(obj, inputs, targets, eta, niterations, momentum, valid, validtargets)
            arguments
                obj
                inputs
                targets
                eta=0.01
                niterations=100
                momentum=0
                valid=[]
                validtargets=[]
            end
            obj.momentum=momentum;
            [obj.ndata, obj.nin] = size(inputs);
            obj.nout = size(targets,2);  % number of nodes in output layer
            % Initialize weights with random values
            % according to the algorithm in the textbook's sample code
            obj.weights1 = 2 / sqrt(obj.nin) * (rand(obj.nin + 1, obj.nhidden) - 0.5);  % Wij (L+1)*M 13*5
            obj.weights2 = 2 / sqrt(obj.nhidden) * (rand(obj.nhidden + 1, obj.nout) - 0.5);  % Wjk (M+1)*N 6*1
            obj.updatew1 = zeros(size(obj.weights1));
            obj.updatew2 = zeros(size(obj.weights2));
            %obj.weights1 = zeros(obj.nin + 1,obj.nhidden);
            %obj.weights2 = zeros(obj.nhidden + 1, obj.nout);

            % earlystopping validation method
            if obj.earlystop==true
                if isempty(valid) || isempty(validtargets)
                    disp('Validation data missing for fit()')
                    return
                end
                obj = obj.earlystopping(inputs, targets, valid, validtargets, eta, niterations);
            else % no validation
                % branch 'sequential' 'batchtraining' 'minibatches'
                if strcmp(obj.batch_method,'sequential')
                    for j=1:niterations
                        fprintf("Iteration of sequential learning: %d\n", j);
                        for i=1:obj.ndata
                            obj = obj.mlptrain(inputs(i,:), targets(i,:), eta, 1);
                        end
                        % recording training error
                        [error, ~] = obj.test(inputs,targets);
                        obj.train_error = [obj.train_error error];
                        fprintf("Training Error  : %.5f\n", error);
                        % shuffle the order of training set
                        % jeopardize the learning performance
                        rand_indices = randperm(obj.ndata);
                        inputs = inputs(rand_indices, :);
                        targets = targets(rand_indices, :);
                    end
                elseif strcmp(obj.batch_method,'minibatches')
%                     [Q,~]=quorem(obj.ndata,obj.batch_size); % Q=3 R=3 ndata=18 batch_size=5
                    Q = floor(obj.ndata./obj.batch_size);
                    for j=1:niterations
                        fprintf("Iteration of minibatches learning: %d\n", j);
                        % first batch 1-5
                        obj = obj.mlptrain(inputs(1:obj.batch_size,:), targets(1:obj.batch_size,:), eta, 1);
                        % other batches 6-15  : 6-10 11-15
                        for i=1:Q-1
                            obj = obj.mlptrain(inputs(obj.batch_size*i+1:obj.batch_size*(i+1),:), targets(obj.batch_size*i+1:obj.batch_size*(i+1),:), eta, 1);
                        end
                        % last batch 16-18
                        obj = obj.mlptrain(inputs(obj.batch_size*Q+1:obj.ndata,:), targets(obj.batch_size*Q+1:obj.ndata,:), eta, 1);
                        % recording training error
                        [error, ~] = obj.test(inputs,targets);
                        obj.train_error = [obj.train_error error];
                        fprintf("Training Error  : %.5f\n", error);
                        % shuffle the order of training set
                        rand_indices = randperm(obj.ndata);
                        inputs = inputs(rand_indices, :);
                        targets = targets(rand_indices, :);
                    end
                else
                    obj = obj.mlptrain(inputs, targets, eta, niterations);
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Earlystopping           %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % omlp = omlp.earlystopping(inputs,targets,valid,validtargets,eta,nIterations);
        function obj = earlystopping(obj, inputs, targets, valid, validtargets, eta, niterations)
            old_val_error1 = 100002;
            old_val_error2 = 100001;
            new_val_error = 100000;
            count = 0;
            Q = floor(obj.ndata./obj.batch_size);
            while ((old_val_error1 - new_val_error) > 0.0001) || ((old_val_error2 - old_val_error1) > 0.0001)
                count = count + 1;
                fprintf("Iteration: %d\n", count*niterations);
                % branch 'sequential' 'batchtraining' 'minibatches'
                if strcmp(obj.batch_method,'sequential')
                    for i=1:obj.ndata
                        obj = obj.mlptrain(inputs(i,:), targets(i,:), eta, 1);
                    end
                    % recording training error
                    [error, ~] = obj.test(inputs,targets);
                    obj.train_error = [obj.train_error error];
                    % shuffle the order of training set
                    % jeopardize the learning performance
                    rand_indices = randperm(obj.ndata);
                    inputs = inputs(rand_indices, :);
                    targets = targets(rand_indices, :);
                elseif strcmp(obj.batch_method,'minibatches')
                    % first batch
                    obj = obj.mlptrain(inputs(1:obj.batch_size,:), targets(1:obj.batch_size,:), eta, 1);
                    % other batches
                    for i=1:Q-1
                        obj = obj.mlptrain(inputs(obj.batch_size*i+1:obj.batch_size*(i+1),:), targets(obj.batch_size*i+1:obj.batch_size*(i+1),:), eta, 1);
                    end
                    % last batch
                    obj = obj.mlptrain(inputs(obj.batch_size*Q+1:obj.ndata,:), targets(obj.batch_size*Q+1:obj.ndata,:), eta, 1);
                     % recording training error
                    [error, ~] = obj.test(inputs,targets);
                    obj.train_error = [obj.train_error error];
                    % shuffle the order of training set
                    % jeopardize the learning performance
                    rand_indices = randperm(obj.ndata);
                    inputs = inputs(rand_indices, :);
                    targets = targets(rand_indices, :);
                else
                    obj = obj.mlptrain(obj, inputs, targets, eta, niterations);
                end
                old_val_error2 = old_val_error1;
                old_val_error1 = new_val_error;
                [new_val_error, ~] = obj.test(valid, validtargets);
                fprintf("Training Error  : %.5f\n", obj.train_error(end));
                fprintf("Validation Error: %.5f\n", new_val_error);
            end
            fprintf("Stopped with Validation Error of last 3 meaturements: %.5f %.5f %.5f\n", new_val_error, old_val_error1, old_val_error2);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Training                %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = mlptrain(obj, inputs, targets, eta, niterations)
            m = size(inputs,1);
            % Add the inputs that match the bias node
            inputs = [inputs, -ones(m, 1)];

            for n = 1:niterations
                obj = mlpfwd(obj, inputs); % forward
                if strcmp(obj.batch_method, 'batchtraining')
                    error = 0.5 / m * sum((obj.outputs - targets) .^ 2);
                    % recording training error
                    obj.train_error = [obj.train_error error];
                end
                % backward propagation
                %                 deltao = (obj.outputs - targets) .* obj.outputs .* (1.0 - obj.outputs);  % δo sigmoid
                deltao = (obj.outputs - targets)/m;  % δo linear
                %                 deltao = obj.outputs - targets;  % δo linear
                deltah = obj.hidden .* (1.0 - obj.hidden) .* (deltao .* obj.weights2');  % δh
                obj.updatew1 = eta * (inputs.' * deltah(:, 1:end-1)) + obj.momentum*obj.updatew1;
                obj.updatew2 = eta * (obj.hidden.' * deltao) + obj.momentum*obj.updatew2; % sum all samples up
                obj.weights1 = obj.weights1 - obj.updatew1;
                obj.weights2 = obj.weights2 - obj.updatew2;
                % 2.e recording h:w11 w95 w133 o:w21 w41
                %obj.weight_record = [obj.weight_record, [obj.weights1(1,1); obj.weights1(9,5); obj.weights1(13,3); obj.weights2(2,1);obj.weights2(4,1)]];
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Run the network forward %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = mlpfwd(obj, inputs)
            % Calculate activations of hidden layer
            % 1400*5 hidden(i,j) net value for hidden node j for sample i
            obj.hidden = inputs * obj.weights1;
            obj.hidden = 1.0 ./ (1.0 + exp(- obj.hidden));  % g(x) sigmoid
            % Add the inputs that match the bias node
            obj.hidden = cat(2,obj.hidden,-ones(size(inputs, 1),1));
            % Calculate net of output layer
            obj.outputs = obj.hidden * obj.weights2;
            if strcmp(obj.nt_output,'threshold')
                obj.outputs(find(obj.outputs>0))=1;
                obj.outputs(find(obj.outputs~=1))=0;
            end
            % Calculation activation of final activation
            % obj.outputs = 1.0 ./ (1.0 + exp(-obj.outputs)); %sigmoid
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Test&get outputs&error  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % [error, output] = omlp.test(inputs, targets);
        function [error, output] = test(obj, inputs, targets)
            % Add the inputs that match the bias node
            inputs = [inputs, -ones(size(inputs,1), 1)];
            obj = mlpfwd(obj, inputs); % forward
            output = obj.outputs;
            error = 0.5 / size(inputs,1) * sum((obj.outputs - targets) .^ 2);
        end
    end
end