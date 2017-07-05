
function  UniversalNN()
    
    %Set training parameters
    rng('default');
    seed = 30;
    
    iterations = 500;
    enditeration = 0;
    errorThreshhold = 0.1;
    learningRate = 0.1;
    batch=1;
   
    hiddenNeurons = [2 2];
    momentum = 0.1;
        
    %Choosing the activation function
    myfunction=str2func('mysigmoid');
    myfunctionD=str2func('mysigmoidD');
    
    %Loading Fisher's Dataset (IRIS)
    dataset=load('fisheriris.mat');
    trainsetCount =100;%size(trainInp, 1);
    validationCount =50;%size(trainInp, 1);
    
    %Shuffle the data
    index = randperm(length(dataset.species));
    trainInp=dataset.meas(index,:);
    trainOut=dataset.species(index,:);
    
    %Transform output
    trainOut=transformOutput(trainOut,length(unique(trainOut)));
    
    %Initialize Network attributes   
    inArgc = size(trainInp, 2);
    outArgc = size(trainOut, 2); 
        
    %---Add output layer
    layerOfNeurons = [hiddenNeurons, outArgc];
    layerCount = size(layerOfNeurons, 2);
    
    %---Set initial random weights
    weightCell = cell(1, layerCount);
    previousweightCell = cell(1, layerCount);
    
    for i = 1:layerCount
        if i == 1
            rand(seed);
            weightCell{1} = -0.5 +1.*rand(inArgc,layerOfNeurons(1));
        else
            rand(seed);
            weightCell{i} = -0.5 +1.*rand(layerOfNeurons(i-1),layerOfNeurons(i));
        end
    end
    
    %---Set initial biases
    biasCell = cell(1, layerCount);
    for i = 1:layerCount
        rand(seed);
        biasCell{i} = -0.5 +1.*rand(1, layerOfNeurons(i));
    end
    
    %Begin of epochs
    for iter = 1:iterations
        
        %Defining variables for batch size and momentum
        previous_D_weight = cell(1, size(layerOfNeurons, 2));
        acum_D_weight = cell(1, size(layerOfNeurons, 2));
        
        for k=1:size(layerOfNeurons, 2)
            previous_D_weight{k} = 0;
            acum_D_weight{k} = 0;
        end
       
        previous_D_bias = cell(1, size(layerOfNeurons, 2));
        acum_D_bias = cell(1, size(layerOfNeurons, 2));
        
        for k=1:size(layerOfNeurons, 2)
            previous_D_bias{k} = 0;
            acum_D_bias{k} = 0;
        end
        
        %Begin training
        trainingerror = 0;
        previousweightCell = weightCell;
        for i = 1:trainsetCount
            choice=i;
            sampleIn = trainInp(choice, :);
            sampleTargetTraining = trainOut(choice,:);
            [realOutput ,layerOutputCells] = ForwardNetwork(sampleIn, layerOfNeurons, weightCell, biasCell,myfunction);
            [weightCell, biasCell, previous_D_weight, previous_D_bias, acum_D_weight, acum_D_bias] = BackPropagate(learningRate, sampleIn, realOutput, sampleTargetTraining, layerOfNeurons, ...
                weightCell, biasCell, layerOutputCells,momentum,previous_D_weight, previous_D_bias, acum_D_weight, acum_D_bias,myfunction,myfunctionD,i,trainsetCount,batch);
        end
        
        for i = 1:trainsetCount
            choice=i;
            sampleIn = trainInp(choice, :);
            sampleTargetTraining = trainOut(choice,:);
            [realOutput ,layerOutputCells] = ForwardNetwork(sampleIn, layerOfNeurons, weightCell, biasCell,myfunction);
            trainingerror = trainingerror+sum((sampleTargetTraining-myfunction(realOutput)).^2);
            activations(i+(iter-1)*trainsetCount,:)=layerOutputCells;
        end
        
        % Variables to calculate Weight angle changes
        P_weights(iter,:) = previousweightCell;
        weights(iter,:) = weightCell;
     
        %Begin Validation
        validationerror = 0;
        for t = 1:validationCount
            choice = t+trainsetCount;
            sampleTargetValidation = trainOut(choice,:);
            [predict] = ForwardNetwork(trainInp(choice, :), layerOfNeurons, weightCell, biasCell,myfunction);
            validationerror = validationerror+sum((sampleTargetValidation - myfunction(predict)).^2);
        end
        
        %Calculate Tranning Error and Validation Error
        tErr(iter) = (trainingerror/trainsetCount);
        vErr(iter) = (validationerror/validationCount);
          
        %Error threshold
        if tErr(iter) < errorThreshhold && enditeration == 0
            enditeration = iter;
        end
        
        %Run 10 iterations after meeting errorThreshhold
        if enditeration ~= 0 && iter == enditeration+10
         break;
        end
    end
    
    %---Print Results
    option =0;
    while option ~= 5    
        prompt = '\n\n\1) Number of iterations.\n2) Training vs Testing Error.\n3) Activation Histogram per node. \n4) Weight Angle changes per node. \n5) Exit.\n\n Option: ';
        option = input(prompt);
        switch option
            case 1
                fprintf('Threshhold meet at %d iterations.\n', enditeration);
            case 2
                figure('name','Training Error vs Validation Error');
                plot(tErr); 
                hold on;
                plot(vErr, 'red');
            case 3
                prompt = '\nInsert the iteraction you want to analyze (zero for complete simulation): ';
                iteration= input(prompt);
                prompt = '\nInsert the layer you want to analyze: ';
                layer= input(prompt);
                prompt = '\nInsert the node you want to analyze: ';
                node= input(prompt);
                if layer <= layerCount && layer > 0 && node <= length(activations{1,layer}(1,:)) && node > 0
                    showActivationHistogramperNode(iteration,trainsetCount,layer,node,activations,myfunction);
                else
                    disp('Invalid layer or node, try again. ');
                end
            case 4
                prompt = '\nInsert the layer you want to analyze: ';
                layer= input(prompt);
                prompt = '\nInsert the node you want to analyze: ';
                node= input(prompt);
                if layer <= layerCount && layer > 0 && node <= length(activations{1,layer}(1,:)) && node > 0
                    showWeightAngleChangesperNode(layer, node, weights, P_weights);
                else
                    disp('Invalid layer or node, try again. ');
                end
            case 5
            otherwise 
                disp('Invalid option, try again. ');
        end
    end
end

