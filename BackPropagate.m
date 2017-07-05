function [weightCell, biasCell, D_weight, D_bias, acum_D_weight, acum_D_bias] = BackPropagate(rate, in, realOutput, sampleTarget, layer, weightCell, biasCell, layerOutputCells,momentum, previous_D_weight, previous_D_bias, acum_D_weight, acum_D_bias, myfunction,myfunctionD,exampleNumber,trainsetCount,batchsize)
    
    layerCount = size(layer, 2);

    %Create variables for deltas
    
    delta = cell(1, layerCount);
    D_weight = cell(1, layerCount);
    D_bias = cell(1, layerCount);

    %Calculate Error for Output Layer
   
    delta{layerCount} = myfunctionD(realOutput) .* (sampleTarget - myfunction(realOutput));
    preoutput = myfunction(layerOutputCells{layerCount-1});
    D_weight{layerCount} = rate .* preoutput' * delta{layerCount} + momentum.*previous_D_weight{layerCount};
    acum_D_weight{layerCount}=acum_D_weight{layerCount}+D_weight{layerCount};
    D_bias{layerCount} = rate .* delta{layerCount}+momentum.*previous_D_bias{layerCount};
    acum_D_bias{layerCount}=acum_D_bias{layerCount}+D_bias{layerCount};
    %Calculate Error for Hidden layers
    
    for layerIndex = layerCount-1:-1:1
        output = layerOutputCells{layerIndex};
        if layerIndex == 1
            preoutput = in;
        else
            preoutput = myfunction(layerOutputCells{layerIndex-1});
        end
        weight = weightCell{layerIndex+1};
        sumup = (weight * delta{layerIndex+1}')';
        delta{layerIndex} = myfunctionD(output) .* sumup;
        D_weight{layerIndex} = rate .* preoutput' * delta{layerIndex} +momentum.*previous_D_weight{layerIndex};
        acum_D_weight{layerIndex}=acum_D_weight{layerIndex}+D_weight{layerIndex};
        D_bias{layerIndex} = rate .* delta{layerIndex}+momentum.*previous_D_bias{layerIndex};
        acum_D_bias{layerIndex}=acum_D_bias{layerIndex}+D_bias{layerIndex};
    end
    
    %Update Weights and Bias
    
    if mod(exampleNumber, batchsize) == 0 || exampleNumber == trainsetCount  % Batch size
        for layerIndex = 1:layerCount
            weightCell{layerIndex} = weightCell{layerIndex} + acum_D_weight{layerIndex};
            biasCell{layerIndex} = biasCell{layerIndex} + acum_D_bias{layerIndex};
            acum_D_weight{layerIndex}=0;
            acum_D_bias{layerIndex}=0;
        end
        
    end
end