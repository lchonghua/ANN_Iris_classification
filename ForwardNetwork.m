function [realOutput, layerOutputCells] = ForwardNetwork(in, layer, weightCell, biasCell,myfunction)
  
    layerCount = size(layer, 2);
    layerOutputCells = cell(1, layerCount);
    out = in;
    for layerIndex = 1:layerCount
        
        %For Input Layer
        
        if layerIndex == 1
            X = out;
            
         %For Hidden Layers
        else
            X = myfunction(out);
        end
        bias = biasCell{layerIndex};

        %Saving X*Wi+b without appliying the activation function to
        %customize the function to be used
        
        out = X*weightCell{layerIndex} + bias;
        layerOutputCells{layerIndex} = out;
    end
    realOutput = out;    
end