function [] = showActivationHistogramperNode(iteration, trainsetCount, layer, node, activations, activationfunction)
    if strcmp(func2str(activationfunction),'mysigmoid')
        lim=0.05:0.1:0.95;
       
    else if strcmp(func2str(activationfunction),'mytanh')
            lim=-0.9:0.2:0.9;
           
        else
            lim=0.5:1:9.5;
            
        end
    end
 
    if iteration == 0
        figurename=strcat('Activations of node ', int2str(node), ' on layer ', int2str(layer));
        figure('name',figurename);
        for i=1:length(activations(:,1))
            act(i)=activationfunction(activations{i,layer}(node));
        end
    else
        figurename=strcat('Activations of node ', int2str(node), ' on layer ', int2str(layer), 'on iteraction: ', int2str(iteration));
        figure('name',figurename);
        for i=(iteration-1)*trainsetCount+1:(iteration-1)*trainsetCount+trainsetCount
            act(i-(iteration-1)*trainsetCount)=activationfunction(activations{i,layer}(node));
        end
    end
    hist(act,lim); 
    
end