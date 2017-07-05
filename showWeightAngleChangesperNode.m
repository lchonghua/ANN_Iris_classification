function [] = showWeightAngleChangesperNode(layer, node, weights, p_weights)

    figurename=strcat('Weight Angle changes of node ', int2str(node), ' on layer ', int2str(layer));
    figure('name',figurename);
    for i=1:length(weights(:,1))
        anglechange(i)=acos((dot(weights{i,layer}(:,node),p_weights{i,layer}(:,node)))./(norm(weights{i,layer}(:,node))*norm(p_weights{i,layer}(:,node))));
    end
    plot(anglechange);
    ylim([0 0.1]);
    
end