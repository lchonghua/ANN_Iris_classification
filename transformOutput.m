function [newoutput] = transformOutput(oldOutput,classes)
    for i=1:length(oldOutput)
        temp=zeros(1,classes);
        specie=oldOutput(i);
        if strcmp(specie,'setosa')
            temp(1)=1;
        else
            if strcmp(specie,'versicolor')
                temp(2)=1;
            else
                temp(3)=1;
            end
        end
        newoutput(i,:)=temp;
    end
end
