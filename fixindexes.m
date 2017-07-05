 function [index] = fixindexes (size,trainingSize,validationSize)
    index=zeros(1,size);
    ind=1;
    for i=1:trainingSize
        if mod(ind,3)==0       
            ind=ind+1;
        end 
        index(i)=ind;
        ind=ind+1;
    end
    ind=1;
    for i=trainingSize+1:size 
        while mod(ind,3)~=0
            ind=ind+1;
        end  
        index(i)=ind;
        ind=ind+1;
    end
end