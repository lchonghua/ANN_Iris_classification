function [ f ] = mysigmoid( x )
    f=1./(1+exp(-x));
end
