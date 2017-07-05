function [ f ] = mysigmoidD( x )
    f=mysigmoid(x).*(1.-mysigmoid(x));
end