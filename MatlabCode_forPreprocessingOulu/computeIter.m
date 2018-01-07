function iter = computeIter(uttID)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

switch(uttID)
    
    case {31,34,37,40,43,46,49,52,55,58}
        iter = 1;
    case {32,35,38,41,44,47,50,53,56,59}
        iter = 2;
    case {33,36,39,42,45,48,51,54,57,60}
        iter = 3;
    otherwise
        iter = NaN;
end

