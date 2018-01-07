function trg = computeTargetsPhrases(uttID)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

switch(uttID)
    
    case {31,32,33}
        trg = 1;
    case {34,35,36}
        trg = 2;
    case {37,38,39}
        trg = 3;
    case {40,41,42}
        trg = 4;
    case {43,44,45}
        trg = 5;
    case {46,47,48}
        trg = 6;
    case {49,50,51}
        trg = 7;
    case {52,53,54}
        trg = 8;
    case {55,56,57}
        trg = 9;
    case {58,59,60}
        trg = 10;
    otherwise
        trg = NaN;
end

