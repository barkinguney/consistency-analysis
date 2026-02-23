% Load your reactions CSV (Expected columns: Name, Nominal, LB, UB)
rxnData = readtable('your_reactions.csv');

myVars = B2BDC.B2Bvariables.VariableList();
for i = 1:height(rxnData)
    v = B2BDC.B2Bvariables.ModelVariable(rxnData.Name{i}, ...
        rxnData.LB(i), rxnData.UB(i), rxnData.Nominal(i));
    myVars = myVars.addVariable(v);
end