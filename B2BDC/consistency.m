function out = consistency()
rootDir = fileparts(fileparts(mfilename('fullpath')));
inputDir = fullfile(rootDir, 'B2BDC', 'input');
trainingDir = fullfile(rootDir, 'surrogate', 'data', 'training');
outputDir = fullfile(rootDir, 'B2BDC', 'output');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

rxnFile = fullfile(inputDir, 'active_reactions.csv');
expFile = fullfile(inputDir, 'experiment_units.csv');

assert(exist(rxnFile, 'file') == 2, 'Missing file: %s', rxnFile);
assert(exist(expFile, 'file') == 2, 'Missing file: %s', expFile);
assert(exist(trainingDir, 'dir') == 7, 'Missing directory: %s', trainingDir);

rxnData = readtable(rxnFile, 'TextType', 'string');
requiredRxnCols = ["equation", "nominal", "lb", "ub"];
assert(all(ismember(requiredRxnCols, string(rxnData.Properties.VariableNames))), ...
    'active_reactions.csv must contain columns: equation, nominal, lb, ub');

varNames = cellstr(rxnData.equation);
H = [rxnData.lb, rxnData.ub];
nominalVals = rxnData.nominal;
varList = generateVar(varNames, H, nominalVals);

expData = readtable(expFile, 'TextType', 'string');
requiredExpCols = ["T", "P", "phi", "exp_idt", "exp_unc", "exp_id"];
assert(all(ismember(requiredExpCols, string(expData.Properties.VariableNames))), ...
    'experiment_units.csv must contain columns: T, P, phi, exp_idt, exp_unc, exp_id');

datasetName = sprintf('UserDataset_%s', datestr(now, 'yyyy-mm-dd_HHMMSS'));
ds = generateDataset(datasetName);

trainingFiles = dir(fullfile(trainingDir, '*_training.csv'));
assert(~isempty(trainingFiles), 'No training files found in: %s', trainingDir);

fitMethod = 'qinf';
nUnitsAdded = 0;
fitDiagnostics = table('Size', [0 4], ...
    'VariableTypes', {'string', 'double', 'double', 'string'}, ...
    'VariableNames', {'exp_id', 'n_samples', 'n_vars', 'training_file'});

for i = 1:height(expData)
    expId = expData.exp_id(i);
    expT = expData.T(i);
    expP = expData.P(i);

    trainingPath = resolveTrainingFile(trainingDir, trainingFiles, expId, expT, expP);
    trainTbl = readtable(trainingPath, 'TextType', 'string');
    trainVarNames = string(trainTbl.Properties.VariableNames);

    assert(any(trainVarNames == "log_idt"), ...
        'Training file %s must contain column log_idt', trainingPath);

    xMask = startsWith(trainVarNames, "m_");
    xCols = find(xMask);
    assert(~isempty(xCols), 'Training file %s must contain m_* columns', trainingPath);

    X = table2array(trainTbl(:, xCols));
    y = table2array(trainTbl(:, "log_idt"));

    assert(size(X, 1) == numel(y), 'X/Y row mismatch in %s', trainingPath);
    assert(size(X, 2) == numel(varNames), ...
        'Variable count mismatch in %s. Training has %d vars, active_reactions has %d vars.', ...
        trainingPath, size(X, 2), numel(varNames));

    obsIdt = expData.exp_idt(i);
    relUnc = expData.exp_unc(i);
    lowerLin = obsIdt * (1 - relUnc);
    upperLin = obsIdt * (1 + relUnc);

    assert(lowerLin > 0 && upperLin > 0, ...
        'Invalid linear bounds for exp_id=%s. exp_idt and exp_unc produced nonpositive bound.', expId);

    qoiBoundsLog = [log(lowerLin), log(upperLin)];

    qModel = generateModelbyFit(X, y, varList, fitMethod);
    dsUnit = generateDSunit(char(expId), qModel, qoiBoundsLog);
    ds.addDSunit(dsUnit);

    nUnitsAdded = nUnitsAdded + 1;
    fitDiagnostics = [fitDiagnostics; {expId, size(X, 1), size(X, 2), string(trainingPath)}]; %#ok<AGROW>
end

ds.DatasetUnits.Values(1).VariableList
ds.Variables

isConsistent = ds.isConsistent;
ds.plotConsistencySensitivity

deletedUnits = ds.deleteUnit([1,2,3,4,5,6,7,8,9,10]);
tryConsistentAgain = ds.isConsistent
ds.plotConsistencySensitivity

cm = ds.ConsistencyMeasure;

reportFile = fullfile(outputDir, 'consistency_report.txt');
fid = fopen(reportFile, 'w');
fprintf(fid, 'Dataset: %s\n', datasetName);
fprintf(fid, 'Units added: %d\n', nUnitsAdded);
fprintf(fid, 'Fit method: %s\n', fitMethod);
fprintf(fid, 'QoI space: log(IDT)\n');
fprintf(fid, 'Is consistent: %d\n', isConsistent);
fprintf(fid, 'Is tryConsistentAgain: %d\n', tryConsistentAgain);
fprintf(fid, 'Consistency measure:\n');
fprintf(fid, '%s\n', evalc('disp(cm)'));

vcmReport = [];
if ~isConsistent
    vcmReport = ds.vectorConsistency('unit', 'null');
    fprintf(fid, '\nVector consistency report:\n');
    fprintf(fid, '%s\n', evalc('disp(vcmReport)'));
end
fclose(fid);

writetable(fitDiagnostics, fullfile(outputDir, 'fit_diagnostics.csv'));
save(fullfile(outputDir, 'consistency_results.mat'), ...
    'datasetName', 'nUnitsAdded', 'isConsistent', 'cm', 'vcmReport', 'fitDiagnostics');

fprintf('B2BDC consistency completed.\n');
fprintf('Units added: %d\n', nUnitsAdded);
fprintf('Is consistent: %d\n', isConsistent);
fprintf('Report: %s\n', reportFile);

out = struct();
out.datasetName = datasetName;
out.nUnitsAdded = nUnitsAdded;
out.isConsistent = isConsistent;
out.reportFile = reportFile;
out.outputDir = outputDir;

end

function trainingPath = resolveTrainingFile(trainingDir, trainingFiles, expId, expT, expP)
exactName = strcat(expId, "_training.csv");
exactPath = fullfile(trainingDir, char(exactName));
if exist(exactPath, 'file') == 2
    trainingPath = exactPath;
    return;
end

tolT = 1e-9 * max(1.0, abs(expT));
tolP = 1e-9 * max(1.0, abs(expP));

matches = strings(0, 1);
for k = 1:numel(trainingFiles)
    fname = string(trainingFiles(k).name);
    tokens = regexp(fname, '^(.*)_K_(.*)_training\.csv$', 'tokens', 'once');
    if isempty(tokens)
        continue;
    end
    tVal = str2double(tokens{1});
    pVal = str2double(tokens{2});
    if isnan(tVal) || isnan(pVal)
        continue;
    end
    if abs(tVal - expT) <= tolT && abs(pVal - expP) <= tolP
        matches(end+1, 1) = string(fullfile(trainingDir, char(fname))); %#ok<AGROW>
    end
end

if numel(matches) == 1
    warning('Exact exp_id training file not found for %s. Using T/P fallback: %s', expId, matches(1));
    trainingPath = char(matches(1));
    return;
end

if isempty(matches)
    error(['No training file found for exp_id=%s. Expected exact file %s or a unique ' ...
        'T/P match in %s.'], expId, exactPath, trainingDir);
end

error('Multiple training files matched exp_id=%s by T/P fallback. Please disambiguate.', expId);
end