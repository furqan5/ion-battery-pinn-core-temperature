function warwick_inspect(matfile)
% Print the structure of one Warwick LG M50 dataTable file (MATLAB table objects).
S = load(matfile);
fn = fieldnames(S);
fprintf('file: %s\n', matfile);
for i = 1:numel(fn)
    v = S.(fn{i});
    fprintf('variable %s: class %s, size %s\n', fn{i}, class(v), mat2str(size(v)));
    if istable(v)
        disp(v.Properties.VariableNames);
        disp(v(1:min(3, height(v)), :));
        for c = {'CellName', 'Condition', 'Sensor', 'Status', 'Cell_Name'}
            if any(strcmp(v.Properties.VariableNames, c{1}))
                u = unique(string(v.(c{1})));
                fprintf('  %s: %s\n', c{1}, strjoin(u(1:min(20, numel(u)))', ' | '));
            end
        end
    end
end
end
