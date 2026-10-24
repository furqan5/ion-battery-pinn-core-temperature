function warwick_export(matfile, outdir)
% Export a Warwick LG M50 dataTable (MATLAB table, one row per cell, each column a 1-by-n cell array
% of test segments) to plain numeric v7 .mat files that scipy can read, one file per cell and segment.
% Column names differ between files, so they are matched after removing spaces, units and case.
% Text columns (Status) are stored as integer codes with the code list saved beside them; a clock
% column (Time Temp, hh:mm:ss) is stored in seconds.
if ~exist(outdir, 'dir'); mkdir(outdir); end
S = load(matfile);
fn = fieldnames(S);
T = S.(fn{1});
names = T.Properties.VariableNames;
key = lower(regexprep(regexprep(names, '\[.*\]', ''), '\s', ''));
want = {'step', 'status', 'steptime', 'progtime', 'cycleno', 'voltage', 'current', 'capacity', ...
        'timetemp', 'surftemp', 'coretemp'};
short = {'step', 'status', 'step_time', 'prog_time', 'cycle', 'V', 'I', 'cap', 'time_temp', 'Ts', 'Tc'};
codes = strings(0);
fid = fopen(fullfile(outdir, 'index.txt'), 'w');
jstep = find(strcmp(key, 'step'), 1);
for r = 1:height(T)
    cellname = string(T{r, 'Cell Name'});
    cond = string(T{r, 'Condition'});
    seg = unwrap(T{r, jstep});
    for s = 1:numel(seg)
        out = struct();
        n = NaN;
        for c = 1:numel(want)
            j = find(strcmp(key, want{c}), 1);
            if isempty(j); continue; end
            col = unwrap(T{r, j});
            if ~iscell(col) || numel(col) < s; continue; end
            x = col{s};
            if isempty(x); continue; end
            if isnumeric(x) || islogical(x)
                x = double(x(:));
            elseif isduration(x)
                x = seconds(x(:));
            elseif isdatetime(x)
                x = seconds(x(:) - x(1));
            elseif strcmp(want{c}, 'timetemp')
                x = seconds(duration(string(x(:))));
            else
                xs = string(x(:));
                u = unique(xs);
                for k = 1:numel(u)
                    if ~any(codes == u(k)); codes(end + 1) = u(k); end %#ok<AGROW>
                end
                [~, x] = ismember(xs, codes);
                x = double(x);
            end
            out.(short{c}) = x;
            n = numel(x);
        end
        f = sprintf('%s_seg%d.mat', cellname, s);
        save(fullfile(outdir, f), '-struct', 'out', '-v7');
        fprintf(fid, '%s\t%s\tseg%d\tn=%d\t%s\n', cellname, cond, s, n, strjoin(fieldnames(out)', ','));
    end
end
fclose(fid);
fid = fopen(fullfile(outdir, 'status_codes.txt'), 'w');
for k = 1:numel(codes); fprintf(fid, '%d\t%s\n', k, codes(k)); end
fclose(fid);
fprintf('exported %s -> %s\n', matfile, outdir);
end

function c = unwrap(c)
while iscell(c) && numel(c) == 1 && iscell(c{1}); c = c{1}; end
end
