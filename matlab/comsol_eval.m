function comsol_eval(in_file, out_file)
%COMSOL_EVAL  Generic job wrapper called by the Python optimizer.
%
%   comsol_eval('job.json', 'result.json')
%
%   job.json must contain
%       case_function : name of the MATLAB case function (must be on the path)
%       params        : struct of COMSOL parameter expressions, e.g. {"lam": "3[mm]"}
%       opts          : struct of options passed straight to the case function
%
%   The case function has signature  res = case_function(params, opts)
%   and returns a struct.  Whatever it returns is written to result.json,
%   together with a status field ('ok' / 'failed') and the wall time.
%   Errors never propagate to the caller: they are reported in the JSON so a
%   single bad geometry cannot kill a long optimization run.

t0 = tic;
try
    job = jsondecode(fileread(in_file));
    if ~isfield(job, 'opts') || isempty(job.opts), job.opts = struct(); end
    if ~isfield(job, 'params') || isempty(job.params), job.params = struct(); end
    fh  = str2func(job.case_function);
    res = fh(job.params, job.opts);
    res.status = 'ok';
catch err
    res = struct();
    res.status  = 'failed';
    res.message = getReport(err, 'extended', 'hyperlinks', 'off');
end
res.wall_time = toc(t0);

fid = fopen(out_file, 'w');
fprintf(fid, '%s', jsonencode(res));
fclose(fid);
end
