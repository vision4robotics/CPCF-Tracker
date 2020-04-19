function [seq, results] = get_sequence_results(seq,all_psr,all_Rmax,all_psrm)

results.psr = all_psr;
results.Rmax = all_Rmax;
results.psrm = all_psrm;
if strcmpi(seq.format, 'otb')
    results.type = 'rect';
    results.res = seq.rect_position;
elseif strcmpi(seq.format, 'vot')
    seq.handle.quit(seq.handle);
else
    error('Uknown sequence format');
end

if isfield(seq, 'time')
    results.fps = seq.num_frames / seq.time;
else
    results.fps = NaN;
    
end

