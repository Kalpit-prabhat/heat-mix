function tags = add_channel_curves(G, W, cuts)
%ADD_CHANNEL_CURVES  Create the channel outline from channel_walls() in a 2D
%   geometry sequence G (a 2D geom, or the geom of a 3D work plane), plus the
%   cross lines that close / split the channel.  Removes all existing
%   features of G first, so the outline can be rebuilt for every design.
%
%   cuts : n x 3 rows [x, y_low, y_high] of the straight cross lines
%   tags : cell of the created feature tags (input for ConvertToSolid)

old = cell(G.feature.tags);
for i = 1:numel(old)
    if strcmp(old{i}, 'fin'), continue; end      % Form Union of a 2D component
    G.feature.remove(old{i});
end
tags = {};
fmt = '%.15g';
for si = 1:numel(W.pieces)
    P = W.pieces{si};
    for i = 1:numel(P)
        p = P{i};
        tag = sprintf('w%d_%d', si, i);
        switch p.type
            case 'line'
                f = G.create(tag, 'LineSegment');
                f.set('specify1', 'coord'); f.set('coord1', p.p);
                f.set('specify2', 'coord'); f.set('coord2', p.q);
            case 'arc'
                f = G.create(tag, 'ParametricCurve');
                f.set('parname', 'sa');
                f.set('parmin', sprintf(fmt, min(p.a))); f.set('parmax', sprintf(fmt, max(p.a)));
                f.set('coord', {sprintf([fmt '+' fmt '*cos(sa)'], p.c(1), p.r), ...
                                sprintf([fmt '+' fmt '*sin(sa)'], p.c(2), p.r)});
            case 'sine'
                % wall = centre line + s*h*(-m, 1)/sqrt(1+m^2), parameter sp = x of the centre line
                arg = sprintf(['(' fmt '*(sp-' fmt ')+' fmt ')'], p.k, p.xa, p.phi);
                m   = sprintf(['(' fmt '*cos%s)'], p.A*p.k, arg);
                nrm = sprintf('sqrt(1+%s^2)', m);
                f = G.create(tag, 'ParametricCurve');
                f.set('parname', 'sp');
                f.set('parmin', sprintf(fmt, p.t(1))); f.set('parmax', sprintf(fmt, p.t(2)));
                f.set('coord', {sprintf(['sp-(' fmt ')*%s/%s'], p.s*p.h, m, nrm), ...
                                sprintf([fmt '*sin%s+(' fmt ')/%s'], p.A, arg, p.s*p.h, nrm)});
                f.set('maxknots', 4000);
        end
        tags{end+1} = tag; %#ok<AGROW>
    end
end
for i = 1:size(cuts, 1)
    tag = sprintf('cut%d', i);
    f = G.create(tag, 'LineSegment');
    f.set('specify1', 'coord'); f.set('coord1', cuts(i, [1 2]));
    f.set('specify2', 'coord'); f.set('coord2', cuts(i, [1 3]));
    tags{end+1} = tag; %#ok<AGROW>
end
end
