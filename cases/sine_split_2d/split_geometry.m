function G = split_geometry(g)
%SPLIT_GEOMETRY  Outline of the wavy -> straight -> N-branch fan channel.
%
%   G = split_geometry(g)        pure MATLAB (no COMSOL), all lengths in mm
%
%   Layout along x (flow direction):
%     0 .. xa            straight inlet port (L_port), centre at y = ya
%     xa .. xb           wavy section (sine or serpentine), length f_sine*L_main
%                        (constant width w, walls from channel_walls.m)
%     xb .. xs           straight section, length (1 - f_sine)*L_main, centre y = yb
%     xs .. xf           knife-edge fan: the width w is cut at x = xs into N branches
%                        of widths w_i = w*r_i/sum(r) (sharp dividers), each running
%                        straight to its outlet port of width w_out = w/N (the
%                        outlet widths add up to the main channel width); port centres
%                        c_i = yb + (i - (N+1)/2)*p_out
%     xf .. L            parallel straight outlet ports (L_out); exit line x = L
%     L  .. Lt           adiabatic extension (L_ext), outflow at x = Lt
%
%   g fields: L_port, L_main, f_sine, lam, A, phi, w, wall_mode, R_bend (see
%             channel_walls.m), L_fan, L_out, L_ext, n_br, p_out, r (1 x n_br
%             branch inlet width weights)
%
%   G.pieces  wall pieces for add_channel_curves ({upper, lower, branch walls})
%   G.cuts    rows [x, y_lo, y_hi] of the straight cross lines; G.cut_kind the kind
%             of each row: 'in' | 'xc' | 'split' | 'fan' | 'meas' | 'out'
%   G.xc      cut between the wavy and the straight domain (just past the joint corner)
%   G.a       (N+1) split points across the width at x = xs;  G.wb branch inlet widths
%   G.c       branch centre heights
%   G.lines   cell of 2 x n wall polylines (walls, branch walls), for plots

N = round(g.n_br);
if N < 1, error('sine_split:params', 'n_br must be >= 1'); end
r = g.r(1:N);
if any(r <= 0), error('sine_split:params', 'branch width weights must be > 0'); end
G.N = N;
G.xa = g.L_port;
G.L_sine = g.f_sine * g.L_main;
G.L_str = g.L_main - G.L_sine;
G.xb = G.xa + G.L_sine;
G.xs = G.xa + g.L_main;
G.xf = G.xs + g.L_fan;
G.L  = G.xf + g.L_out;
G.Lt = G.L + g.L_ext;
h = g.w/2;

% ---- sine + straight walls (channel_walls: outlet port = the straight section)
gm = struct('L', G.xs, 'Lt', G.xs, 'xa', G.xa, 'xb', G.xb, 'lam', g.lam, ...
    'A', g.A, 'phi', g.phi, 'w', g.w, 'wall_mode', g.wall_mode, 'R_bend', g.R_bend);
W = channel_walls(gm);
G.W = W;
G.ya = W.ya;  G.yb = W.yb;
x_tail = max(W.pieces{1}{end}.p(1), W.pieces{2}{end}.p(1));   % both walls straight from here
if g.wall_mode == 1
    % serpentine: the wave ends (smoothly, at a crest apex) at x_tail <= xb, so the
    % straight domain starts there
    G.xc = x_tail + 0.02;
else
    G.xc = max(x_tail, G.xb) + 0.02;
end
if G.xc > G.xs - 0.1
    error('sine_split:geometry', 'Straight section too short: joint corner at x = %.3g mm, split at %.3g mm', ...
        x_tail, G.xs);
end

% ---- fan + outlet ports
G.wb = g.w * r(:)' / sum(r);
G.a  = G.yb - h + [0, cumsum(G.wb)];
G.a(end) = G.yb + h;                     % exact
G.c  = G.yb + ((1:N) - (N + 1)/2) * g.p_out;
G.w_out = g.w / N;                       % outlet widths sum to the main channel width
if N > 1 && g.p_out <= G.w_out
    error('sine_split:params', 'Outlet pitch p_out must exceed the branch width w_out');
end
ho = G.w_out/2;
B = {};
for i = 1:N
    B{end+1} = line_piece([G.xs, G.a(i)],   [G.xf, G.c(i) - ho]); %#ok<AGROW>
    B{end+1} = line_piece([G.xf, G.c(i) - ho], [G.Lt, G.c(i) - ho]); %#ok<AGROW>
    B{end+1} = line_piece([G.xs, G.a(i+1)], [G.xf, G.c(i) + ho]); %#ok<AGROW>
    B{end+1} = line_piece([G.xf, G.c(i) + ho], [G.Lt, G.c(i) + ho]); %#ok<AGROW>
end
G.pieces = {W.pieces{1}, W.pieces{2}, B};

% ---- cross lines
cuts = [0, G.ya - h, G.ya + h; G.xc, G.yb - h, G.yb + h];
kind = {'in', 'xc'};
for i = 1:N
    cuts(end+1, :) = [G.xs, G.a(i), G.a(i+1)]; kind{end+1} = 'split'; %#ok<AGROW>
end
for i = 1:N
    cuts(end+1, :) = [G.xf, G.c(i) - ho, G.c(i) + ho]; kind{end+1} = 'fan'; %#ok<AGROW>
end
for i = 1:N
    cuts(end+1, :) = [G.L, G.c(i) - ho, G.c(i) + ho]; kind{end+1} = 'meas'; %#ok<AGROW>
end
if G.Lt > G.L + 1e-9
    for i = 1:N
        cuts(end+1, :) = [G.Lt, G.c(i) - ho, G.c(i) + ho]; kind{end+1} = 'out'; %#ok<AGROW>
    end
end
G.cuts = cuts;
G.cut_kind = kind;
G.n_domains = 2 + N*(2 + (G.Lt > G.L + 1e-9));

% ---- polylines for plots
L = {W.upper, W.lower};
for i = 1:N
    L{end+1} = [G.xs, G.xf, G.Lt; G.a(i),   G.c(i) - ho, G.c(i) - ho]; %#ok<AGROW>
    L{end+1} = [G.xs, G.xf, G.Lt; G.a(i+1), G.c(i) + ho, G.c(i) + ho]; %#ok<AGROW>
end
G.lines = L;
G.y_range = [min([G.c - ho, W.lower(2,:)]), max([G.c + ho, W.upper(2,:)])];
end

function L = line_piece(p, q)
L = struct('type', 'line', 'p', p, 'q', q);
end
