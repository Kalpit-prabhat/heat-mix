function W = channel_walls(g)
%CHANNEL_WALLS  Wall pieces of the sine channel with constant width normal to the path.
%
%   W = channel_walls(g)      (all lengths in mm)
%
%   g.L, g.Lt (= L + L_ext), g.xa, g.xb (start / end of the wavy section),
%   g.lam, g.A, g.phi, g.w
%
%   Centre line:  inlet port  y = ya          for 0  <= x <= xa
%                 sine        y = A*sin(2*pi*(x - xa)/lam + phi),  xa <= x <= xb
%                 outlet port y = yb          for xb <= x <= Lt
%   (ya, yb = sine values at xa, xb).  The ports join the sine directly, so the
%   centre line has a corner at xa and xb wherever the sine is not flat there.
%
%   Each wall lies w/2 from the centre line, measured along the local normal,
%   so the width normal to the flow is w everywhere.  At a joint corner the
%   wall on the outside of the turn is a circular arc of radius w/2 about the
%   joint; on the inside the port wall and the sine wall are cut where they
%   cross (sharp corner).  The walls fold over themselves (cusps) where the
%   bend radius of the sine, lam^2/(4*pi^2*A), is below w/2 -> error.
%
%   W.pieces{s}  pieces of wall s (1 = upper/left, 2 = lower/right), in order
%                from x = 0 to x = Lt, each a struct with .type 'line' | 'arc' |
%                'sine' and its numbers (see add_channel_curves)
%   W.upper, W.lower  sampled wall outlines (2 x n), for plots
%   W.ya, W.yb   port centre heights;  W.R_min  bend radius of the sine
%   W.turn_in, W.turn_out  turning angle of the centre line at xa / xb [deg]
%   W.path_length  centre-line length over 0..L
%   W.w_min      smallest channel width normal to the flow
%   W.cline      2 x n samples of the centre line, x increasing (for plots / profiles)
%
%   g.wall_mode (optional, default 0) picks the wall construction:
%     0  sine centre line, walls offset w/2 along its normal (above)
%     1  serpentine: triangle-wave centre line (pitch lam, crest height of the
%        filleted centre line = A) whose corners, including the joints to the
%        inlet port and the straight section, are rounded with arcs of radius
%        g.R_bend >= w/2.  The width normal to the flow is w everywhere and A is
%        not limited by lam (legs approach vertical as A grows); needs
%        lam > 4*R_bend.  phi is ignored (the wave starts at y = 0, going up).
%        The wave ends at the apex of the last crest at or before xb (y = +-A,
%        horizontal), so the straight section starts there; needs xb - xa >= lam/4.

mode = 0;
if isfield(g, 'wall_mode'), mode = round(g.wall_mode); end
if mode == 1
    W = serpentine_walls(g);  return
elseif mode ~= 0
    error('sine_channel:params', 'wall_mode must be 0 (sine) or 1 (serpentine)');
end

h = g.w/2;
k = 2*pi/g.lam;
th  = @(t) k*(t - g.xa) + g.phi;
yc  = @(t) g.A*sin(th(t));
mc  = @(t) g.A*k*cos(th(t));
W.ya = yc(g.xa);  W.yb = yc(g.xb);
ma = mc(g.xa);    mb = mc(g.xb);
if g.A > 0
    W.R_min = 1/(g.A*k^2);
else
    W.R_min = Inf;
end
if W.R_min < h
    error('sine_channel:geometry', ['Walls fold over at the crests: bend radius lam^2/(4 pi^2 A) = ' ...
        '%.3g mm < w/2 = %.3g mm'], W.R_min, h);
end
W.turn_in  = atand(ma);
W.turn_out = -atand(mb);
tol = 1e-9;

ts = linspace(g.xa, g.xb, 40001);         % for root brackets
names = {'upper', 'lower'};
for si = 1:2
    s = 3 - 2*si;                         % +1 upper (left of the flow), -1 lower
    P = {};
    % ---- inlet joint: port direction (1,0) -> sine; left turn if ma > 0
    turn = sign(ma) * (abs(ma) > tol);
    if turn == s                          % inside of the turn: cut where the walls cross
        t0 = cross_root(ts, @(t) yc(t) + s*h./sqrt(1 + mc(t).^2) - (W.ya + s*h), 'first');
        p = offset_pt(t0, s);
        if p(1) <= 1e-3
            error('sine_channel:geometry', 'Inlet joint corner at x = %.3g mm, outside the inlet port', p(1));
        end
        P{end+1} = line_piece([0, W.ya + s*h], [p(1), W.ya + s*h]); %#ok<AGROW>
    else
        t0 = g.xa;
        P{end+1} = line_piece([0, W.ya + s*h], [g.xa, W.ya + s*h]); %#ok<AGROW>
        if turn ~= 0                      % outside of the turn: arc about the joint
            P{end+1} = arc_piece([g.xa, W.ya], s*pi/2, normal_angle(ma, s)); %#ok<AGROW>
        end
    end
    % ---- outlet joint: sine -> port direction (1,0); left turn if mb < 0
    turn = -sign(mb) * (abs(mb) > tol);
    if turn == s
        t1 = cross_root(ts, @(t) yc(t) + s*h./sqrt(1 + mc(t).^2) - (W.yb + s*h), 'last');
        p = offset_pt(t1, s);
        if p(1) >= g.L - 1e-3
            error('sine_channel:geometry', 'Outlet joint corner at x = %.3g mm, past the exit plane x = L', p(1));
        end
        tail = {line_piece([p(1), W.yb + s*h], [g.Lt, W.yb + s*h])};
    else
        t1 = g.xb;
        tail = {line_piece([g.xb, W.yb + s*h], [g.Lt, W.yb + s*h])};
        if turn ~= 0
            tail = [{arc_piece([g.xb, W.yb], normal_angle(mb, s), s*pi/2)}, tail];
        end
    end
    if t1 <= t0
        error('sine_channel:geometry', 'Wavy section too short for the joint corners');
    end
    P{end+1} = struct('type', 'sine', 't', [t0, t1], 's', s, 'h', h, 'A', g.A, 'k', k, ...
        'xa', g.xa, 'phi', g.phi); %#ok<AGROW>
    P = [P, tail]; %#ok<AGROW>
    W.pieces{si} = P;
    W.(names{si}) = sample_pieces(P);
end
% centre-line length over the device
x = linspace(g.xa, g.xb, 40001);
W.path_length = g.xa + trapz(x, sqrt(1 + mc(x).^2)) + (g.L - g.xb);
W.w_min = g.w;
xs_ = linspace(g.xa, g.xb, 2001);
W.cline = [0, xs_, g.Lt; W.ya, yc(xs_), W.yb];   % centre line samples (x increasing)

    function p = offset_pt(t, s)
        m = mc(t); n = sqrt(1 + m^2);
        p = [t - s*h*m/n, yc(t) + s*h/n];
    end

    function a = normal_angle(m, s)
        % direction of s * unit normal (-m, 1)/sqrt(1+m^2)
        a = atan2(s, -s*m);
    end

    function A = arc_piece(c, a0, a1)
        if a1 - a0 > pi,  a1 = a1 - 2*pi; end
        if a0 - a1 > pi,  a1 = a1 + 2*pi; end
        A = struct('type', 'arc', 'c', c, 'r', h, 'a', [a0, a1]);
    end
end

function L = line_piece(p, q)
L = struct('type', 'line', 'p', p, 'q', q);
end

function t = cross_root(ts, f, which)
v = f(ts);
i = find(sign(v(1:end-1)) ~= sign(v(2:end)) & v(1:end-1) ~= 0);
if isempty(i)
    error('sine_channel:geometry', 'Joint corner not found');
end
if strcmp(which, 'first'), i = i(1); else, i = i(end); end
t = fzero(f, [ts(i), ts(i+1)]);
end

function xy = sample_pieces(P)
xy = zeros(2, 0);
for i = 1:numel(P)
    p = P{i};
    switch p.type
        case 'line'
            q = [p.p(:), p.q(:)];
        case 'arc'
            a = linspace(p.a(1), p.a(2), 30);
            q = [p.c(1) + p.r*cos(a); p.c(2) + p.r*sin(a)];
        case 'sine'
            t = linspace(p.t(1), p.t(2), max(200, ceil(40*diff(p.t))));
            m = p.A*p.k*cos(p.k*(t - p.xa) + p.phi); n = sqrt(1 + m.^2);
            q = [t - p.s*p.h*m./n; p.A*sin(p.k*(t - p.xa) + p.phi) + p.s*p.h./n];
    end
    xy = [xy, q]; %#ok<AGROW>
end
end

function W = serpentine_walls(g)
% wall_mode 1: filleted triangle wave, constant width normal to the flow
h = g.w/2;
R = g.R_bend;
lam = g.lam;  A = g.A;
if R < h + 1e-6
    error('sine_channel:geometry', 'Bend radius R_bend = %.3g mm must exceed w/2 = %.3g mm', R, h);
end
if g.xb <= g.xa, error('sine_channel:geometry', 'Wavy section has no length'); end
if A <= 1e-9                                   % straight channel
    V = 0;  alpha = 0;
else
    if lam <= 4*R
        error('sine_channel:geometry', 'Serpentine needs lam > 4*R_bend (lam %.3g, R_bend %.3g mm)', lam, R);
    end
    % crest vertex height V = A + R*(1/cos(alpha) - 1), leg slope tan(alpha) = 4*V/lam
    f = @(a) 4*(A + R*(1/cos(a) - 1))/lam - tan(a);
    alpha = fzero(f, [1e-12, pi/2 - 1e-9]);
    V = A + R*(1/cos(alpha) - 1);
end
% vertices of the centre-line polyline: port start, wave start (xa, 0), crest
% vertices x_j = xa + lam/4 + j*lam/2 up to xb, plus the next one (it only sets
% the turn of the last crest).  The wave ends at the apex of the last crest
% (horizontal tangent, y = +-A) and the straight section follows from there.
if V > 0
    vx = [0, g.xa];  vy = [0, 0];
    j = 0;
    while g.xa + lam/4 + j*lam/2 <= g.xb + 1e-9
        vx(end+1) = g.xa + lam/4 + j*lam/2;  vy(end+1) = V*(1 - 2*mod(j, 2)); %#ok<AGROW>
        j = j + 1;
    end
    if j == 0
        error('sine_channel:geometry', ['Serpentine: wavy section (%.3g mm) shorter than ' ...
            'lam/4 = %.3g mm (no crest)'], g.xb - g.xa, lam/4);
    end
    vx(end+1) = g.xa + lam/4 + j*lam/2;  vy(end+1) = V*(1 - 2*mod(j, 2));
    [~, arcs] = fillet([vx; vy], R);
    a = arcs(end);                             % last crest: keep its first half
    a.phi = a.phi/2;
    a.t2 = a.c + R*[cos(a.a0 + a.phi); sin(a.a0 + a.phi)];
    a.d2 = [1; 0];
    arcs(end) = a;
    pts = [[0; 0], [g.Lt; a.t2(2)]];
else                                           % A = 0: straight channel
    pts = [0, g.Lt; 0, 0];
    arcs = struct('c', {}, 'R', {}, 'a0', {}, 'phi', {}, 't1', {}, 't2', {}, 'd1', {}, 'd2', {});
end
W.ya = 0;  W.yb = pts(2, end);
% walls: offset the straight segments and the arcs
for si = 1:2
    s = 3 - 2*si;                              % +1 left of the flow (upper), -1 right
    P = {};
    p0 = pts(:, 1) + s*h*[0; 1];
    for i = 1:numel(arcs)
        a = arcs(i);
        nrm = [-a.d1(2); a.d1(1)];             % left normal of the incoming direction
        p1 = a.t1 + s*h*nrm;
        if norm(p1 - p0) > 1e-9, P{end+1} = line_piece(p0', p1'); end %#ok<AGROW>
        r = a.R - s*sign(a.phi)*h;             % inside of a left turn = left wall
        P{end+1} = struct('type', 'arc', 'c', a.c', 'r', r, 'a', [a.a0, a.a0 + a.phi]); %#ok<AGROW>
        n2 = [-a.d2(2); a.d2(1)];
        p0 = a.t2 + s*h*n2;
    end
    pe = pts(:, end) + s*h*[0; 1];
    P{end+1} = line_piece(p0', pe'); %#ok<AGROW>
    W.pieces{si} = P;
    if si == 1, W.upper = sample_pieces(P); else, W.lower = sample_pieces(P); end
end
W.R_min = R;
if isempty(arcs)
    W.turn_in = 0;  W.turn_out = 0;
else
    W.turn_in  = rad2deg(arcs(1).phi);
    W.turn_out = -rad2deg(arcs(end).phi);
end
W.alpha_deg = rad2deg(alpha);
% centre-line length over 0..L
len = 0;  q = pts(:, 1);
for i = 1:numel(arcs)
    len = len + norm(arcs(i).t1 - q) + R*abs(arcs(i).phi);
    q = arcs(i).t2;
end
W.path_length = len + (g.L - q(1));
W.w_min = g.w;
% centre line samples (x increasing: legs never pass vertical)
C = pts(:, 1);
for i = 1:numel(arcs)
    C = [C, arcs(i).t1]; %#ok<AGROW>
    t = linspace(arcs(i).a0, arcs(i).a0 + arcs(i).phi, 40);
    C = [C, [arcs(i).c(1) + R*cos(t); arcs(i).c(2) + R*sin(t)]]; %#ok<AGROW>
end
C = [C, pts(:, end)];
[~, iu] = unique(C(1, :), 'stable');
W.cline = C(:, iu);
end

function [pts, arcs] = fillet(pts, R)
% round every interior corner of the polyline pts (2 x n) with radius R;
% errors if the tangent points of two neighbouring bends overlap
n = size(pts, 2);
arcs = struct('c', {}, 'R', {}, 'a0', {}, 'phi', {}, 't1', {}, 't2', {}, 'd1', {}, 'd2', {});
tl = zeros(1, n);
for i = 2:n-1
    d1 = pts(:, i) - pts(:, i-1);  d1 = d1/norm(d1);
    d2 = pts(:, i+1) - pts(:, i);  d2 = d2/norm(d2);
    phi = atan2(d1(1)*d2(2) - d1(2)*d2(1), d1'*d2);   % signed turn (+ = left)
    if abs(phi) < 1e-9, continue; end
    tl(i) = R*tan(abs(phi)/2);
    t1 = pts(:, i) - tl(i)*d1;  t2 = pts(:, i) + tl(i)*d2;
    c = t1 + sign(phi)*R*[-d1(2); d1(1)];
    a0 = atan2(t1(2) - c(2), t1(1) - c(1));
    arcs(end+1) = struct('c', c, 'R', R, 'a0', a0, 'phi', phi, 't1', t1, 't2', t2, ...
        'd1', d1, 'd2', d2); %#ok<AGROW>
end
seg = vecnorm(diff(pts, 1, 2));
need = tl(1:end-1) + tl(2:end);
bad = find(seg < need - 1e-9);
if ~isempty(bad)
    error('sine_channel:geometry', ['Serpentine bends do not fit (segment %d: %.3g mm < %.3g mm ' ...
        'of fillet tangents); lam too short for this A / R_bend, or the inlet port too short'], ...
        bad(1), seg(bad(1)), need(bad(1)));
end
end
