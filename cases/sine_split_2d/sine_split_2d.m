function res = sine_split_2d(P, opts)
%SINE_SPLIT_2D  2D (x-y) channel: wavy section (sine or serpentine) -> straight
%   section -> optional knife-edge fan into N branches.  Used for every 2D case of
%   the final study: one outlet (n_br = 1) and four branches (n_br = 4).
%
%   Geometry (split_geometry.m, all lengths in mm):
%     inlet port L_port | wavy section f_sine*L_main (lam, A, phi, width w; wall_mode
%       0 = sine, 1 = serpentine with bend radius R_bend, see channel_walls.m)
%     | straight section (1 - f_sine)*L_main
%     | fan L_fan: the width w is cut into N = n_br branches of inlet widths in the
%       ratio 1 : rho_br : ... : rho_br : 1 (outer : inner branches) that run straight
%       to outlet ports of width w_out = w/n_br (so the outlet widths add up to w) at
%       pitch p_out
%     | outlet ports L_out (exit lines x = L, where everything is measured)
%     | adiabatic extension L_ext (p = 0 at its end).
%   L_main = wavy + straight is fixed; f_sine is the wavy fraction.  With n_br = 1
%   the "fan" and the outlet port are simply the continuation of the straight channel.
%
%   Heaters (side walls, wall held at a fixed temperature, flux h_heat*(T_h - T)):
%     heater 1 on the wavy section, from heat1_s to heat1_s + heat1_l (fractions of
%              its length, from its start; heat1_abs = 1: fractions of the whole main
%              channel xa..xs instead, used for the straight channel); temperature
%              T_heat, solved (global equation) so the mixed bulk temperature of all
%              outlets at x = L is T_need = T_in + dT_target, limited to
%              [T_need, T_heat_max]
%     heater 2 on the straight section, heat2_s / heat2_l as fractions of its length
%              (heat2_l = 0: no heater 2; below 2*dxs its strength fades linearly to 0);
%              temperature T_need + dT_h2, limited to [T_need, T_heat_max]
%   No heater is ever below T_need (no cooling).  Other walls lose h_l*(T_amb - T),
%   top/bottom walls (depth) as the volume loss 2*h_l*(T_amb - T)/depth, for x <= L;
%   h_l = h_loss up to the split line x = xs, h_loss_br in the branches.
%
%   Outputs: res.T / res.un on the exit lines of all branches (concatenated, branch 1
%   = lowest y), integrals and diagnostics, plus per branch
%     res.branch_Q_ml_min, res.branch_Tb, res.branch_T_rms, res.branch_w_in
%
%   opts: .mesh  structured quads: .n_b elements across every branch (n_br*n_b across
%                the main channel), graded towards the walls by .w_ratio, element
%                size .hmax [mm] along the flow
%         .ny    outlet samples per branch (default 40)
%         .heat_list  rows [heat1_s heat1_l]: heat-only re-solves on the same flow
%                and mesh, outlet of each in res.sweep
%         .flow_ramp  flow-rate continuation of the flow step (factors of Qv,
%                default '0.1 0.4 1')
%         .reuse_model, .profile_nb, .fields, .save_mph, .verbose: as sine_channel.m
%
%   Release a kept model: sine_split_2d([], struct('release', true));

import com.comsol.model.*
import com.comsol.model.util.*
persistent cache

if nargin < 1 || isempty(P),    P = struct();    end
if nargin < 2 || isempty(opts), opts = struct(); end
mesh_opts = getopt(opts, 'mesh', struct());
ny        = getopt(opts, 'ny', 40);
save_mph  = getopt(opts, 'save_mph', '');
verbose   = getopt(opts, 'verbose', true);
reuse     = logical(getopt(opts, 'reuse_model', false));
ramp      = getopt(opts, 'flow_ramp', '0.1 0.4 1');

if ~isempty(cache) && (~reuse || getopt(opts, 'release', false) || ...
        ~any(strcmp(cell(ModelUtil.tags()), cache.tag)))
    try ModelUtil.remove(cache.tag); catch, end
    cache = [];
end
if getopt(opts, 'release', false), res = struct(); return; end

t_start = tic;
tag = '';
try
    %% ---------------- Model / parameters ----------------
    if ~isempty(cache)
        tag = cache.tag;
        model = ModelUtil.model(tag);
    else
        [tag, model] = new_model();
    end
    set_params(model, P);
    g = eval_params(model);
    G = split_geometry(g);               % errors on an impossible geometry
    has_ext = G.Lt > G.L + 1e-9;
    skey = jsonencode(struct('mesh', mesh_opts, 'ext', has_ext, 'ramp', ramp, 'N', G.N));
    if ~isempty(cache) && ~strcmp(cache.key, skey)
        ModelUtil.remove(tag); cache = [];
        [tag, model] = new_model();
        set_params(model, P);
        g = eval_params(model);
        G = split_geometry(g);
    end
    fresh = isempty(cache);
    set_derived(model, G);

    %% ---------------- Geometry / physics / mesh ----------------
    if fresh
        comp = model.component.create('comp1', true);
        geom = comp.geom.create('geom1', 2);
        geom.lengthUnit('mm');
    end
    comp = model.component('comp1');
    geom = comp.geom('geom1');
    set_walls(geom, G);
    geom.run;
    if geom.getNDomains ~= G.n_domains
        error('sine_split:geometry', 'Geometry has %d domains, expected %d (self-intersecting walls?)', ...
            geom.getNDomains, G.n_domains);
    end
    if fresh
        build_physics(model);
        create_mesh(comp);
    end
    update_selections(comp, G, g);
    ids = check_selections(comp, G);
    say('geometry ok: wavy %.2f mm + straight %.2f mm, %d branch(es) (inlet widths %s mm)%s', ...
        G.L_sine, G.L_str, G.N, mat2str(G.wb, 3), ternary(fresh, '', ' [reused model]'));
    set_mesh(comp, mesh_opts, G);
    t_mesh = tic;
    comp.mesh('mesh1').run;
    t_mesh = toc(t_mesh);
    st = mphmeshstats(model, 'mesh1');
    types = cellstr(st.types);
    n_elem = sum(double(st.numelem(ismember(types, {'tri', 'quad'}))));
    if fresh
        build_study(model, ramp);
    end
    say('mesh: %d elements', n_elem);

    %% ---------------- Solve ----------------
    t_solve = tic;
    model.sol('sol1').runAll;
    t_solve = toc(t_solve);
    say('solved in %.1f s', t_solve);

    %% ---------------- Outlet data (exit lines x = L) ----------------
    N = G.N;
    eta = ((1:ny) - 0.5) / ny - 0.5;
    Y = G.c(:) + G.w_out * eta;                         % N x ny
    coords = [G.L * ones(1, N*ny); reshape(Y', 1, [])];
    [Tg, ung] = mphinterp(model, {'T', 'u'}, 'coord', coords, ...
        'edim', 'boundary', 'selection', ids.meas, 'unit', {'K', 'm/s'});
    depth = mphevaluate(model, 'depth', 'mm');
    dm = depth * 1e-3;

    res = struct();
    res.n_branch  = N;
    res.grid_eta  = reshape((((1:N)' - 1) + (eta + 0.5)) / N - 0.5, 1, []);  % branches side by side
    res.grid_y_mm = reshape(Y', 1, []);
    res.grid_z_mm = depth / 2;
    res.T  = Tg(:);
    res.un = ung(:);
    res.symmetric = false;
    res.model_dim = 2;

    I = @(op, e) mphglobal(model, sprintf('%s(%s)', op, e));
    Aout = I('intop_meas', '1+0*T');        % '+0*T' forces SI units in post-processing
    Qout = I('intop_meas', 'u');
    res.T_mean_area = I('intop_meas', 'T') / Aout;
    res.T_bulk      = I('intop_meas', 'u*T') / Qout;
    res.T_rms_area  = sqrt(I('intop_meas', '(T - T_need)^2') / Aout);
    res.T_rms_flow  = sqrt(I('intop_meas', 'max(u,0)*(T - T_need)^2') / I('intop_meas', 'max(u,0)'));
    res.T_max = max(Tg); res.T_min = min(Tg);
    res.T_need = mphevaluate(model, 'T_need', 'K');
    res.T_in   = mphevaluate(model, 'T_in', 'K');
    res.dT_target = mphevaluate(model, 'dT_target', 'K');

    % per branch: exact integrals over the exit lines, one branch picked by its y window
    tol = G.w_out/2 + 0.25*max(g.p_out - G.w_out, 0);
    res.branch_Q_ml_min = zeros(1, N); res.branch_Tb = zeros(1, N); res.branch_T_rms = zeros(1, N);
    for i = 1:N
        m = sprintf('(abs(y - %.12g[mm]) < %.12g[mm])', G.c(i), tol);
        qi = I('intop_meas', ['u*' m]);
        res.branch_Q_ml_min(i) = qi * dm * 6e7;
        res.branch_Tb(i)       = I('intop_meas', ['u*T*' m]) / qi;
        res.branch_T_rms(i)    = sqrt(I('intop_meas', ['(T - T_need)^2*' m]) / I('intop_meas', [m '+0*T']));
    end
    res.branch_w_in = G.wb;
    res.branch_Q_dev = res.branch_Q_ml_min / mean(res.branch_Q_ml_min) - 1;
    res.branch_Tb_dev = res.branch_Tb - res.T_need;

    res.Q_out_ml_min = Qout * dm * 6e7;
    res.dp_Pa = I('intop_in', 'p') / I('intop_in', '1+0*T');
    res.P_tot_W  = mphevaluate(model, 'P_tot', 'W');
    res.P_loss_W = -(I('intop_side', '(1 - hw1 - hw2)*q_loss') + I('intop_dom', 'Q_loss')) * dm;
    res.P_heater_W = I('intop_meas', 'rho0*Cp0*u*(T - T_in)') * dm + res.P_loss_W;   % energy balance
    res.P_factor   = res.P_heater_W / res.P_tot_W;
    res.T_heater   = mphglobal(model, 'T_heat');
    res.T_heater2  = mphglobal(model, 'T_heat2');
    th = mphglobal(model, 'th_heat');
    th_max = mphevaluate(model, '(T_heat_max - T_in)/dT_target');
    res.power_clamped = (th <= 1 + 1e-6 || th >= th_max - 1e-6) && abs(res.T_bulk - res.T_need) > 1e-3;
    res.A_heat_mm2 = I('intop_side', 'hw1 + hw2 + 0*T') * 1e3 * depth;
    res.T_wall_max = mphmax(model, 'T', 'line', 'selection', ids.side);
    res.u_max      = mphmax(model, 'spf.U', 'surface');
    res.Re         = mphevaluate(model, 'rho0*Qv/(w*depth)*2*w*depth/(w+depth)/mu0');
    res.path_length_mm     = G.W.path_length + (G.L - G.xs);
    res.min_gap_mm         = min([G.wb, G.w_out]);
    res.min_curv_radius_mm = G.W.R_min;
    res.w_min_mm           = G.W.w_min;           % narrowest width normal to the flow
    res.joint_turn_deg     = [G.W.turn_in, G.W.turn_out];
    res.L_sine_mm = G.L_sine;  res.L_str_mm = G.L_str;  res.L_mm = G.L;
    res.mesh_elements = n_elem;
    npx = getopt(opts, 'profile_nb', 60);
    if npx > 0
        res.profile = x_profile(model, G, g, npx, 30, res.T_need);
    end
    res.t_mesh  = t_mesh;
    res.t_solve = t_solve;
    res.t_total = toc(t_start);

    if isfield(opts, 'fields') && isstruct(opts.fields)
        say('exporting field data ...');
        res.fields = export_fields(model, G, g, getopt(opts, 'fields', struct()));
    end
    if ~isempty(save_mph)
        mphsave(model, save_mph);
    end
    % ---- heater sweep (heat step only; flow and mesh unchanged) ----
    % opts.heat_list: rows [heat1_s heat1_l] (same fractions as the parameters).  For
    % each row heater 1 is moved and only the heat step is solved again; the outlet
    % temperatures go to res.sweep (comsol_opt scores each and keeps the best).
    hlist = getopt(opts, 'heat_list', []);
    if ~isempty(hlist)
        hlist = reshape(hlist, [], 2);
        sol = model.sol('sol1');
        stt = cell(sol.feature.tags);  stt = stt(startsWith(stt, 'st'));
        heat_st = stt{end};
        s0 = char(model.param.get('heat1_s'));  l0 = char(model.param.get('heat1_l'));
        sw = struct('heat1_s', {}, 'heat1_l', {}, 'T', {}, 'T_bulk', {}, 'T_heater', {}, ...
            'power_clamped', {}, 'branch_Tb', {}, 't_heat', {});
        t_sw = tic;
        for k = 1:size(hlist, 1)
            if hlist(k,1) < 0 || hlist(k,2) <= 0 || sum(hlist(k,:)) > 1 + 1e-9, continue; end
            t1 = tic;
            model.param.set('heat1_s', sprintf('%.10g', hlist(k,1)));
            model.param.set('heat1_l', sprintf('%.10g', hlist(k,2)));
            sol.runFrom(heat_st);
            e = struct('heat1_s', hlist(k,1), 'heat1_l', hlist(k,2));
            e.T = mphinterp(model, 'T', 'coord', coords, 'edim', 'boundary', 'selection', ids.meas, 'unit', 'K');
            e.T = e.T(:);
            e.T_bulk = I('intop_meas', 'u*T') / Qout;
            e.T_heater = mphglobal(model, 'T_heat');
            thk = mphglobal(model, 'th_heat');
            e.power_clamped = (thk <= 1 + 1e-6 || thk >= th_max - 1e-6) && abs(e.T_bulk - res.T_need) > 1e-3;
            e.branch_Tb = zeros(1, N);
            for i = 1:N
                m = sprintf('(abs(y - %.12g[mm]) < %.12g[mm])', G.c(i), tol);
                e.branch_Tb(i) = I('intop_meas', ['u*T*' m]) / I('intop_meas', ['u*' m]);
            end
            e.t_heat = toc(t1);
            sw(end+1) = e; %#ok<AGROW>
        end
        model.param.set('heat1_s', s0);  model.param.set('heat1_l', l0);
        sol.runFrom(heat_st);                 % leave the model at the main design
        res.sweep = sw;
        res.t_sweep = toc(t_sw);
        say('heater sweep: %d positions in %.1f s', numel(sw), res.t_sweep);
    end
    say(['done: branch Q dev %s %%, branch Tb - T_need %s mK, heaters %.2f / %.2f C, ' ...
         'heater %.3g W, losses %.3g W, total %.1f s'], mat2str(100*res.branch_Q_dev, 3), ...
        mat2str(1e3*res.branch_Tb_dev, 3), res.T_heater - 273.15, res.T_heater2 - 273.15, ...
        res.P_heater_W, res.P_loss_W, res.t_total);

    if reuse
        cache = struct('tag', tag, 'key', skey);
    else
        ModelUtil.remove(tag);
    end
catch err
    if ~isempty(tag)
        try ModelUtil.remove(tag); catch, end
    end
    cache = [];
    rethrow(err);
end

    function say(varargin)
        if verbose
            fprintf('[%s] sine_split_2d: %s\n', datestr(now, 'HH:MM:SS'), sprintf(varargin{:}));
        end
    end
end

%% ======================= model construction =======================
function [tag, model] = new_model()
tag = sprintf('SineSplit2D_%09d', randi(1e9 - 1));
model = com.comsol.model.util.ModelUtil.create(tag);
model.hist.disable;
end

function set_params(model, P)
% defaults: the final-study branched serpentine (cases/final/b2d_serp.yaml)
defaults = {
    'L_port',      '2[mm]',             'Straight inlet port length'
    'L_main',      '24[mm]',            'Wavy + straight section length (fixed)'
    'f_sine',      '0.6',               'Fraction of L_main that is the wavy section'
    'lam',         '8[mm]',             'Wavelength'
    'A',           '3[mm]',             'Amplitude of the centre line'
    'phi',         '0',                 'Phase of the sine at its start [rad] (wall_mode 0)'
    'wall_mode',   '1',                 'Walls: 0 sine + normal offset, 1 serpentine (arcs R_bend + legs)'
    'R_bend',      '1.55[mm]',          'Centre-line bend radius of the serpentine (wall_mode 1), > w/2'
    'w',           '3[mm]',             'Main channel width, normal to the path'
    'depth',       '3[mm]',             'Channel depth (z), out-of-plane'
    'L_fan',       '6[mm]',             'Fan length (split line to outlet ports)'
    'L_out',       '2[mm]',             'Straight outlet port length (exit lines at their end)'
    'L_ext',       '3[mm]',             'Adiabatic extension after the exit lines, 0 = none'
    'n_br',        '4',                 'Number of branches'
    'p_out',       '4[mm]',             'Outlet port pitch (centre to centre)'
    'rho_br',      '1',                 'Inlet width ratio inner / outer branches'
    'heat1_s',     '0',                 'Heater 1 start, fraction of the wavy section'
    'heat1_l',     '1',                 'Heater 1 length, fraction of the wavy section'
    'heat1_abs',   '0',                 '1 = heater 1 fractions refer to the whole main channel xa..xs'
    'heat2_s',     '0',                 'Heater 2 start, fraction of the straight section'
    'heat2_l',     '0',                 'Heater 2 length, fraction of the straight section'
    'dT_h2',       '0[K]',              'Heater 2 temperature - T_need (>= 0, no cooling)'
    'dxs',         '0.1[mm]',           'Half-width of the smoothed heater edge'
    'Qv',          '6[ml/min]',         'Volumetric flow rate'
    'T_in',        '343.15[K]',         'Inlet temperature'
    'dT_target',   '2[K]',              'Required temperature rise'
    'h_loss',      '2[W/(m^2*K)]',      'Heat-loss coefficient of the non-heated walls (x < xs)'
    'h_loss_br',   '10[W/(m^2*K)]',     'Heat-loss coefficient in the branches (fan + outlet ports)'
    'T_amb',       '293.15[K]',         'Ambient temperature'
    'rho0',        '972.4[kg/m^3]',     'Density (water ~79 degC)'
    'mu0',         '0.355e-3[Pa*s]',    'Dynamic viscosity'
    'k0',          '0.669[W/(m*K)]',    'Thermal conductivity'
    'Cp0',         '4196[J/(kg*K)]',    'Heat capacity'
    'T_heat_max',  '372.15[K]',         'Upper limit of the heater temperatures'
    'h_heat',      '1e5[W/(m^2*K)]',    'Heater-to-fluid conductance, ~ thin metal wall'
    };
for i = 1:size(defaults, 1)
    model.param.set(defaults{i, 1}, defaults{i, 2}, defaults{i, 3});
end
fn = fieldnames(P);
for i = 1:numel(fn)
    v = P.(fn{i});
    if isnumeric(v), v = num2str(v, 15); end
    model.param.set(fn{i}, v);
end
model.param.set('T_need',  'T_in + dT_target',        'Target outlet temperature');
model.param.set('P_tot',   'rho0*Qv*Cp0*dT_target',   'Nominal heating power');
model.param.set('U_in',    'Qv/(w*depth)',            'Mean inlet velocity');
model.param.set('w_out',   'w/n_br',                  'Final (outlet port) width of every branch: widths sum to w');
model.param.set('Q_ramp',  '1',                       'Flow-rate continuation factor (flow step only)');
end

function g = eval_params(model)
mm = @(n) mphevaluate(model, n, 'mm');
nu = @(n) mphevaluate(model, n);
for n = {'L_port', 'L_main', 'lam', 'A', 'w', 'L_fan', 'L_out', 'L_ext', 'p_out', 'R_bend'}
    g.(n{1}) = mm(n{1});
end
g.f_sine = nu('f_sine');  g.phi = nu('phi');  g.wall_mode = round(nu('wall_mode'));
g.n_br = round(nu('n_br'));
if g.n_br < 1 || g.n_br > 8, error('sine_split:params', 'n_br must be 1..8'); end
g.r = ones(1, g.n_br);                        % symmetric openings: outer 1, inner rho_br
g.r(2:g.n_br-1) = nu('rho_br');
for n = {'heat1_s', 'heat1_l', 'heat2_s', 'heat2_l'}
    g.(n{1}) = nu(n{1});
end
if g.f_sine <= 0 || g.f_sine >= 1
    error('sine_split:params', 'f_sine must be in (0, 1)');
end
if g.heat1_s < 0 || g.heat1_s + g.heat1_l > 1 + 1e-9 || g.heat2_s < 0 || g.heat2_s + g.heat2_l > 1 + 1e-9
    error('sine_split:params', 'Heater windows must lie inside their sections (fractions 0..1)');
end
end

function set_derived(model, G)
% section positions of the current design, as plain numbers (mm)
s = @(n, v, d) model.param.set(n, sprintf('%.15g[mm]', v), d);
s('xa', G.xa, 'Start of the sine section');
s('xb', G.xb, 'End of the sine section');
s('xs', G.xs, 'Split line (end of the straight section)');
s('L',  G.L,  'Exit lines (device length)');
s('L_tot', G.Lt, 'End of the computational domain');
if mphevaluate(model, 'heat1_abs') > 0.5          % fractions of the whole main channel
    model.param.set('xh1a', 'xa + heat1_s*(xs - xa)', 'Heater 1 start');
    model.param.set('xh1b', 'xa + (heat1_s + heat1_l)*(xs - xa)', 'Heater 1 end');
else
    model.param.set('xh1a', 'xa + heat1_s*(xb - xa)', 'Heater 1 start');
    model.param.set('xh1b', 'xa + (heat1_s + heat1_l)*(xb - xa)', 'Heater 1 end');
end
model.param.set('xh2a', 'xb + heat2_s*(xs - xb)', 'Heater 2 start');
model.param.set('xh2b', 'xb + (heat2_s + heat2_l)*(xs - xb)', 'Heater 2 end');
end

function set_walls(geom, G)
% outline: walls + cross lines
tags = add_channel_curves(geom, struct('pieces', {G.pieces}), G.cuts);
cs = geom.create('csol1', 'ConvertToSolid');
cs.selection('input').set(tags);
end

function build_physics(model)
comp = model.component('comp1');
for t = {'sel_in', 'sel_xc', 'sel_split', 'sel_fan', 'sel_meas', 'sel_out'}
    s = comp.selection.create(t{1}, 'Box');
    s.set('entitydim', 1);
    s.set('condition', 'inside');
end
s = comp.selection.create('sel_maincuts', 'Union');   % n_br*n_b elements across
s.set('entitydim', 1);
s.set('input', {'sel_in', 'sel_xc'});
s = comp.selection.create('sel_brcuts', 'Union');     % n_b elements across
s.set('entitydim', 1);
s.set('input', {'sel_split', 'sel_fan', 'sel_meas', 'sel_out'});
s = comp.selection.create('sel_allb', 'Explicit');
s.geom('geom1', 1);
s = comp.selection.create('sel_side', 'Difference');
s.set('entitydim', 1);
s.set('add', {'sel_allb'});
s.set('subtract', {'sel_in', 'sel_xc', 'sel_split', 'sel_fan', 'sel_meas', 'sel_out'});
s = comp.selection.create('sel_alld', 'Explicit');
s.geom('geom1', 2);

op = comp.cpl.create('intop_side', 'Integration');
op.selection.geom('geom1', 1); op.selection.named('sel_side');
for t = {'meas', 'in', 'out'}
    op = comp.cpl.create(['intop_' t{1}], 'Integration');
    op.selection.geom('geom1', 1); op.selection.named(['sel_' t{1}]);
end
op = comp.cpl.create('intop_dom', 'Integration');
op.selection.geom('geom1', 2); op.selection.all;
var = comp.variable.create('var1');
var.set('hw1',     'flc1hs(x - xh1a, dxs)*flc1hs(xh1b - x, dxs)', 'Heater 1 window (0..1)');
var.set('hw2',     'min((xh2b - xh2a)/(2*dxs), 1)*flc1hs(x - xh2a, dxs)*flc1hs(xh2b - x, dxs)', ...
    'Heater 2 window (0..1); fades out for lengths < 2*dxs, 0 at heat2_l = 0 (no heater 2)');
var.set('Tb_L',    'intop_meas(u*T)/intop_meas(u)',            'Mixed bulk temperature at the exit lines');
var.set('th_heat', 'min(max(s_pow, 1), (T_heat_max - T_in)/dT_target)', 'Heater 1 temperature rise / dT_target (>= 1: T_heat >= T_need)');
var.set('T_heat',  'T_in + dT_target*th_heat',                 'Heater 1 temperature');
var.set('T_heat2', 'min(max(T_need + dT_h2, T_need), T_heat_max)', 'Heater 2 temperature (>= T_need)');
var.set('q_wall',  'h_heat*(hw1*(T_heat - T) + hw2*(T_heat2 - T))', 'Heater flux');
var.set('in_dev',  'x <= L',                                   '1 inside the device, 0 in the extension');
var.set('h_l',     'h_loss + (x > xs)*(h_loss_br - h_loss)',  'Loss coefficient (branches: h_loss_br)');
var.set('q_loss',  'in_dev*h_l*(T_amb - T)',                   'Loss flux of a non-heated wall');
var.set('Q_loss',  '2*in_dev*h_l*(T_amb - T)/depth',           'Top + bottom wall losses as a volume source');

mat = comp.material.create('mat1', 'Common');
mat.label('Water (constant properties)');
pg = mat.propertyGroup('def');
pg.set('density', 'rho0');
pg.set('dynamicviscosity', 'mu0');
pg.set('thermalconductivity', {'k0' '0' '0' '0' 'k0' '0' '0' '0' 'k0'});
pg.set('heatcapacity', 'Cp0');
pg.set('ratioofspecificheat', '1');

spf = comp.physics.create('spf', 'LaminarFlow', 'geom1');
spf.prop('PhysicalModelProperty').set('Compressibility', 'Incompressible');
f = spf.create('inl1', 'InletBoundary', 1);
f.selection.named('sel_in');
f.set('BoundaryCondition', 'FullyDevelopedFlow');
f.set('FullyDevelopedFlowOption', 'Uav');
f.set('Uavfdf', 'U_in*Q_ramp');
f = spf.create('out1', 'OutletBoundary', 1);
f.selection.named('sel_out');
f.set('BoundaryCondition', 'Pressure');
f.set('p0', '0');

ht = comp.physics.create('ht', 'HeatTransferInFluids', 'geom1');
ht.feature('init1').set('Tinit', 'T_in');
% no crosswind diffusion (nonlinear shock capturing): the heat problem stays linear
ht.prop('ConsistentStabilization').set('heatCrosswindDiffusion', '0');
f = ht.create('temp1', 'TemperatureBoundary', 1);
f.selection.named('sel_in');
f.set('T0', 'T_in');
f = ht.create('ofl1', 'ConvectiveOutflow', 1);
f.selection.named('sel_out');
f = ht.create('hf1', 'HeatFluxBoundary', 1);
f.selection.named('sel_side');
f.set('HeatFluxType', 'GeneralInwardHeatFlux');
f.set('q0_input', 'q_wall + (1 - hw1 - hw2)*q_loss');
f = ht.create('hs1', 'HeatSource', 2);
f.selection.all;
f.set('Q0', 'Q_loss');

mp = comp.multiphysics.create('nitf1', 'NonIsothermalFlow', 2);
mp.set('Fluid_physics', 'spf');
mp.set('Heat_physics', 'ht');

% heater 1 temperature: solved so that Tb_L = T_need (clamped; see add_heater_control
% in sine_channel.m)
ge = comp.physics.create('ge', 'GlobalEquations');
f = ge.feature('ge1');
f.setIndex('name', 's_pow', 0, 0);
f.setIndex('equation', '(Tb_L - T_need)/dT_target + s_pow - th_heat', 0, 0);
f.setIndex('initialValueU', '1.5', 0, 0);   % away from the lower clamp (th = 1)
f.setIndex('initialValueUt', '0', 0, 0);
f.setIndex('description', 'Heater 1 temperature rise / dT_target (unclamped)', 0, 0);
end

function update_selections(comp, G, g)
tol = 1e-3 * min([g.w, G.w_out]);
h = g.w/2;
yall = G.y_range + [-1 1];
set_box(comp.selection('sel_in'),    [0 0],       G.ya + [-h h], tol);
set_box(comp.selection('sel_xc'),    [G.xc G.xc], G.yb + [-h h], tol);
set_box(comp.selection('sel_split'), [G.xs G.xs], G.yb + [-h h], tol);
set_box(comp.selection('sel_fan'),   [G.xf G.xf], yall, tol);
set_box(comp.selection('sel_meas'),  [G.L G.L],   yall, tol);
set_box(comp.selection('sel_out'),   [G.Lt G.Lt], yall, tol);
comp.selection('sel_allb').all;
comp.selection('sel_alld').all;
end

function ids = check_selections(comp, G)
for t = {'in', 'xc', 'split', 'fan', 'meas', 'out', 'side'}
    ids.(t{1}) = sel_ids(comp, ['sel_' t{1}]);
end
N = G.N;
want = struct('in', 1, 'xc', 1, 'split', N, 'fan', N, 'meas', N, 'out', N);
for t = fieldnames(want)'
    if numel(ids.(t{1})) ~= want.(t{1})
        error('sine_split:selection', 'Selection sel_%s has %d entities, expected %d', ...
            t{1}, numel(ids.(t{1})), want.(t{1}));
    end
end
end

function create_mesh(comp)
% structured quads on all domains; counts and sizes set per design in set_mesh
mesh = comp.mesh.create('mesh1');
m = mesh.create('map1', 'Map');
m.selection.geom('geom1', 2);
m.selection.named('sel_alld');
sz = m.create('size1', 'Size');
sz.set('custom', 'on');
sz.set('hmaxactive', true); sz.set('hminactive', true);
d = m.create('dis1', 'Distribution');          % cross lines of the branches: n_b
d.selection.named('sel_brcuts');
d.set('type', 'predefined');
d.set('symmetric', true);
d = m.create('dis2', 'Distribution');          % cross lines of the main channel: n_br*n_b
d.selection.named('sel_maincuts');
d.set('type', 'predefined');
d.set('symmetric', true);
end

function set_mesh(comp, mo, G)
n_b   = getopt(mo, 'n_b', 6);
ratio = getopt(mo, 'w_ratio', 3);
hmax  = getopt(mo, 'hmax', 0.07);
m = comp.mesh('mesh1').feature('map1');
m.feature('size1').set('hmax', hmax);
m.feature('size1').set('hmin', hmax/10);
m.feature('dis1').set('elemcount', n_b);
m.feature('dis1').set('elemratio', ratio);
m.feature('dis2').set('elemcount', G.N*n_b);
m.feature('dis2').set('elemratio', ratio);
end

function build_study(model, ramp)
std = model.study.create('std1');
st = std.create('stat1', 'Stationary');
if ~isempty(ramp) && ~strcmp(strtrim(ramp), '1')
    st.set('useparam', true);
    st.setIndex('pname', 'Q_ramp', 0);
    st.setIndex('plistarr', ramp, 0);
    st.setIndex('punit', '', 0);
end
st.setEntry('activate', 'ht', false);
st.setEntry('activate', 'ge', false);
st.setEntry('activateCoupling', 'nitf1', false);
st = std.create('stat2', 'Stationary');
st.setEntry('activate', 'spf', false);
std.createAutoSequences('all');
st = model.sol('sol1').feature('s2');
tags = cell(st.feature.tags);
if ~any(strcmp(tags, 'd1'))
    d = st.create('d1', 'Direct');
    d.set('linsolver', 'mumps');
end
if ~any(strcmp(tags, 'fc1')), st.create('fc1', 'FullyCoupled'); end
if any(strcmp(tags, 'se1')), st.feature.remove('se1'); end
st.feature('fc1').set('linsolver', 'd1');
end

%% ======================= post-processing =======================
function [yc, m] = centre_line(G, g, x)
% centre line of the main channel (inlet port, sine, straight section), used to
% place the cross-section cuts of the along-x statistics and the slices
k  = 2*pi/g.lam;
th = k*(x - G.xa) + g.phi;
yc = g.A*sin(th);
m  = g.A*k*cos(th);
yc(x < G.xa) = G.ya;  m(x < G.xa) = 0;
yc(x > G.xb) = G.yb;  m(x > G.xb) = 0;
end

function S = x_profile(model, G, g, nb, nyb, T_need)
% cross-section statistics on nb cuts x = const over the main channel (0 .. xs)
S = struct();
S.x_stat = linspace(0, G.xs, nb + 1); S.x_stat = 0.5*(S.x_stat(1:end-1) + S.x_stat(2:end));
[yc, m] = centre_line(G, g, S.x_stat);
hw = (g.w/2)*sqrt(1 + m.^2);
eta = ((1:nyb) - 0.5)/nyb - 0.5;
[XI, EI] = ndgrid(1:nb, eta);
X = S.x_stat(XI); Y = yc(XI) + 2*EI.*hw(XI);
[T, u] = mphinterp(model, {'T', 'u'}, 'coord', [X(:)'; Y(:)'], 'unit', {'K', 'm/s'});
T = reshape(T, nb, []); u = reshape(u, nb, []);
ok = isfinite(T) & isfinite(u);              % points outside the fluid give NaN
T0 = T; T0(~ok) = 0; u0 = u; u0(~ok) = 0;
n = max(sum(ok, 2), 1);
S.T_bulk_x = sum(u0.*T0, 2) ./ sum(u0, 2);
S.T_mean_x = sum(T0, 2) ./ n;
S.T_std_x  = sqrt(max(sum(T0.^2, 2)./n - S.T_mean_x.^2, 0));
S.T_rms_x  = sqrt(sum(((T0 - T_need).*ok).^2, 2) ./ n);
Tm = T; Tm(~ok) = NaN;
S.T_min_x = min(Tm, [], 2); S.T_max_x = max(Tm, [], 2);
end

function F = export_fields(model, G, g, fo)
nx  = getopt(fo, 'nx', 700);   nyf = getopt(fo, 'ny', 200);
nb  = getopt(fo, 'nb', 200);   ns  = getopt(fo, 'n_slices', 6);
nyb = 60;
F = struct();
F.model_dim = 2;
F.L = G.L; F.L_tot = G.Lt; F.w = g.w;
F.heaters = [mphevaluate(model, 'xh1a', 'mm'), mphevaluate(model, 'xh1b', 'mm');
             mphevaluate(model, 'xh2a', 'mm'), mphevaluate(model, 'xh2b', 'mm')];
F.heat_start = F.heaters(1, 1);  F.heat_end = F.heaters(1, 2);
F.T_need = mphevaluate(model, 'T_need', 'K');
F.T_in   = mphevaluate(model, 'T_in', 'K');
F.walls = cellfun(@(c) struct('x', c(1, :), 'y', c(2, :)), G.lines, 'UniformOutput', false);
F.wall_upper_x = G.W.upper(1, :);  F.wall_upper_y = G.W.upper(2, :);
F.wall_lower_x = G.W.lower(1, :);  F.wall_lower_y = G.W.lower(2, :);
F.map_x = linspace(0, G.Lt, nx);
F.map_y = linspace(G.y_range(1) - 0.05, G.y_range(2) + 0.05, nyf);
[XX, YY] = ndgrid(F.map_x, F.map_y);
[T, gT, U] = mphinterp(model, {'T', 'sqrt(Tx^2 + Ty^2)', 'spf.U'}, ...
    'coord', [XX(:)'; YY(:)'], 'unit', {'K', 'K/mm', 'm/s'});
F.map_T = reshape(T, nx, nyf); F.map_gradT = reshape(gT, nx, nyf); F.map_U = reshape(U, nx, nyf);
S = x_profile(model, G, g, nb, nyb, F.T_need);
for f = fieldnames(S)', F.(f{1}) = S.(f{1}); end
eta = ((1:nyb) - 0.5)/nyb - 0.5;
F.slice_x = G.xs * (1:ns) / ns - 0.05;
[yc, m] = centre_line(G, g, F.slice_x);
hw = (g.w/2)*sqrt(1 + m.^2);
[SI, EI] = ndgrid(1:ns, eta);
X = F.slice_x(SI); Y = yc(SI) + 2*EI.*hw(SI);
[T, U] = mphinterp(model, {'T', 'spf.U'}, 'coord', [X(:)'; Y(:)'], 'unit', {'K', 'm/s'});
F.slice_eta = eta;
F.slice_y = yc(:) + 2*hw(:)*eta;
F.slice_T = reshape(T, ns, nyb);
F.slice_U = reshape(U, ns, nyb);
end

%% ======================= helpers =======================
function v = getopt(s, name, default)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function set_box(s, xr, yr, tol)
s.set('xmin', xr(1) - tol); s.set('xmax', xr(2) + tol);
s.set('ymin', yr(1) - tol); s.set('ymax', yr(2) + tol);
end

function e = sel_ids(comp, tag)
e = double(comp.selection(tag).entities(1));
e = e(:)';
end
