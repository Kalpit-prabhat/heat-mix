function res = sine_channel(P, opts)
%SINE_CHANNEL  3D channel with a wavy (sine or serpentine) path in the x-y plane
%   and a rectangular w x depth cross-section, heated through its walls.
%
%   res = sine_channel(P, opts)
%
%   Geometry (in the x-y plane, extruded in z), 0 <= x <= L:
%
%       |-- L_port --|-- wavy section --|-- L_port_out --|== L_ext ==|
%        straight in                     straight out      extension
%       <------------------- device, L ---------------------->
%
%       wall_mode 0 (sine): centre line y_c(x) = A*sin(2*pi*(x - L_port)/lam + phi)
%         on the wavy section; walls offset by +/- w/2 along the local normal, so
%         the width normal to the flow is w everywhere.  Needs a bend radius
%         lam^2/(4*pi^2*A) >= w/2, else the walls fold over (error).
%       wall_mode 1 (serpentine): triangle-wave centre line of pitch lam and crest
%         height A whose corners are arcs of radius R_bend (> w/2); same constant
%         width.  See matlab/channel_walls.m.
%       The straight ports start where the wave ends; inlet plane x = 0, exit plane
%       x = L (where the outlet is measured).  The adiabatic extension L_ext moves
%       the outflow boundary condition away from the exit plane.
%       Only the lower half of the depth (0 <= z <= depth/2) is modelled; z = depth/2
%       is a symmetry plane.  (COMSOL reserves 'h' for the mesh size, hence 'depth'.)
%
%   Physics: steady laminar flow (constant properties) + heat transfer in the
%   fluid, solved one after the other (flow first, then heat).  Inlet: fully
%   developed flow of volumetric rate Qv at T_in.
%     heater : over x in [heat_start, heat_start + heat_len] (edges smoothed over
%              +/- dxs) the side and bottom walls (and, by symmetry, the top wall)
%              are held at T_heat: flux h_heat*(T_heat - T), h_heat = 1e5 W/m^2K
%              (thin metal wall).  T_heat is solved (global equation) so that the
%              outlet bulk temperature at x = L equals T_need = T_in + dT_target, and
%              is limited to [T_in, T_heat_max]; at the limit the outlet misses
%              T_need (res.power_clamped = true).
%     losses : every device wall outside the heater window loses
%              q = h_loss*(T_amb - T) to the surroundings; the extension is adiabatic.
%
%   P    : struct of COMSOL parameter expressions (strings with units), e.g.
%          P.lam = '8[mm]'.  Missing fields fall back to the defaults below.
%   opts : struct, all fields optional
%          .mesh       structured mesh: quads in the plane (.n_w across the width,
%                      graded towards both walls by .w_ratio; size .hxy [mm] along
%                      the flow), swept to hexahedra (.nz layers through the
%                      modelled half depth, graded by .z_ratio)
%          .mesh_init  coarser mesh (same fields) on which the flow is solved first
%                      as the initial guess for .mesh
%          .reuse_model keep the COMSOL model alive between calls in the same MATLAB
%                      session and only update parameters, geometry and mesh
%          .flow_ramp_retry  flow-rate continuation (factors of Qv) used to retry a
%                      failed flow solve (default '0.1 0.4 1'; '' = no retry)
%          .ny, .nz    outlet sampling grid (default 40 x 20)
%          .heat_list  rows [heat_start heat_len] in mm: heat-only re-solves on the
%                      same flow and mesh, outlet of each in res.sweep
%          .profile_nb cuts for the light along-x profile stored with every run
%                      (res.profile, default 60; 0 = off)
%          .fields     struct -> also export field data for plots (res.fields):
%                      mid-depth T, |grad T| and |u| maps, cross-section statistics
%                      along x, cross-section slices (.nx, .ny, .nb, .n_slices)
%          .save_mph   file name to save the solved model ('' = do not save)
%          .verbose    print progress (default true)
%
%   res  : struct with the outlet temperature / normal velocity on the grid
%          (T, un: ny x nz), grid coordinates and scalar diagnostics.
%
%   Stand-alone use (after mphstart):  res = sine_channel(struct(), struct());
%   Release a kept model:              sine_channel([], struct('release', true));

import com.comsol.model.*
import com.comsol.model.util.*
persistent cache

if nargin < 1 || isempty(P),    P = struct();    end
if nargin < 2 || isempty(opts), opts = struct(); end
mesh_opts = getopt(opts, 'mesh', struct());
init_opts = getopt(opts, 'mesh_init', struct());
use_init  = isstruct(init_opts) && ~isempty(fieldnames(init_opts));
ny        = getopt(opts, 'ny', 40);
nzs       = getopt(opts, 'nz', 20);
save_mph  = getopt(opts, 'save_mph', '');
verbose   = getopt(opts, 'verbose', true);
reuse     = logical(getopt(opts, 'reuse_model', false));
ramp      = '1';                                   % first try: directly at full flow
ramp_retry = strtrim(getopt(opts, 'flow_ramp_retry', '0.1 0.4 1'));

% drop a cached model that is gone or not wanted any more
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
    [g, hm, has_ext] = eval_params(model);
    skey = jsonencode(struct('mesh', mesh_opts, 'mesh_init', init_opts, 'ext', has_ext));
    if ~isempty(cache) && ~strcmp(cache.key, skey)
        % structural change (mesh settings, extension): start over
        ModelUtil.remove(tag); cache = [];
        [tag, model] = new_model();
        set_params(model, P);
        [g, hm, has_ext] = eval_params(model);
    end
    fresh = isempty(cache);

    if g.heat_end > g.L + 1e-9
        error('sine_channel:params', 'Heated zone ends at %.3g mm, beyond the device length %.3g mm', g.heat_end, g.L);
    end
    W = channel_walls(g);                 % errors if the walls would fold over
    g.ya = W.ya;  g.yb = W.yb;  g.W = W;

    %% ---------------- Build (first call) / rebuild geometry ----------------
    if fresh
        build_geometry(model);
    end
    comp = model.component('comp1');
    geom = comp.geom('geom1');
    set_walls(geom, g, has_ext);
    geom.run;
    n_dom = 1 + has_ext;
    if geom.getNDomains ~= n_dom
        error('sine_channel:geometry', 'Geometry has %d domains, expected %d (self-intersecting walls?)', ...
            geom.getNDomains, n_dom);
    end
    if fresh
        build_physics(model);
        create_mesh(comp, 'mesh1', mesh_opts);
        if use_init, create_mesh(comp, 'mesh0', init_opts); end
    end
    update_selections(comp, g, hm);
    ids = check_selections(comp, has_ext);
    say('geometry ok (path length %.1f mm, bend radius %.3g mm, joint turns %.0f / %.0f deg)%s', ...
        W.path_length, W.R_min, W.turn_in, W.turn_out, ternary(fresh, '', ' [reused model]'));

    t_mesh = tic;
    comp.mesh('mesh1').run;
    [n_elem, q_min] = mesh_size(model, 'mesh1');
    if use_init, comp.mesh('mesh0').run; end
    t_mesh = toc(t_mesh);
    if fresh
        build_study(model, use_init);
    end
    set_ramp(model, use_init, ramp);
    say('mesh: %d elements (min quality %.3f)', n_elem, q_min);

    %% ---------------- Solve ----------------
    sol = model.sol('sol1');
    say('solving flow then heat ...');
    t_solve = tic;
    ramp_used = ramp;
    try
        sol.runAll;
    catch err
        if isempty(ramp_retry) || strcmp(ramp_retry, ramp)
            rethrow(err);
        end
        say('flow solve failed, retrying with flow-rate continuation Q x [%s]', ramp_retry);
        ramp_used = ramp_retry;
        set_ramp(model, use_init, ramp_retry);
        sol.runAll;
    end
    t_solve = toc(t_solve);
    say('solved in %.1f s', t_solve);

    %% ---------------- Outlet data (device exit plane x = L, normal +x) ----------------
    eta = ((1:ny) - 0.5) / ny - 0.5;          % -0.5 .. 0.5 across the width
    zz  = ((1:nzs) - 0.5) / nzs * hm;         % 0 .. hm (mm)
    [EE, ZZ] = ndgrid(eta, zz);
    coords = [g.L * ones(1, numel(EE)); g.yb + EE(:)' * g.w; ZZ(:)'];
    [Tg, ung] = mphinterp(model, {'T', 'u'}, 'coord', coords, ...
        'edim', 'boundary', 'selection', ids.meas, 'unit', {'K', 'm/s'});

    res = struct();
    res.grid_eta  = eta;                   % fraction of width, -0.5..0.5
    res.grid_y_mm = eta * g.w;             % mm across the channel
    res.grid_z_mm = zz;                    % mm, from bottom wall (0) to hm
    res.T  = reshape(Tg,  ny, nzs);        % K
    res.un = reshape(ung, ny, nzs);        % m/s, normal (x) velocity
    res.symmetric = true;                  % half depth, mirror about z = hm

    so = {'surface', 'selection', ids.meas};
    walls = [ids.side, ids.bot];
    Aout = mphint2(model, '1+0*T', so{:});   % '+0*T' forces SI units in post-processing
    Qout = mphint2(model, 'u', so{:});
    res.T_mean_area = mphint2(model, 'T', so{:}) / Aout;
    res.T_bulk      = mphint2(model, 'u*T', so{:}) / Qout;
    res.T_rms_area  = sqrt(mphint2(model, '(T - T_need)^2', so{:}) / Aout);
    res.T_rms_flow  = sqrt(mphint2(model, 'max(u,0)*(T - T_need)^2', so{:}) / mphint2(model, 'max(u,0)', so{:}));
    res.T_max = mphmax(model, 'T', 'surface', 'selection', ids.meas);
    res.T_min = mphmin(model, 'T', 'surface', 'selection', ids.meas);
    res.T_need = mphevaluate(model, 'T_need', 'K');
    res.T_in   = mphevaluate(model, 'T_in', 'K');
    fac = mphevaluate(model, 'depth/hm');
    res.Q_out_ml_min = Qout * fac * 6e7;   % full device
    res.dp_Pa = mphint2(model, 'p', 'surface', 'selection', ids.in) / ...
                mphint2(model, '1+0*T', 'surface', 'selection', ids.in);
    res.P_tot_W  = mphevaluate(model, 'P_tot', 'W');
    res.P_loss_W = -mphint2(model, '(1 - hwx)*q_loss', 'surface', 'selection', walls) * fac;
    % heater power from the energy balance (enthalpy rise at x = L + losses): the
    % near-Dirichlet wall flux h_heat*(T_heat - T) is too ill-conditioned to integrate
    res.P_heater_W = mphint2(model, 'rho0*Cp0*u*(T - T_in)', so{:}) * fac + res.P_loss_W;
    res.P_factor   = res.P_heater_W / res.P_tot_W;
    res.T_heater   = mphglobal(model, 'T_heat');
    th_max = mphevaluate(model, '(T_heat_max - T_in)/dT_target');
    res.power_clamped = clamped(mphglobal(model, 'th_heat'), th_max, res.T_bulk, res.T_need);
    res.A_heat_mm2 = mphint2(model, 'hwx+0*T', 'surface', 'selection', walls) * 1e6 * fac;
    res.q_heater_W_cm2 = res.P_heater_W / max(res.A_heat_mm2, eps) * 100;
    res.T_wall_max = mphmax(model, 'T', 'surface', 'selection', walls);
    res.u_max      = mphmax(model, 'spf.U', 'volume');
    res.Re         = mphevaluate(model, 'rho0*Qv/(w*depth)*2*w*depth/(w+depth)/mu0');
    res.path_length_mm     = W.path_length;
    res.min_gap_mm         = g.w;
    res.min_curv_radius_mm = W.R_min;
    res.joint_turn_deg     = [W.turn_in, W.turn_out];
    res.outlet_y_mm        = g.yb;
    res.mesh_elements = n_elem;
    res.mesh_min_quality = q_min;
    res.flow_ramp     = ramp_used;
    % light temperature profile along the channel for the optimization-progress plots
    npx = getopt(opts, 'profile_nb', 60);
    if npx > 0
        res.profile = x_profile(model, g, hm, npx, 20, 4, res.T_need);
    end
    res.t_mesh  = t_mesh;
    res.t_solve = t_solve;
    res.t_total = toc(t_start);

    fo = getopt(opts, 'fields', struct());
    if isstruct(fo) && isfield(opts, 'fields')
        say('exporting field data ...');
        res.fields = export_fields(model, g, hm, fo);
    end
    if ~isempty(save_mph)
        mphsave(model, save_mph);
    end

    %% ------- heater sweep (heat step only, flow and mesh kept) -------
    % The flow and the mesh do not depend on the heater window (it is a smooth
    % window hwx(x), not a geometric edge), so other windows only need the heat
    % step again.  comsol_opt scores every entry of res.sweep and keeps the best.
    hp_list = getopt(opts, 'heat_list', []);       % rows [heat_start heat_len] in mm
    if ~isempty(hp_list)
        tags = cell(sol.feature.tags);
        st_tags = tags(startsWith(tags, 'st'));
        heat_st = st_tags{end};                       % study step of the heat solve
        hl0 = mphevaluate(model, 'heat_len', 'mm');
        hs0 = mphevaluate(model, 'heat_start', 'mm');
        hp_list = reshape(hp_list, [], 2);
        hl_str  = char(model.param.get('heat_len'));
        hs_str  = char(model.param.get('heat_start'));
        outlet_T = @() reshape(mphinterp(model, 'T', 'coord', coords, 'edim', 'boundary', ...
            'selection', ids.meas, 'unit', 'K'), ny, nzs);
        sw = struct('heat_start', {}, 'heat_len', {}, 'T', {}, 'T_heater', {}, 'T_bulk', {}, 't_heat', {});
        t_sw = tic;
        first_main = true;
        for k = 1:size(hp_list, 1)
            hs = hp_list(k, 1);  hl = hp_list(k, 2);
            if hs < 0 || hl <= 0 || hs + hl > g.L + 1e-9, continue; end
            t1 = tic;
            e = struct('heat_start', hs, 'heat_len', hl, 'T', [], 'T_heater', NaN, 'T_bulk', NaN, 't_heat', 0);
            model.param.set('heat_start', sprintf('%.10g[mm]', hs));
            model.param.set('heat_len', sprintf('%.10g[mm]', hl));
            if first_main && abs(hl - hl0) < 1e-9 && abs(hs - hs0) < 1e-9
                first_main = false;
                e.T = res.T;  e.T_heater = res.T_heater;  e.T_bulk = res.T_bulk;   % the main solve
            else
                sol.runFrom(heat_st);
                e.T = outlet_T();  e.T_heater = mphglobal(model, 'T_heat');
                e.T_bulk = mphglobal(model, 'Tb_L');
            end
            e.t_heat = toc(t1);
            sw(end+1) = e; %#ok<AGROW>
        end
        model.param.set('heat_len', hl_str);
        model.param.set('heat_start', hs_str);
        if size(hp_list, 1) > 1, sol.runFrom(heat_st); end   % back to the main design
        res.sweep = sw;
        res.t_sweep = toc(t_sw);
        say('heater sweep: %d heater windows in %.1f s', numel(sw), res.t_sweep);
    end
    say(['done: outlet RMS(T - T_need) = %.4g K, bulk T rise %.3f K, heater %.3g W (x%.3f), ' ...
         'losses %.3g W, total %.1f s'], res.T_rms_area, res.T_bulk - res.T_in, res.P_heater_W, ...
        res.P_factor, res.P_loss_W, res.t_total);

    if reuse
        cache = struct('tag', tag, 'key', skey);
    else
        ModelUtil.remove(tag);
    end
catch err
    % never keep a model in an unknown state
    if ~isempty(tag)
        try ModelUtil.remove(tag); catch, end
    end
    cache = [];
    rethrow(err);
end

    function say(varargin)
        if verbose
            fprintf('[%s] sine_channel: %s\n', datestr(now, 'HH:MM:SS'), sprintf(varargin{:}));
        end
    end
end

%% ======================= model construction =======================
function [tag, model] = new_model()
tag = sprintf('SineCh_%09d', randi(1e9 - 1));
model = com.comsol.model.util.ModelUtil.create(tag);
model.hist.disable;   % no history recording (faster, less memory)
end

function set_params(model, P)
% defaults: the final-study 3D serpentine (cases/final/u3d_serp.yaml) at lam 8, A 3
defaults = {
    'L',           '29[mm]',            'Device length (x extent, inlet to exit plane)'
    'L_port',      '2[mm]',             'Straight inlet port length'
    'L_port_out',  '3[mm]',             'Straight outlet length before the exit plane'
    'L_ext',       '5[mm]',             'Adiabatic extension after the exit plane, 0 = none'
    'wall_mode',   '1',                 'Walls: 0 sine + normal offset, 1 serpentine (arcs R_bend + legs)'
    'R_bend',      '1.55[mm]',          'Centre-line bend radius of the serpentine (wall_mode 1), > w/2'
    'lam',         '8[mm]',             'Wavelength'
    'A',           '3[mm]',             'Amplitude of the centre line'
    'phi',         '0',                 'Phase of the sine at x = L_port [rad] (wall_mode 0)'
    'w',           '3[mm]',             'Channel width, normal to the path'
    'depth',       '3[mm]',             'Channel depth (z)'
    'heat_start',  '2[mm]',             'Start of heated zone (x)'
    'heat_len',    '24[mm]',            'Length of heated zone (x)'
    'dxs',         '0.1[mm]',           'Half-width of the smoothed heater edge'
    'Qv',          '6[ml/min]',         'Volumetric flow rate'
    'T_in',        '343.15[K]',         'Inlet temperature'
    'dT_target',   '2[K]',              'Required temperature rise'
    'h_loss',      '2[W/(m^2*K)]',      'Heat-loss coefficient of the non-heated walls'
    'T_amb',       '293.15[K]',         'Ambient temperature'
    'rho0',        '972.4[kg/m^3]',     'Density (water ~79 degC)'
    'mu0',         '0.355e-3[Pa*s]',    'Dynamic viscosity'
    'k0',          '0.669[W/(m*K)]',    'Thermal conductivity'
    'Cp0',         '4196[J/(kg*K)]',    'Heat capacity'
    'T_heat_max',  '372.15[K]',         'Upper limit of the heater temperature'
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
model.param.set('k_w',     '2*pi/lam',                'Wave number');
model.param.set('xa',      'L_port',                  'Start of wavy section');
model.param.set('xb',      'L - L_port_out',          'End of wavy section');
model.param.set('L_tot',   'L + L_ext',               'Computational domain length');
model.param.set('hm',      'depth/2',                 'Modelled depth (symmetry plane at z = hm)');
model.param.set('xh1',     'heat_start',              'Heater start');
model.param.set('xh2',     'heat_start + heat_len',   'Heater end');
model.param.set('T_need',  'T_in + dT_target',        'Target outlet temperature');
model.param.set('P_tot',   'rho0*Qv*Cp0*dT_target',   'Nominal heating power');
model.param.set('Q_model', 'Qv*hm/depth',             'Flow rate through the modelled half');
model.param.set('Q_ramp',  '1',                       'Flow-rate continuation factor (flow step only)');
end

function [g, hm, has_ext] = eval_params(model)
% numeric values (mm) for things that need plain numbers
g.L   = mphevaluate(model, 'L', 'mm');
g.Lt  = mphevaluate(model, 'L_tot', 'mm');
g.xa  = mphevaluate(model, 'xa', 'mm');
g.xb  = mphevaluate(model, 'xb', 'mm');
g.lam = mphevaluate(model, 'lam', 'mm');
g.A   = mphevaluate(model, 'A', 'mm');
g.phi = mphevaluate(model, 'phi');
g.w   = mphevaluate(model, 'w', 'mm');
g.wall_mode = round(mphevaluate(model, 'wall_mode'));
g.R_bend = mphevaluate(model, 'R_bend', 'mm');
g.heat_end = mphevaluate(model, 'xh2', 'mm');
hm      = mphevaluate(model, 'hm', 'mm');
has_ext = (g.Lt - g.L) > 1e-9;
end

function build_geometry(model)
% component, work plane and extrusion; the outline itself is (re)made by set_walls
comp = model.component.create('comp1', true);
geom = comp.geom.create('geom1', 3);
geom.lengthUnit('mm');
wp = geom.create('wp1', 'WorkPlane');
wp.set('quickplane', 'xy');
wp.set('unite', true);
ext = geom.create('ext1', 'Extrude');
ext.set('workplane', 'wp1');
ext.selection('input').set({'wp1'});
ext.setIndex('distance', 'hm', 0);
end

function set_walls(geom, g, has_ext)
% channel outline of the current design in the work plane: walls + the
% inlet (x=0), device exit (x=L) and, with an extension, outflow (x=L_tot)
% lines, which lie in the straight ports
h = g.w/2;
cuts = [0, g.ya - h, g.ya + h; g.L, g.yb - h, g.yb + h];
if has_ext, cuts(end+1, :) = [g.Lt, g.yb - h, g.yb + h]; end
wg = geom.feature('wp1').geom;
tags = add_channel_curves(wg, g.W, cuts);
cs = wg.create('csol1', 'ConvertToSolid');
cs.selection('input').set(tags);
end

function build_physics(model)
comp = model.component('comp1');
for t = {'sel_in', 'sel_meas', 'sel_out', 'sel_bot', 'sel_top'}
    s = comp.selection.create(t{1}, 'Box');
    s.set('entitydim', 2);
    s.set('condition', 'inside');
end
for t = {'sel_ecut1', 'sel_ecut2', 'sel_ecut3'}  % cross edges on z = 0 (mapped mesh)
    s = comp.selection.create(t{1}, 'Box');
    s.set('entitydim', 1);
    s.set('condition', 'inside');
end
s = comp.selection.create('sel_ecut', 'Union');
s.set('entitydim', 1);
s.set('input', {'sel_ecut1', 'sel_ecut2', 'sel_ecut3'});
s = comp.selection.create('sel_allb', 'Explicit');
s.geom('geom1', 2);
s = comp.selection.create('sel_side', 'Difference');
s.set('entitydim', 2);
s.set('add', {'sel_allb'});
s.set('subtract', {'sel_in', 'sel_meas', 'sel_out', 'sel_bot', 'sel_top'});

op = comp.cpl.create('intop_meas', 'Integration');
op.selection.geom('geom1', 2); op.selection.named('sel_meas');
var = comp.variable.create('var1');
var.set('hwx',     'flc1hs(x - xh1, dxs)*flc1hs(xh2 - x, dxs)', 'Heater window (0..1)');
var.set('Tb_L',    'intop_meas(u*T)/intop_meas(u)',            'Bulk temperature at the device exit x = L');
var.set('th_heat', 'min(max(s_pow, 0), (T_heat_max - T_in)/dT_target)', 'Heater temperature rise / dT_target');
var.set('T_heat',  'T_in + dT_target*th_heat',                  'Heater wall temperature');
var.set('q_wall',  'hwx*h_heat*(T_heat - T)',                   'Heater flux');
var.set('in_dev',  'x <= L',                                    '1 inside the device, 0 in the extension');
var.set('q_loss',  'in_dev*h_loss*(T_amb - T)',                 'Loss flux of a non-heated wall');

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
f = spf.create('inl1', 'InletBoundary', 2);
f.selection.named('sel_in');
f.set('BoundaryCondition', 'FullyDevelopedFlow');
f.set('FullyDevelopedFlowOption', 'V0');
f.set('V0fdf', 'Q_model*Q_ramp');
f = spf.create('out1', 'OutletBoundary', 2);
f.selection.named('sel_out');
f.set('BoundaryCondition', 'Pressure');
f.set('p0', '0');
f = spf.create('sym1', 'Symmetry', 2);
f.selection.named('sel_top');

ht = comp.physics.create('ht', 'HeatTransferInFluids', 'geom1');
ht.feature('init1').set('Tinit', 'T_in');
% Crosswind diffusion is a nonlinear shock-capturing term that stops Newton
% converging at high Peclet numbers; streamline diffusion (SUPG) stays on, so the
% heat problem is linear.
ht.prop('ConsistentStabilization').set('heatCrosswindDiffusion', '0');
f = ht.create('temp1', 'TemperatureBoundary', 2);
f.selection.named('sel_in');
f.set('T0', 'T_in');
f = ht.create('ofl1', 'ConvectiveOutflow', 2);
f.selection.named('sel_out');
% side walls: heater inside the window, losses outside it
f = ht.create('hf1', 'HeatFluxBoundary', 2);
f.selection.named('sel_side');
f.set('HeatFluxType', 'GeneralInwardHeatFlux');
f.set('q0_input', 'q_wall + (1 - hwx)*q_loss');
% bottom wall (the top wall by symmetry): the same
f = ht.create('hf2', 'HeatFluxBoundary', 2);
f.selection.named('sel_bot');
f.set('HeatFluxType', 'GeneralInwardHeatFlux');
f.set('q0_input', 'q_wall + (1 - hwx)*q_loss');

mp = comp.multiphysics.create('nitf1', 'NonIsothermalFlow', 3);
mp.set('Fluid_physics', 'spf');
mp.set('Heat_physics', 'ht');
add_heater_control(comp);
end

function add_heater_control(comp)
% Global equation for the heater temperature rise s_pow (in units of dT_target),
% solved with the heat step.  Inside [0, (T_heat_max - T_in)/dT_target] it enforces
% Tb_L = T_need.  Outside, th_heat sits at the bound, the first term no longer
% depends on s_pow and the (s_pow - th_heat) term keeps the equation solvable; the
% problem stays piecewise linear, so Newton needs 1-3 iterations.
ge = comp.physics.create('ge', 'GlobalEquations');
f = ge.feature('ge1');
f.setIndex('name', 's_pow', 0, 0);
f.setIndex('equation', '(Tb_L - T_need)/dT_target + s_pow - th_heat', 0, 0);
f.setIndex('initialValueU', '1', 0, 0);
f.setIndex('initialValueUt', '0', 0, 0);
f.setIndex('description', 'Heater temperature rise / dT_target (unclamped)', 0, 0);
end

function update_selections(comp, g, hm)
tol = 1e-3 * min([g.w, hm]);
big = 10 * (g.Lt + g.A + g.w);
set_box(comp.selection('sel_in'),   [0 0],       g.ya + [-g.w/2 g.w/2], [0 hm], tol);
set_box(comp.selection('sel_meas'), [g.L g.L],   g.yb + [-g.w/2 g.w/2], [0 hm], tol);
set_box(comp.selection('sel_out'),  [g.Lt g.Lt], g.yb + [-g.w/2 g.w/2], [0 hm], tol);
set_box(comp.selection('sel_bot'),  [-big big],  [-big big],     [0 0],   tol);
set_box(comp.selection('sel_top'),  [-big big],  [-big big],     [hm hm], tol);
set_box(comp.selection('sel_ecut1'), [0 0],       g.ya + [-g.w/2 g.w/2], [0 0], tol);
set_box(comp.selection('sel_ecut2'), [g.L g.L],   g.yb + [-g.w/2 g.w/2], [0 0], tol);
set_box(comp.selection('sel_ecut3'), [g.Lt g.Lt], g.yb + [-g.w/2 g.w/2], [0 0], tol);
comp.selection('sel_allb').all;          % re-take all boundaries of the new geometry
end

function ids = check_selections(comp, has_ext)
ids.in   = sel_ids(comp, 'sel_in');
ids.meas = sel_ids(comp, 'sel_meas');
ids.out  = sel_ids(comp, 'sel_out');
ids.bot  = sel_ids(comp, 'sel_bot');
ids.top  = sel_ids(comp, 'sel_top');
ids.side = sel_ids(comp, 'sel_side');
nd = 1 + has_ext;
if numel(ids.in) ~= 1 || numel(ids.meas) ~= 1 || numel(ids.out) ~= 1 || numel(ids.bot) ~= nd || ...
        numel(ids.top) ~= nd || numel(ids.side) < 2*nd
    error('sine_channel:selection', 'Unexpected boundary selection (in=%d meas=%d out=%d bot=%d top=%d side=%d)', ...
        numel(ids.in), numel(ids.meas), numel(ids.out), numel(ids.bot), numel(ids.top), numel(ids.side));
end
end

function create_mesh(comp, tag, mo)
% structured quads on the bottom face, swept up to the symmetry plane
mesh = comp.mesh.create(tag);
hxy = getopt(mo, 'hxy', 0.10);         % element size along the flow [mm]
m = mesh.create('map1', 'Map');
m.selection.named('sel_bot');
sz = m.create('size1', 'Size');
sz.set('custom', 'on');
sz.set('hmax', hxy);    sz.set('hmaxactive', true);
sz.set('hmin', hxy/10); sz.set('hminactive', true);
d = m.create('dis1', 'Distribution');  % across the width, finer at both walls
d.selection.named('sel_ecut');
d.set('type', 'predefined');
d.set('elemcount', getopt(mo, 'n_w', 12));
d.set('elemratio', getopt(mo, 'w_ratio', 3));
d.set('symmetric', true);
m = mesh.create('swe1', 'Sweep');
d = m.create('dis1', 'Distribution');  % through the half depth, finer at the bottom wall
d.set('type', 'predefined');
d.set('elemcount', getopt(mo, 'nz', 6));
d.set('elemratio', getopt(mo, 'z_ratio', 3));
end

function [n_elem, q_min] = mesh_size(model, tag)
st = mphmeshstats(model, tag);
types = cellstr(st.types);
v = ismember(types, {'tet', 'pyr', 'prism', 'hex'});
n_elem = sum(double(st.numelem(v)));
q_min = min(double(st.minquality(:)));     % over all element types
end

function build_study(model, use_init)
% flow first (on mesh0 then mesh1 if use_init), then heat on mesh1
std = model.study.create('std1');
if use_init
    st = std.create('stat0', 'Stationary');
    set_param_sweep(st);
    st.setEntry('activate', 'ht', false);
    st.setEntry('activate', 'ge', false);
    st.setEntry('activateCoupling', 'nitf1', false);
    st.set('mesh', {'geom1', 'mesh0'});
end
st = std.create('stat1', 'Stationary');
if ~use_init, set_param_sweep(st); end
st.setEntry('activate', 'ht', false);
st.setEntry('activate', 'ge', false);
st.setEntry('activateCoupling', 'nitf1', false);
st.set('mesh', {'geom1', 'mesh1'});
st = std.create('stat2', 'Stationary');
st.setEntry('activate', 'spf', false);
st.set('mesh', {'geom1', 'mesh1'});
std.createAutoSequences('all');
sol = model.sol('sol1');
if use_init
    heat_step = 's3';
    v = sol.feature('v2');               % mesh1 flow starts from the mesh0 flow
    v.set('initmethod', 'sol');
    v.set('initsol', 'sol1');
else
    heat_step = 's2';
end
% flow steps: COMSOL's default solvers.  Heat step: MUMPS, fully coupled (T and
% s_pow solved together, not segregated).
st = sol.feature(heat_step);
tags = cell(st.feature.tags);
if ~any(strcmp(tags, 'd1'))
    st.create('d1', 'Direct');
end
st.feature('d1').set('linsolver', 'mumps');
if ~any(strcmp(tags, 'fc1'))
    st.create('fc1', 'FullyCoupled');
end
if any(strcmp(tags, 'se1'))
    st.feature.remove('se1');
end
st.feature('fc1').set('linsolver', 'd1');
end

function set_param_sweep(st)
% the first flow step carries a sweep over Q_ramp (normally just '1'),
% so continuation can be switched on for a retry without rebuilding the study
st.set('useparam', true);
st.setIndex('pname', 'Q_ramp', 0);
st.setIndex('plistarr', '1', 0);
st.setIndex('punit', '', 0);
end

function set_ramp(model, use_init, list)
% flow-rate continuation list of the first flow step (study step and its solver)
if isempty(list), list = '1'; end
model.study('std1').feature(ternary(use_init, 'stat0', 'stat1')).setIndex('plistarr', list, 0);
s1 = model.sol('sol1').feature('s1');
tags = cell(s1.feature.tags);
for i = 1:numel(tags)
    f = s1.feature(tags{i});
    if strcmp(char(f.getType), 'Parametric')
        f.set('plistarr', list);
    end
end
end

%% ======================= post-processing =======================
function [yc, m] = centre_line(g, x)
% numeric centre line y_c(x) and slope (mm): ports at ya / yb, wave in between
if g.wall_mode == 1                      % serpentine: sampled by channel_walls
    C = g.W.cline;
    yc = interp1(C(1,:), C(2,:), x, 'linear', 'extrap');
    s  = gradient(C(2,:)) ./ gradient(C(1,:));
    m  = interp1(C(1,:), s, x, 'linear', 0);
    return
end
k  = 2*pi/g.lam;
th = k*(x - g.xa) + g.phi;
yc = g.A*sin(th);
m  = g.A*k*cos(th);
yc(x < g.xa) = g.ya;  m(x < g.xa) = 0;
yc(x > g.xb) = g.yb;  m(x > g.xb) = 0;
end

function F = export_fields(model, g, hm, fo)
% Field samples for plotting (all lengths in mm, NaN outside the fluid).
nx  = getopt(fo, 'nx', 600);   nyf = getopt(fo, 'ny', 120);
nb  = getopt(fo, 'nb', 200);   ns  = getopt(fo, 'n_slices', 6);
nyb = 40; nzb = 8;             % cross-section sampling (statistics, slices)
F = struct();
F.model_dim = 3;
F.L = g.L; F.L_tot = g.Lt; F.w = g.w; F.depth_modelled = hm; F.symmetric = true;
F.heat_start = mphevaluate(model, 'xh1', 'mm');
F.heat_end   = mphevaluate(model, 'xh2', 'mm');
F.T_need = mphevaluate(model, 'T_need', 'K');
F.T_in   = mphevaluate(model, 'T_in', 'K');
% wall outlines
F.wall_upper_x = g.W.upper(1, :);  F.wall_upper_y = g.W.upper(2, :);
F.wall_lower_x = g.W.lower(1, :);  F.wall_lower_y = g.W.lower(2, :);
% 1) mid-depth map (just below the symmetry plane z = hm)
zmid = hm*(1 - 1e-4);
ylim = max(abs([F.wall_upper_y, F.wall_lower_y])) * 1.02;
F.map_x = linspace(0, g.Lt, nx);
F.map_y = linspace(-ylim, ylim, nyf);
F.map_z = zmid;
[XX, YY] = ndgrid(F.map_x, F.map_y);
[T, gT, U] = mphinterp(model, {'T', 'sqrt(Tx^2 + Ty^2 + Tz^2)', 'spf.U'}, ...
    'coord', [XX(:)'; YY(:)'; zmid*ones(1, numel(XX))], 'unit', {'K', 'K/mm', 'm/s'});
F.map_T = reshape(T, nx, nyf); F.map_gradT = reshape(gT, nx, nyf); F.map_U = reshape(U, nx, nyf);
% 2) cross-section statistics along x (vertical cuts x = const, cell centres)
S = x_profile(model, g, hm, nb, nyb, nzb, F.T_need);
for f = fieldnames(S)', F.(f{1}) = S.(f{1}); end
% 3) cross-section slices at evenly spaced stations up to the exit plane
F.slice_x = g.L * (1:ns) / ns;
[yc, m] = centre_line(g, F.slice_x);
hw = (g.w/2)*sqrt(1 + m.^2);          % half height of the vertical cut through the band
eta = ((1:nyb) - 0.5)/nyb - 0.5;  zz = ((1:nzb*2) - 0.5)/(nzb*2)*hm;
[SI, EI, ZI] = ndgrid(1:ns, eta, zz);
X = F.slice_x(SI); Y = yc(SI) + 2*EI.*hw(SI);
[T, U] = mphinterp(model, {'T', 'spf.U'}, 'coord', [X(:)'; Y(:)'; ZI(:)'], 'unit', {'K', 'm/s'});
F.slice_eta = eta; F.slice_z = zz;
F.slice_y = yc(:) + 2*hw(:)*eta;                    % ns x nyb, actual y [mm]
F.slice_T = reshape(T, ns, nyb, numel(zz));
F.slice_U = reshape(U, ns, nyb, numel(zz));
end

function S = x_profile(model, g, hm, nb, nyb, nzb, T_need)
% cross-section statistics on nb vertical cuts x = const (cell centres over 0..L_tot)
S = struct();
S.x_stat = linspace(0, g.Lt, nb + 1); S.x_stat = 0.5*(S.x_stat(1:end-1) + S.x_stat(2:end));
[yc, m] = centre_line(g, S.x_stat);
hw = (g.w/2)*sqrt(1 + m.^2);         % half height of the vertical cut through the band
eta = ((1:nyb) - 0.5)/nyb - 0.5;  zz = ((1:nzb) - 0.5)/nzb*hm;
[XI, EI, ZI] = ndgrid(1:nb, eta, zz);
X = S.x_stat(XI); Y = yc(XI) + 2*EI.*hw(XI);
[T, u] = mphinterp(model, {'T', 'u'}, 'coord', [X(:)'; Y(:)'; ZI(:)'], 'unit', {'K', 'm/s'});
S = profile_stats(S, reshape(T, nb, []), reshape(u, nb, []), T_need);
end

function S = profile_stats(S, T, u, T_need)
ok = isfinite(T) & isfinite(u);
T0 = T; T0(~ok) = 0; u0 = u; u0(~ok) = 0;
n = max(sum(ok, 2), 1);
S.T_bulk_x = sum(u0.*T0, 2) ./ sum(u0, 2);
S.T_mean_x = sum(T0, 2) ./ n;
S.T_std_x  = sqrt(max(sum(T0.^2, 2)./n - S.T_mean_x.^2, 0));
S.T_rms_x  = sqrt(sum(((T0 - T_need).*ok).^2, 2) ./ n);
Tm = T; Tm(~ok) = NaN;
S.T_min_x = min(Tm, [], 2); S.T_max_x = max(Tm, [], 2);
end

%% ======================= helpers =======================
function c = clamped(th, th_max, T_bulk, T_need)
% heater temperature at a limit and the outlet missing its target
c = (th <= 1e-6 || th >= th_max - 1e-6) && abs(T_bulk - T_need) > 1e-3;
end

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

function set_box(s, xr, yr, zr, tol)
s.set('xmin', xr(1) - tol); s.set('xmax', xr(2) + tol);
s.set('ymin', yr(1) - tol); s.set('ymax', yr(2) + tol);
s.set('zmin', zr(1) - tol); s.set('zmax', zr(2) + tol);
end

function e = sel_ids(comp, tag)
e = double(comp.selection(tag).entities(2));
e = e(:)';
end
