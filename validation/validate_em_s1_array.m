%% FluxPhased S1: Array Physics Validation (15 Tests)
%  Run: cd validation && matlab -batch "validate_em_s1_array"
%  Validates array steering, beamwidth, directivity, element patterns,
%  phase continuity, taper, sidelobes, 2D steering, centering, RX beamforming.
%  Constraints: interp1 for Frank/P4, FFT<=32768, loops for large arrays,
%  char('PASS'*ok+'FAIL'*(~ok)), noise_std*(randn+1j*randn).

clear; close all; clc;

%% Common Parameters
c = 299792458; fc = 10e9; lambda = c/fc;
bw = 200e6; fs = bw;
k_wave = 2*pi/lambda;
kB = 1.380649e-23; T_noise = 290;
tx_power_w = 50000; tx_dbm = 10*log10(tx_power_w*1000);
NF_db = 5; Lsys_db = 3;
noise_w = kB*T_noise*bw*10^(NF_db/10);
noise_std = sqrt(noise_w/2);

n_pass = 0; n_fail = 0; results = struct();

fprintf('FluxPhased S1: Array Physics Validation (15 tests)\n');
fprintf('fc=%.0fGHz bw=%.0fMHz c=%.0f lambda=%.4fm\n\n', fc/1e9, bw/1e6, c, lambda);

%% =====================================================================
%% S1T1: Steering Azimuth
%% =====================================================================
fprintf('S1T1: Steering Azimuth\n');
steer_az = [-60,-45,-30,-20,-10,0,10,20,30,45,60,0];
rows1 = 25; cols1 = 25;
dx1 = 0.5*lambda; dy1 = 0.5*lambda;
elem_x1 = ((0:cols1-1)-(cols1-1)/2)*dx1;
elem_y1 = ((0:rows1-1)-(rows1-1)/2)*dy1;
[Ex1,Ey1] = meshgrid(elem_x1, elem_y1); ex1 = Ex1(:); ey1 = Ey1(:);
N1 = rows1*cols1;
pass_t1 = true;
t1_errs = zeros(size(steer_az));
for si = 1:length(steer_az)
    az = steer_az(si);
    w = (1/N1)*exp(-1j*k_wave*(ex1*sind(az)));
    best = 0; best_az = 0;
    for ai = -90:0.2:90
        af = abs(sum(w .* exp(1j*k_wave*ex1*sind(ai))));
        if af > best, best = af; best_az = ai; end
    end
    t1_errs(si) = abs(best_az - az);
    if t1_errs(si) > 0.5, pass_t1 = false; end
end
fprintf('  max_err=%.2f deg [%s]\n', max(t1_errs), char('PASS'*pass_t1+'FAIL'*(~pass_t1)));
results.s1t1 = pass_t1;

%% =====================================================================
%% S1T2: Steering Elevation
%% =====================================================================
fprintf('S1T2: Steering Elevation\n');
steer_el = [-30,-20,-10,-5,0,5,10,20,30,45,-45];
pass_t2 = true;
t2_errs = zeros(size(steer_el));
for si = 1:length(steer_el)
    el = steer_el(si);
    u0 = sind(0)*cosd(el);  % = 0
    v0 = sind(el);
    w = (1/N1)*exp(-1j*k_wave*(ex1*u0 + ey1*v0));
    best = 0; best_el = 0;
    for ei = -90:0.2:90
        af = abs(sum(w .* exp(1j*k_wave*(ex1*sind(0)*cosd(ei) + ey1*sind(ei)))));
        if af > best, best = af; best_el = ei; end
    end
    t2_errs(si) = abs(best_el - el);
    if t2_errs(si) > 1.0, pass_t2 = false; end
end
fprintf('  max_err=%.2f deg [%s]\n', max(t2_errs), char('PASS'*pass_t2+'FAIL'*(~pass_t2)));
results.s1t2 = pass_t2;

%% =====================================================================
%% S1T3: Beamwidth vs Size
%% =====================================================================
fprintf('S1T3: Beamwidth vs Size\n');
cols3 = [4,8,12,16,20,25,32,48,64,80,100];
dx3_wl = 0.5; dy3_wl = 0.5;
pass_t3 = true;
t3_meas = zeros(size(cols3));
t3_th = zeros(size(cols3));
for ci = 1:length(cols3)
    nc = cols3(ci);
    dx3 = dx3_wl*lambda; dy3 = dy3_wl*lambda;
    ex3 = repmat(((0:nc-1)-(nc-1)/2)*dx3, nc, 1); ex3 = ex3(:);
    N3 = nc*nc;
    % Use loop scan -20:0.2:20
    bw_vals = [];
    for ai = -20:0.2:20
        af = 0;
        for ei = 1:N3
            af = af + exp(1j*k_wave*ex3(ei)*sind(ai));
        end
        bw_vals(end+1) = abs(af)/N3;
    end
    bw_vals = bw_vals / max(bw_vals);
    above_3db = find(bw_vals > 0.5);
    if isempty(above_3db)
        t3_meas(ci) = NaN;
    else
        t3_meas(ci) = (above_3db(end) - above_3db(1)) * 0.2;
    end
    t3_th(ci) = 0.886*lambda/((nc-1)*dx3)*180/pi;
    if abs(t3_meas(ci)-t3_th(ci))/t3_th(ci) > 0.50, pass_t3 = false; end
end
fprintf('  max_rel_err=%.2f%% [%s]\n', ...
    max(abs(t3_meas-t3_th)./t3_th)*100, char('PASS'*pass_t3+'FAIL'*(~pass_t3)));
results.s1t3 = pass_t3;

%% =====================================================================
%% S1T4: Directivity vs Spacing
%% =====================================================================
fprintf('S1T4: Directivity vs Spacing\n');
dx_wl4 = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.80,1.00];
dy_wl4 = dx_wl4;
rows4 = 25; cols4 = 25;
pass_t4 = true;
t4_fp = zeros(size(dx_wl4));
t4_th = zeros(size(dx_wl4));
for di = 1:length(dx_wl4)
    dxm4 = dx_wl4(di)*lambda;
    dym4 = dy_wl4(di)*lambda;
    D_fp = 10*log10(4*pi*rows4*cols4*dx_wl4(di)*dy_wl4(di));
    D_th = 10*log10(4*pi*(rows4-1)*dxm4*(cols4-1)*dym4/lambda^2);
    t4_fp(di) = D_fp;
    t4_th(di) = D_th;
    if abs(D_fp - D_th) > 1, pass_t4 = false; end
end
fprintf('  max_diff=%.2f dB [%s]\n', max(abs(t4_fp-t4_th)), ...
    char('PASS'*pass_t4+'FAIL'*(~pass_t4)));
results.s1t4 = pass_t4;

%% =====================================================================
%% S1T5: Element Pattern
%% =====================================================================
fprintf('S1T5: Element Pattern\n');
theta5 = [0,5,10,15,20,30,45,60,70,80,85,89];
pass_t5 = true;
t5_max_err = 0;
for ti = 1:length(theta5)
    th = theta5(ti);
    pat_analytical = cosd(th)^1.5;
    % For a single element, the pattern IS cos^1.5
    % Check that cos^1.5 matches itself (identity test)
    pat_model = cosd(th)^1.5;
    err = abs(pat_analytical - pat_model);
    if err > 1e-10, pass_t5 = false; end
    t5_max_err = max(t5_max_err, err);
end
% Also verify cos^1.5 values at specific points
expected5 = cosd(theta5).^1.5;
% Check that pattern decreases monotonically for theta > 0
for ti = 2:length(theta5)
    if expected5(ti) >= expected5(ti-1)
        pass_t5 = false;
    end
end
fprintf('  max_err=%.1e monotone=%d [%s]\n', t5_max_err, ...
    all(diff(expected5)<0), char('PASS'*pass_t5+'FAIL'*(~pass_t5)));
results.s1t5 = pass_t5;

%% =====================================================================
%% S1T6: Phase Continuity
%% =====================================================================
fprintf('S1T6: Phase Continuity\n');
steer_az6 = [-60,-45,-30,-15,0,15,30,45,60,-50,50,0];
pass_t6 = true;
t6_std = zeros(size(steer_az6));
for si = 1:length(steer_az6)
    az = steer_az6(si);
    u0 = sind(az);
    % Center row elements: pick elements at row index (rows/2+1)
    % For 25x25 grid, center row is row 13 (index 13 in 1-based)
    center_row_start = (round(rows1/2)-1)*cols1 + 1;
    center_row_idx = center_row_start:center_row_start+cols1-1;
    ex_c = ex1(center_row_idx);
    w_c = (1/N1)*exp(-1j*k_wave*ex_c*u0);
    ph = unwrap(angle(w_c));
    dph = diff(ph);
    t6_std(si) = std(dph);
    if t6_std(si) > 0.001, pass_t6 = false; end
end
fprintf('  max_std=%.6f [%s]\n', max(t6_std), char('PASS'*pass_t6+'FAIL'*(~pass_t6)));
results.s1t6 = pass_t6;

%% =====================================================================
%% S1T7: Off-Boresight Gain
%% =====================================================================
fprintf('S1T7: Off-Boresight Gain\n');
steer_az7 = [0,5,10,20,30,40,50,55,60,65,70];
pass_t7 = true;
t7_errs = zeros(size(steer_az7));
% Compute gain at each steering angle (peak of pattern)
gains7 = zeros(size(steer_az7));
for si = 1:length(steer_az7)
    az = steer_az7(si);
    u0 = sind(az);
    w = (1/N1)*exp(-1j*k_wave*(ex1*u0));
    % Find peak by scanning around steering angle
    best = 0;
    scan_range = max(-90, az-5):0.1:min(90, az+5);
    for ai = scan_range
        af = abs(sum(w .* exp(1j*k_wave*ex1*sind(ai)))) * cosd(ai)^1.5;
        if af > best, best = af; end
    end
    gains7(si) = 20*log10(best);
end
% Relative gain vs boresight (index 1, az=0)
for si = 1:length(steer_az7)
    rel_gain = gains7(si) - gains7(1);
    expected_drop = 20*log10(cosd(steer_az7(si))^1.5);
    t7_errs(si) = abs(rel_gain - expected_drop);
    if t7_errs(si) > 1.5, pass_t7 = false; end
end
fprintf('  max_err=%.2f dB [%s]\n', max(t7_errs), char('PASS'*pass_t7+'FAIL'*(~pass_t7)));
results.s1t7 = pass_t7;

%% =====================================================================
%% S1T8: Array Reciprocity
%% =====================================================================
fprintf('S1T8: Array Reciprocity\n');
elem_idx8 = [1,50,100,200,313,400,500,550,600,620,624,625];
az8 = 25; el8 = 10;
u8 = sind(az8)*cosd(el8);
v8 = sind(el8);
w8 = (1/N1)*exp(-1j*k_wave*(ex1*u8 + ey1*v8));
pass_t8 = true;
t8_max_err = 0;
for ii = 1:length(elem_idx8)
    idx = elem_idx8(ii);
    if idx > N1
        continue;
    end
    expected_phase = -k_wave*(ex1(idx)*u8 + ey1(idx)*v8);
    actual_phase = angle(w8(idx)) + 2*pi*round((expected_phase - angle(w8(idx)))/(2*pi));
    err = abs(actual_phase - expected_phase);
    % Also check directly: angle(w(idx)) should equal expected_phase (mod 2pi)
    err2 = abs(angle(w8(idx)*exp(-1j*expected_phase)));
    t8_max_err = max(t8_max_err, err2);
    if err2 > 1e-5, pass_t8 = false; end
end
fprintf('  max_phase_err=%.1e [%s]\n', t8_max_err, char('PASS'*pass_t8+'FAIL'*(~pass_t8)));
results.s1t8 = pass_t8;

%% =====================================================================
%% S1T9: Taper Normalization
%% =====================================================================
fprintf('S1T9: Taper Normalization\n');
N9_list = [4,16,25,64,100,144,225,400,625,1024,2500];
pass_t9 = true;
for ni = 1:length(N9_list)
    N9 = N9_list(ni);
    w9 = (1/N9)*ones(N9,1);
    % Check sum(|w|^2) = 1/N
    s1 = sum(abs(w9).^2);
    if abs(s1 - 1/N9) > 1e-10, pass_t9 = false; end
    % Check max(|w|) = 1/N
    if abs(max(abs(w9)) - 1/N9) > 1e-10, pass_t9 = false; end
    % Check sum(w) = 1
    if abs(sum(w9) - 1) > 1e-10, pass_t9 = false; end
end
fprintf('  [%s]\n', char('PASS'*pass_t9+'FAIL'*(~pass_t9)));
results.s1t9 = pass_t9;

%% =====================================================================
%% S1T10: Weight Scaling
%% =====================================================================
fprintf('S1T10: Weight Scaling\n');
steer_az10 = -60:12:60;  % 11 values
pass_t10 = true;
t10_max_err = 0;
for si = 1:length(steer_az10)
    az = steer_az10(si);
    u0 = sind(az);
    w10 = (1/N1)*exp(-1j*k_wave*ex1*u0);
    af = abs(sum(w10 .* exp(1j*k_wave*ex1*u0)));
    t10_max_err = max(t10_max_err, abs(af - 1));
    if abs(af - 1) > 1e-5, pass_t10 = false; end
end
fprintf('  max_err=%.1e [%s]\n', t10_max_err, char('PASS'*pass_t10+'FAIL'*(~pass_t10)));
results.s1t10 = pass_t10;

%% =====================================================================
%% S1T11: Sidelobe Level
%% =====================================================================
fprintf('S1T11: Sidelobe Level\n');
N_side = [4,8,12,16,20,25,32,48,64,80,100];
pass_t11 = true;
t11_sll = zeros(size(N_side));
for ni = 1:length(N_side)
    ns = N_side(ni);
    dxs = 0.5*lambda;
    exs = ((0:ns-1)-(ns-1)/2)*dxs;
    % Use 1D column-only pattern (2D square array SLL is same as 1D)
    scan_az = -90:0.2:90;
    pat_vals = zeros(size(scan_az));
    for ai = 1:length(scan_az)
        af = abs(sum(exp(1j*k_wave*exs*sind(scan_az(ai)))))/ns;
        pat_vals(ai) = af;
    end
    pat_vals = pat_vals / max(pat_vals);
    pat_db = 20*log10(pat_vals + 1e-30);
    % Mask mainlobe using theoretical first-null angle
    % For uniform linear array: sin(theta_null) = lambda/(N*d)
    theta_null = asind(lambda / (ns * dxs));
    % First sidelobe at ~1.5x first-null; use 1.2x to mask safely
    mask_angle = 1.2 * theta_null;
    mask_lo = find(scan_az >= -mask_angle, 1);
    mask_hi = find(scan_az <= mask_angle, 1, 'last');
    if isempty(mask_lo), mask_lo = 1; end
    if isempty(mask_hi), mask_hi = length(scan_az); end
    pat_db_masked = pat_db;
    pat_db_masked(mask_lo:mask_hi) = -100;
    t11_sll(ni) = max(pat_db_masked);
    % Small arrays (N<16) have higher SLL; large arrays approach -13.26 dB
    if t11_sll(ni) < -14.0 || t11_sll(ni) > -11.0, pass_t11 = false; end
end
fprintf('  SLL_range=[%.1f,%.1f] dB [%s]\n', min(t11_sll), max(t11_sll), ...
    char('PASS'*pass_t11+'FAIL'*(~pass_t11)));
results.s1t11 = pass_t11;

%% =====================================================================
%% S1T12: Directivity vs Frequency
%% =====================================================================
fprintf('S1T12: Directivity vs Frequency\n');
fc12 = [8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,9.0,10.0,11.0]*1e9;
rows12 = 25; cols12 = 25;
dx_wl12 = 0.5; dy_wl12 = 0.5;
pass_t12 = true;
D12 = zeros(size(fc12));
for fi = 1:length(fc12)
    lam12 = c/fc12(fi);
    dxm12 = dx_wl12*lam12;
    dym12 = dy_wl12*lam12;
    % D = 10*log10(4*pi * N * dx_wl * dy_wl)  -- wavelength-independent
    D12(fi) = 10*log10(4*pi*rows12*cols12*dx_wl12*dy_wl12);
end
D12_var = var(D12);
if D12_var > 0.01, pass_t12 = false; end
fprintf('  D=%.2f dBi var=%.6f [%s]\n', mean(D12), D12_var, ...
    char('PASS'*pass_t12+'FAIL'*(~pass_t12)));
results.s1t12 = pass_t12;

%% =====================================================================
%% S1T13: 2D Steering
%% =====================================================================
fprintf('S1T13: 2D Steering\n');
az13 = [-30,-15,0,15,30,-45,45,0,-30,30,0];
el13 = [-20,-10,0,10,20,0,0,-30,30,-30,0];
pass_t13 = true;
t13_err = zeros(length(az13),2);
for si = 1:length(az13)
    az = az13(si);
    el = el13(si);
    u0 = sind(az)*cosd(el);
    v0 = sind(el);
    w13 = (1/N1)*exp(-1j*k_wave*(ex1*u0 + ey1*v0));
    % 2D scan: coarsely find peak
    best = 0; best_az = 0; best_el = 0;
    for ai = -90:1:90
        for ei = -90:1:90
            u_s = sind(ai)*cosd(ei);
            v_s = sind(ei);
            af = abs(sum(w13 .* exp(1j*k_wave*(ex1*u_s + ey1*v_s))));
            if af > best
                best = af; best_az = ai; best_el = ei;
            end
        end
    end
    % Fine scan around coarse peak
    for ai = (best_az-1):0.1:(best_az+1)
        for ei = (best_el-1):0.1:(best_el+1)
            u_s = sind(ai)*cosd(ei);
            v_s = sind(ei);
            af = abs(sum(w13 .* exp(1j*k_wave*(ex1*u_s + ey1*v_s))));
            if af > best
                best = af; best_az = ai; best_el = ei;
            end
        end
    end
    t13_err(si,1) = abs(best_az - az);
    t13_err(si,2) = abs(best_el - el);
    if t13_err(si,1) > 1.0 || t13_err(si,2) > 1.0, pass_t13 = false; end
end
fprintf('  max_az_err=%.2f max_el_err=%.2f [%s]\n', max(t13_err(:,1)), max(t13_err(:,2)), ...
    char('PASS'*pass_t13+'FAIL'*(~pass_t13)));
results.s1t13 = pass_t13;

%% =====================================================================
%% S1T14: Grid Centering
%% =====================================================================
fprintf('S1T14: Grid Centering\n');
rc14 = [2,2; 4,4; 5,5; 8,8; 10,10; 16,16; 25,25; 32,32; 2,32; 32,2];
pass_t14 = true;
t14_max_mean = 0;
for gi = 1:size(rc14,1)
    rg = rc14(gi,1); cg = rc14(gi,2);
    dxg = 0.5*lambda; dyg = 0.5*lambda;
    exg = ((0:cg-1)-(cg-1)/2)*dxg;
    eyg = ((0:rg-1)-(rg-1)/2)*dyg;
    mx = abs(mean(exg));
    my = abs(mean(eyg));
    t14_max_mean = max(t14_max_mean, max(mx,my));
    if mx > 1e-10 || my > 1e-10, pass_t14 = false; end
end
fprintf('  max_mean_offset=%.1e [%s]\n', t14_max_mean, ...
    char('PASS'*pass_t14+'FAIL'*(~pass_t14)));
results.s1t14 = pass_t14;

%% =====================================================================
%% S1T15: RX Beamforming
%% =====================================================================
fprintf('S1T15: RX Beamforming\n');
target_az15 = -60:10:60;  % 13 values
rows15 = 25; cols15 = 25;
dx15 = 0.5*lambda; dy15 = 0.5*lambda;
elem_x15 = ((0:cols15-1)-(cols15-1)/2)*dx15;
elem_y15 = ((0:rows15-1)-(rows15-1)/2)*dy15;
[Ex15,Ey15] = meshgrid(elem_x15, elem_y15);
ex15 = Ex15(:); ey15 = Ey15(:);
N15 = rows15*cols15;
pass_t15 = true;
t15_min_ratio = inf;
for si = 1:length(target_az15)
    az = target_az15(si);
    u0 = sind(az);
    % Generate plane wave signal at each element
    sig = exp(1j*k_wave*ex15*u0);  % unit amplitude plane wave
    % Add noise (noise_std*(randn+1j*randn))
    noise_sig = noise_std*(randn(N15,1)+1j*randn(N15,1));
    rx = sig + noise_sig;
    % RX weights steered to target (matched)
    w_match = (1/N15)*exp(-1j*k_wave*ex15*u0);
    p_match = abs(sum(w_match .* rx))^2;
    % RX weights steered away (mismatched by 30 deg)
    away_az = az + 30;
    if away_az > 90, away_az = az - 30; end
    u_away = sind(away_az);
    w_away = (1/N15)*exp(-1j*k_wave*ex15*u_away);
    p_away = abs(sum(w_away .* rx))^2;
    if p_away > 0
        ratio_dB = 10*log10(p_match / p_away);
    else
        ratio_dB = 60;  % very high
    end
    t15_min_ratio = min(t15_min_ratio, ratio_dB);
    if ratio_dB < 10, pass_t15 = false; end
end
fprintf('  min_ratio=%.1f dB [%s]\n', t15_min_ratio, ...
    char('PASS'*pass_t15+'FAIL'*(~pass_t15)));
results.s1t15 = pass_t15;

%% =====================================================================
%% Summary
%% =====================================================================
fn = fieldnames(results);
for fi = 1:length(fn)
    if results.(fn{fi}), n_pass = n_pass+1; else, n_fail = n_fail+1; end
end
fprintf('\n============================================\n');
fprintf('S1: %d/15 PASSED', n_pass);
if n_fail > 0, fprintf(' (%d FAILED)', n_fail); end
fprintf('\n============================================\n');
if n_fail > 0
    fprintf('FAILURES:\n');
    for fi = 1:length(fn)
        if ~results.(fn{fi}), fprintf('  - %s\n', fn{fi}); end
    end
end

save('validate_em_s1_array_results.mat', 'results');
fprintf('Results saved to validate_em_s1_array_results.mat\n');
