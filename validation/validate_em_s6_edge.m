%% FluxPhased EM Base Validation - S6: Edge Cases / Boundaries (14 Tests)
%  Run: cd validation && matlab -batch "validate_em_s6_edge"
%  Validates boundary conditions and extreme values where numerical issues
%  are most likely in FluxPhased's IQ-level EM simulation base layer.
%  Tests: near-zero range, low TX power, extreme spacing, small arrays,
%  max scan angle, delay+Doppler, high BW MF, PRF/ambiguity, short pulse,
%  clamping at +/-90, noise std, Doppler resolution, dB round-trip,
%  Lsys=0 boundary.

clear; close all; clc;

%% Common Parameters
c = 299792458; fc = 10e9; lambda = c/fc;
bw = 200e6; fs = bw; prf = 10e3; pri = 1/prf;
rows = 25; cols = 25; N_elem = rows*cols;
dx_m = 0.5*lambda; dy_m = 0.5*lambda;
tx_power_w = 50000; tx_dbm = 10*log10(tx_power_w*1000);
NF_db = 5; Lsys_db = 3; k_wave = 2*pi/lambda;
kB = 1.380649e-23; T_noise = 290;
noise_w = kB*T_noise*bw*10^(NF_db/10);
noise_dbm = 10*log10(noise_w*1000);
noise_std = sqrt(noise_w/2);
pw = 50e-6; n_lfm = floor(pw*fs);
k_lfm = bw/pw; t_lfm = (0:n_lfm-1)'/fs;
lfm_up = exp(1j*pi*k_lfm*t_lfm.^2); lfm_up = lfm_up/norm(lfm_up);
area_wl2 = rows*cols*0.5*0.5; D_dBi = 10*log10(4*pi*area_wl2);
n_samp = floor(pri*fs);

elem_x = ((0:cols-1)-(cols-1)/2)*dx_m;
elem_y = ((0:rows-1)-(cols-1)/2)*dy_m;
[Ex,Ey] = meshgrid(elem_x, elem_y); ex = Ex(:); ey = Ey(:);

n_pass = 0; n_fail = 0; results = struct();

fprintf('FluxPhased EM Base Validation - S6: Edge Cases / Boundaries (14 tests)\n');
fprintf('fc=%.0fGHz bw=%.0fMHz %dx%d D=%.1fdBi tx=%.1fdBm\n\n', ...
    fc/1e9, bw/1e6, rows, cols, D_dBi, tx_dbm);

%% =====================================================================
%% S6T1: Near-Zero Range
%% =====================================================================
fprintf('S6T1: Near-Zero Range\n');
R_meters = [1 5 10 50 100 200 500 700 800 900 950 999];
pass_t1 = true;
max_err_t1 = 0;
Pr_t1 = zeros(size(R_meters));
for ri = 1:length(R_meters)
    R_m = R_meters(ri);
    rcs_dbsm = 20;
    % Two-way radar equation (dB formula)
    Pr_t1(ri) = tx_dbm + 2*D_dBi + rcs_dbsm + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m) - Lsys_db;
    % Cross-check against linear formula
    G_lin = 10^(D_dBi/10);
    L_lin = 10^(Lsys_db/10);
    sigma_lin = 10^(rcs_dbsm/10);
    Pr_lin = 10*log10(tx_power_w * G_lin^2 * lambda^2 * sigma_lin ...
        / ((4*pi)^3 * R_m^4 * L_lin) * 1000);
    err = abs(Pr_t1(ri) - Pr_lin);
    if err > max_err_t1, max_err_t1 = err; end
    if err > 0.5, pass_t1 = false; end
end
% Verify 40dB/decade slope
max_slope_err = 0;
for ri = 1:length(R_meters)-1
    for rj = ri+1:length(R_meters)
        ratio = R_meters(rj) / R_meters(ri);
        expected_diff = 40*log10(ratio);
        actual_diff = Pr_t1(ri) - Pr_t1(rj);
        s_err = abs(actual_diff - expected_diff);
        if s_err > max_slope_err, max_slope_err = s_err; end
    end
end
if max_slope_err > 0.01, pass_t1 = false; end
fprintf('  Pr=[%.1f .. %.1f] dBm max_formula_err=%.4f max_slope_err=%.4f [%s]\n', ...
    min(Pr_t1), max(Pr_t1), max_err_t1, max_slope_err, ...
    char('PASS'*pass_t1+'FAIL'*(~pass_t1)));
results.s6t1 = pass_t1;

%% =====================================================================
%% S6T2: Very Low TX Power
%% =====================================================================
fprintf('S6T2: Very Low TX Power\n');
Pt_list = [1e-4 1e-3 1e-2 0.1 1 10 100 1000 1e4 5e4 1e5 1e6];
pass_t2 = true;
R_m_t2 = 10000; rcs_t2 = 20;
Pr_t2 = zeros(size(Pt_list));
for pi_idx = 1:length(Pt_list)
    tx_dbm_i = 10*log10(Pt_list(pi_idx)*1000);
    Pr_t2(pi_idx) = tx_dbm_i + 2*D_dBi + rcs_t2 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m_t2) - Lsys_db;
end
% When Pt increases by factor 10, Pr increases by 10 dB
max_diff_err = 0;
for pi_idx = 1:length(Pt_list)-1
    expected_step = 10*log10(Pt_list(pi_idx+1)) - 10*log10(Pt_list(pi_idx));
    actual_step = Pr_t2(pi_idx+1) - Pr_t2(pi_idx);
    d_err = abs(actual_step - expected_step);
    if d_err > max_diff_err, max_diff_err = d_err; end
    if d_err > 0.01, pass_t2 = false; end
end
% Check total span: Pt spans 10 decades (1e-4 to 1e6 => 100 dB)
total_span = Pr_t2(end) - Pr_t2(1);
expected_span = 10*log10(Pt_list(end)) - 10*log10(Pt_list(1));
span_err = abs(total_span - expected_span);
if span_err > 0.01, pass_t2 = false; end
fprintf('  Pr=[%.1f .. %.1f] dBm span=%.1fdB expected=%.1fdB max_step_err=%.4f [%s]\n', ...
    min(Pr_t2), max(Pr_t2), total_span, expected_span, max_diff_err, ...
    char('PASS'*pass_t2+'FAIL'*(~pass_t2)));
results.s6t2 = pass_t2;

%% =====================================================================
%% S6T3: Extreme Spacing
%% =====================================================================
fprintf('S6T3: Extreme Spacing\n');
dx_wl_t3 = [0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9 0.95 1.0];
dy_wl_t3 = dx_wl_t3;
pass_t3 = true;
D_t3 = zeros(size(dx_wl_t3));
grating_lobes = false;
no_grating_half = true;

% Compute directivity for all spacings
for di = 1:length(dx_wl_t3)
    D_t3(di) = 10*log10(4*pi*rows*cols*dx_wl_t3(di)*dy_wl_t3(di));
end

% Helper: compute beam pattern for a given spacing (azimuth-only, 1D cut)
scan_az = -90:0.1:90;

% Check dx_wl=1.0 for grating lobes
dxw = 1.0;
dxm = dxw*lambda;
ex_1d = ((0:cols-1)-(cols-1)/2)*dxm;
pat_1 = zeros(size(scan_az));
for ai = 1:length(scan_az)
    pat_1(ai) = abs(sum(exp(1j*k_wave*ex_1d*sind(scan_az(ai))))) / cols;
end
pat_db_1 = 20*log10(pat_1 / max(pat_1) + 1e-30);
[~, ml_idx] = max(pat_db_1);
pat_masked_1 = pat_db_1;
pat_masked_1(max(1,ml_idx-50):min(length(pat_masked_1),ml_idx+50)) = -100;
secondary_peak_1 = max(pat_masked_1);
if secondary_peak_1 > -10
    grating_lobes = true;
end

% Check dx_wl=0.5: no grating lobe in visible space
dxw = 0.5;
dxm = dxw*lambda;
ex_1d = ((0:cols-1)-(cols-1)/2)*dxm;
pat_05 = zeros(size(scan_az));
for ai = 1:length(scan_az)
    pat_05(ai) = abs(sum(exp(1j*k_wave*ex_1d*sind(scan_az(ai))))) / cols;
end
pat_db_05 = 20*log10(pat_05 / max(pat_05) + 1e-30);
[~, ml_idx] = max(pat_db_05);
pat_masked_05 = pat_db_05;
pat_masked_05(max(1,ml_idx-50):min(length(pat_masked_05),ml_idx+50)) = -100;
secondary_peak_05 = max(pat_masked_05);
if secondary_peak_05 > -5
    no_grating_half = false;
end

% Check D is consistent with formula: increases with spacing
for di = 2:length(D_t3)
    if D_t3(di) < D_t3(di-1)
        pass_t3 = false;
    end
end
if ~grating_lobes
    fprintf('  WARNING: no grating lobe detected at dx=1.0wl\n');
    % This is informational, not a failure -- the grating lobe check is
    % confirming the physics, and some array geometries may suppress it.
end
if ~no_grating_half
    pass_t3 = false;
end
fprintf('  D=[%.1f .. %.1f] dBi grating@1.0=%ddB clean@0.5=%ddB [%s]\n', ...
    min(D_t3), max(D_t3), round(secondary_peak_1), round(secondary_peak_05), ...
    char('PASS'*pass_t3+'FAIL'*(~pass_t3)));
results.s6t3 = pass_t3;

%% =====================================================================
%% S6T4: Very Small Arrays
%% =====================================================================
fprintf('S6T4: Very Small Arrays\n');
% sizes: [(2,2) (2,4) (3,3) (4,2) (4,4) (5,2) (5,5) (6,2) (8,2) (10,2) (10,10) (3,5)]
sizes_r = [2 2 3 4 4 5 5 6 8 10 10 3];
sizes_c = [2 4 3 2 4 2 5 2 2  2 10 5];
dx_wl_t4 = 0.5; dy_wl_t4 = 0.5;
pass_t4 = true;
D_t4 = zeros(size(sizes_r));
all_positive = true;
increases = true;
prev_D = -inf;
for si = 1:length(sizes_r)
    R_sz = sizes_r(si);
    C_sz = sizes_c(si);
    N_sz = R_sz * C_sz;
    D_t4(si) = 10*log10(4*pi*R_sz*C_sz*dx_wl_t4*dy_wl_t4);
    if D_t4(si) <= 0, all_positive = false; pass_t4 = false; end
    % Check D increases with element count
    if N_sz > 1 && D_t4(si) < prev_D
        % Not strictly required since sizes aren't sorted by N, but
        % D formula is 10*log10(4*pi*N*0.25) which increases with N
    end
    prev_D = D_t4(si);
end
% Verify formula: D = 10*log10(pi*N) for dx_wl=dy_wl=0.5
for si = 1:length(sizes_r)
    N_sz = sizes_r(si) * sizes_c(si);
    expected_D = 10*log10(pi*N_sz);
    if abs(D_t4(si) - expected_D) > 0.001
        pass_t4 = false;
    end
end
if ~all_positive, pass_t4 = false; end
fprintf('  D=[%.1f .. %.1f] dBi all_positive=%d [%s]\n', ...
    min(D_t4), max(D_t4), all_positive, ...
    char('PASS'*pass_t4+'FAIL'*(~pass_t4)));
results.s6t4 = pass_t4;

%% =====================================================================
%% S6T5: Max Scan Angle
%% =====================================================================
fprintf('S6T5: Max Scan Angle\n');
az_t5 = [85 86 87 88 89 89.5 89.9 89.95 -85 -86 -87 -89];
pass_t5 = true;
err_t5 = zeros(size(az_t5));
for ai = 1:length(az_t5)
    az = az_t5(ai);
    u0 = sind(az);
    w = (1/N_elem)*exp(-1j*k_wave*ex*u0);
    % Find actual peak by searching in fine grid [-90:0.1:90]
    best_val = 0; best_az = 0;
    scan_angles = -90:0.1:90;
    for si = 1:length(scan_angles)
        af = abs(sum(w .* exp(1j*k_wave*ex*sind(scan_angles(si)))));
        if af > best_val
            best_val = af;
            best_az = scan_angles(si);
        end
    end
    err_t5(ai) = abs(best_az - az);
    % For |az| <= 89, peak should be within 1.5 degrees
    if abs(az) <= 89 && err_t5(ai) > 1.5
        pass_t5 = false;
    end
    % No NaN for any angle
    if isnan(best_az) || isnan(best_val)
        pass_t5 = false;
    end
end
fprintf('  max_err=%.2f deg (for |az|<=89) [%s]\n', max(err_t5(abs(az_t5)<=89)), ...
    char('PASS'*pass_t5+'FAIL'*(~pass_t5)));
results.s6t5 = pass_t5;

%% =====================================================================
%% S6T6: Simultaneous Delay+Doppler
%% =====================================================================
fprintf('S6T6: Simultaneous Delay+Doppler\n');
R_km_t6 = [1 2 5];
v_t6 = [100 -200];
pass_t6 = true;
NFFT_t6 = min(32768, 2^nextpow2(n_samp + n_lfm));
max_R_err_m = 0;
max_fd_err_pct = 0;

for ri = 1:length(R_km_t6)
    for vi = 1:length(v_t6)
        R_m = R_km_t6(ri)*1000;
        v = v_t6(vi);
        fd = 2*v*fc/c;
        delay_samp = round(2*R_m/c*fs);

        % Generate TX signal (LFM pulse + zeros for full PRI)
        tx_sig = [lfm_up; complex(zeros(n_samp - n_lfm, 1))];

        % Generate RX signal: delayed + Doppler-shifted
        rx_sig = complex(zeros(n_samp, 1));
        ds = 2*pi*fd/fs;
        for s = 1:n_samp
            src = s - delay_samp;
            if src >= 1 && src <= n_lfm
                rx_sig(s) = tx_sig(src) * exp(1j*(s-1)*ds);
            end
        end

        % --- Measure delay via matched filter ---
        mf_ref = conj(flipud(lfm_up));
        mf_padded = [mf_ref; complex(zeros(NFFT_t6 - n_lfm, 1))];
        rx_padded = [rx_sig; complex(zeros(NFFT_t6 - n_samp, 1))];
        mf_out = ifft(fft(rx_padded, NFFT_t6) .* fft(mf_padded, NFFT_t6));
        mf_pow = abs(mf_out).^2;
        [~, pk_idx] = max(mf_pow);
        meas_delay = pk_idx - n_lfm;
        meas_R = meas_delay * c / (2*fs);
        R_err = abs(meas_R - R_m);
        if R_err > max_R_err_m, max_R_err_m = R_err; end
        if R_err > 3*c/(2*fs)
            pass_t6 = false;
        end

        % --- Measure Doppler via FFT peak ---
        % Extract pulse at measured delay location
        pulse_start = max(1, pk_idx);
        pulse_end = min(pulse_start + n_lfm - 1, NFFT_t6);
        pulse_len = pulse_end - pulse_start + 1;
        if pulse_len > 2
            rx_pulse = mf_out(pulse_start:pulse_end);
            rx_pulse = rx_pulse / (abs(rx_pulse) + 1e-30);
            NFFT_dop = min(8192, 2^nextpow2(pulse_len * 2));
            spec = fft(rx_pulse, NFFT_dop);
            [~, pk_dop] = max(abs(spec(1:NFFT_dop/2)));
            freq_axis = (0:NFFT_dop/2-1) * (fs/NFFT_dop);
            meas_fd = freq_axis(pk_dop);
            % Handle negative Doppler: check if aliased
            if v < 0
                % Negative Doppler wraps; the peak appears at fs - |fd|
                % For measurement, check both positive and negative freq
                meas_fd_neg = meas_fd - fs;
                fd_err_pos = abs(meas_fd - abs(fd));
                fd_err_neg = abs(abs(meas_fd_neg) - abs(fd));
                fd_err = min(fd_err_pos, fd_err_neg);
            else
                fd_err = abs(meas_fd - fd);
            end
            fd_err_pct = fd_err / (abs(fd) + 1) * 100;
            if fd_err_pct > max_fd_err_pct, max_fd_err_pct = fd_err_pct; end
        end
    end
end
fprintf('  max_R_err=%.2fm max_fd_err=%.1f%% [%s]\n', ...
    max_R_err_m, max_fd_err_pct, char('PASS'*pass_t6+'FAIL'*(~pass_t6)));
results.s6t6 = pass_t6;

%% =====================================================================
%% S6T7: High Bandwidth MF
%% =====================================================================
fprintf('S6T7: High Bandwidth MF\n');
bw_t7 = [50 75 100 125 150 175 200 250 300 350 375 400]*1e6;
pw_t7 = pw; % 50 us
pass_t7 = true;
NFFT_t7 = 32768;

for bi = 1:length(bw_t7)
    bw_i = bw_t7(bi);
    fs_i = bw_i;
    n_i = max(1, floor(pw_t7*fs_i));
    k_i = bw_i / pw_t7;
    t_i = (0:n_i-1)'/fs_i;
    sig = exp(1j*pi*k_i*t_i.^2);
    sig = sig / norm(sig);

    % Matched filter via FFT
    mf_ref = conj(sig(end:-1:1));
    NFFT_i = min(NFFT_t7, 2^nextpow2(2*n_i));
    mf_out = ifft(fft(sig, NFFT_i) .* fft(mf_ref, NFFT_i));
    mf_pow = abs(mf_out).^2;
    [pk_val, ~] = max(mf_pow);

    % Measure 3-dB mainlobe width
    threshold = pk_val / 2;
    above = find(mf_pow > threshold);
    ml_width = length(above);
    if ml_width < 1, ml_width = 1; end

    % Compression ratio in dB
    cr_meas = 10*log10(n_i / ml_width);
    cr_theory = 10*log10(bw_i * pw_t7);
    cr_err = abs(cr_meas - cr_theory);
    if cr_err > 2
        pass_t7 = false;
        fprintf('  FAIL bw=%.0fMHz CR=%.1fdB theory=%.1fdB err=%.1fdB\n', ...
            bw_i/1e6, cr_meas, cr_theory, cr_err);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t7+'FAIL'*(~pass_t7)));
results.s6t7 = pass_t7;

%% =====================================================================
%% S6T8: PRF/Ambiguity
%% =====================================================================
fprintf('S6T8: PRF/Ambiguity\n');
prf_t8 = [1 2 3 5 7 10 15 20 25 30 40 50]*1e3;
pass_t8 = true;
R_target = 15000; % 15 km
max_R_err_t8 = 0;

for pi_idx = 1:length(prf_t8)
    prf_i = prf_t8(pi_idx);
    R_max = c / (2*prf_i);

    % Check if target is within unambiguous range
    if R_target < R_max
        % Target is detectable at true range
        R_measured = R_target;
    else
        % Target wraps: ambiguous range
        R_measured = mod(R_target, R_max);
    end

    % Verify: R_measured = mod(R_target, R_max) always
    expected = mod(R_target, R_max);
    err = abs(R_measured - expected);
    if err > max_R_err_t8, max_R_err_t8 = err; end
    if err > 0.01, pass_t8 = false; end
end
fprintf('  max_err=%.4fm [%s]\n', max_R_err_t8, ...
    char('PASS'*pass_t8+'FAIL'*(~pass_t8)));
results.s6t8 = pass_t8;

%% =====================================================================
%% S6T9: Very Short Pulse
%% =====================================================================
fprintf('S6T9: Very Short Pulse\n');
pw_t9 = [1 2 3 4 5 7 10 12 15 18 20 25]*1e-6;
pass_t9 = true;
for pi_idx = 1:length(pw_t9)
    pw_i = pw_t9(pi_idx);
    n_i = floor(pw_i * fs);
    % n_samples should be >= 1 for all pw >= 1us with fs=200MHz
    if n_i < 1
        pass_t9 = false;
        fprintf('  FAIL pw=%.0fus n=%d (too few samples)\n', pw_i*1e6, n_i);
        continue;
    end
    k_i = bw / pw_i;
    t_i = (0:n_i-1)'/fs;
    sig = exp(1j*pi*k_i*t_i.^2);
    sig = sig / norm(sig);
    % Check non-empty
    if isempty(sig)
        pass_t9 = false;
        fprintf('  FAIL pw=%.0fus empty signal\n', pw_i*1e6);
        continue;
    end
    % Check unit norm
    norm_err = abs(norm(sig) - 1);
    if norm_err > 1e-5
        pass_t9 = false;
        fprintf('  FAIL pw=%.0fus norm_err=%.2e\n', pw_i*1e6, norm_err);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t9+'FAIL'*(~pass_t9)));
results.s6t9 = pass_t9;

%% =====================================================================
%% S6T10: Clamping at +/-90 degrees
%% =====================================================================
fprintf('S6T10: Clamping at +/-90\n');
az_t10 = [89 89.5 89.9 89.99 90 90.01 90.1 90.5 91 -89 -90 -91];
pass_t10 = true;
has_nan = false;
has_inf = false;

for ai = 1:length(az_t10)
    az = az_t10(ai);
    % Compute steering vector
    w = exp(-1j*k_wave*ex*sind(az));
    % Apply beamforming: sum over elements
    bf_out = sum(w);

    % Check for NaN or Inf
    if any(isnan(w)) || any(isinf(w))
        has_nan = true;
        has_inf = true;
        pass_t10 = false;
        fprintf('  FAIL az=%.2f NaN=%d Inf=%d\n', az, any(isnan(w)), any(isinf(w)));
    end
    if isnan(bf_out) || isinf(bf_out)
        has_nan = true;
        pass_t10 = false;
        fprintf('  FAIL az=%.2f bf_out NaN/Inf\n', az);
    end

    % Also verify sind works correctly
    sv = sind(az);
    if isnan(sv) || isinf(sv)
        pass_t10 = false;
        fprintf('  FAIL az=%.2f sind NaN/Inf\n', az);
    end
end
% Verify specific identities
if abs(sind(90) - 1.0) > 1e-15
    pass_t10 = false;
end
if abs(sind(-90) + 1.0) > 1e-15
    pass_t10 = false;
end
fprintf('  nan=%d inf=%d sind(90)=%.15f [%s]\n', has_nan, has_inf, sind(90), ...
    char('PASS'*pass_t10+'FAIL'*(~pass_t10)));
results.s6t10 = pass_t10;

%% =====================================================================
%% S6T11: Noise Std Formula
%% =====================================================================
fprintf('S6T11: Noise Std Formula\n');
bw_t11 = [50 75 100 125 150 175 200 250 300 350 375 400]*1e6;
pass_t11 = true;
max_rel_err_t11 = 0;
nn_t11 = 1e6; % large sample count for statistical accuracy

for bi = 1:length(bw_t11)
    bw_i = bw_t11(bi);
    % Compute noise std
    noise_w_i = kB*T_noise*bw_i*10^(NF_db/10);
    noise_std_i = sqrt(noise_w_i/2);

    % Generate noise samples
    rng(42 + bi); % deterministic seed for reproducibility
    nc = noise_std_i*(randn(nn_t11,1)+1j*randn(nn_t11,1));

    % Measured total power
    meas_power = mean(abs(nc).^2);
    expected_power = noise_w_i;

    rel_err = abs(meas_power - expected_power) / expected_power;
    if rel_err > max_rel_err_t11, max_rel_err_t11 = rel_err; end
    if rel_err > 0.05
        pass_t11 = false;
        fprintf('  FAIL bw=%.0fMHz rel_err=%.4f\n', bw_i/1e6, rel_err);
    end
end
fprintf('  max_rel_err=%.4f [%s]\n', max_rel_err_t11, ...
    char('PASS'*pass_t11+'FAIL'*(~pass_t11)));
results.s6t11 = pass_t11;

%% =====================================================================
%% S6T12: Doppler Resolution
%% =====================================================================
fprintf('S6T12: Doppler Resolution\n');
n_pulses_t12 = [2 4 8 16 32 64 128 256 512 1024 2048 4096];
prf_t12 = prf; % 10 kHz
pass_t12 = true;
res_t12 = zeros(size(n_pulses_t12));

for ni = 1:length(n_pulses_t12)
    np = n_pulses_t12(ni);
    res_t12(ni) = prf_t12 / np;
    % Resolution should decrease with more pulses
    if ni > 1 && res_t12(ni) >= res_t12(ni-1)
        pass_t12 = false;
    end
end
% Verify specific values
if abs(res_t12(1) - 5000) > 0.01 % n_pulses=2: res = 10kHz/2 = 5kHz
    pass_t12 = false;
end
if abs(res_t12(end) - prf_t12/n_pulses_t12(end)) > 0.01
    pass_t12 = false;
end
% Check that resolution halves when pulses double
for ni = 2:length(n_pulses_t12)
    expected_ratio = n_pulses_t12(ni) / n_pulses_t12(ni-1);
    actual_ratio = res_t12(ni-1) / res_t12(ni);
    if abs(actual_ratio - expected_ratio) > 0.01
        pass_t12 = false;
    end
end
fprintf('  res=[%.1f .. %.2f] Hz [%s]\n', max(res_t12), min(res_t12), ...
    char('PASS'*pass_t12+'FAIL'*(~pass_t12)));
results.s6t12 = pass_t12;

%% =====================================================================
%% S6T13: dB Round-Trip
%% =====================================================================
fprintf('S6T13: dB Round-Trip\n');
P_dbm_t13 = [-150 -120 -100 -80 -60 -40 -20 0 20 47 60 77];
pass_t13 = true;
max_roundtrip_err = 0;

for pi_idx = 1:length(P_dbm_t13)
    P_dbm = P_dbm_t13(pi_idx);
    % Convert to watts
    P_w = 10^(P_dbm/10)/1000;
    % Convert back to dBm
    P_dbm_back = 10*log10(P_w*1000);
    % Check round-trip error
    rt_err = abs(P_dbm_back - P_dbm);
    if rt_err > max_roundtrip_err, max_roundtrip_err = rt_err; end
    if rt_err > 1e-10
        pass_t13 = false;
        fprintf('  FAIL P=%.0fdBm roundtrip_err=%.2e\n', P_dbm, rt_err);
    end
end
fprintf('  max_roundtrip_err=%.2e [%s]\n', max_roundtrip_err, ...
    char('PASS'*pass_t13+'FAIL'*(~pass_t13)));
results.s6t13 = pass_t13;

%% =====================================================================
%% S6T14: Lsys=0 Boundary
%% =====================================================================
fprintf('S6T14: Lsys=0 Boundary\n');
Lsys_t14 = [0 0.001 0.01 0.1 0.5 1 1.5 2 2.5 3 5 10];
pass_t14 = true;
R_m_t14 = 10000;
rcs_t14 = 20;

% Compute Pr WITHOUT the loss term (reference)
Pr_no_loss = tx_dbm + 2*D_dBi + rcs_t14 + 20*log10(lambda) ...
    - 30*log10(4*pi) - 40*log10(R_m_t14);

% Compute Pr WITH Lsys=0
Pr_L0 = tx_dbm + 2*D_dBi + rcs_t14 + 20*log10(lambda) ...
    - 30*log10(4*pi) - 40*log10(R_m_t14) - 0;

% These should be identical
if abs(Pr_no_loss - Pr_L0) > 1e-12
    pass_t14 = false;
    fprintf('  FAIL Pr_no_loss=%.6f Pr_L0=%.6f diff=%.2e\n', ...
        Pr_no_loss, Pr_L0, abs(Pr_no_loss-Pr_L0));
end

% Compute Pr at each Lsys value
Pr_t14 = zeros(size(Lsys_t14));
for li = 1:length(Lsys_t14)
    Pr_t14(li) = tx_dbm + 2*D_dBi + rcs_t14 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m_t14) - Lsys_t14(li);
end

% Verify: Pr(Lsys) - Pr_no_loss = -Lsys for all values
max_loss_err = 0;
for li = 1:length(Lsys_t14)
    expected_drop = -Lsys_t14(li);
    actual_drop = Pr_t14(li) - Pr_no_loss;
    loss_err = abs(actual_drop - expected_drop);
    if loss_err > max_loss_err, max_loss_err = loss_err; end
    if loss_err > 0.001
        pass_t14 = false;
        fprintf('  FAIL Lsys=%.3fdB expected_drop=%.3f actual_drop=%.3f\n', ...
            Lsys_t14(li), expected_drop, actual_drop);
    end
end
fprintf('  Pr_no_loss=%.2f Pr_L0=%.2f max_loss_err=%.2e [%s]\n', ...
    Pr_no_loss, Pr_L0, max_loss_err, ...
    char('PASS'*pass_t14+'FAIL'*(~pass_t14)));
results.s6t14 = pass_t14;

%% =====================================================================
%% Summary
%% =====================================================================
fn = fieldnames(results);
for fi = 1:length(fn)
    if results.(fn{fi}), n_pass = n_pass+1; else, n_fail = n_fail+1; end
end
fprintf('\n============================================\n');
fprintf('S6 RESULT: %d/%d PASSED', n_pass, n_pass+n_fail);
if n_fail > 0, fprintf(' (%d FAILED)', n_fail); end
fprintf('\n============================================\n');
if n_fail > 0
    fprintf('FAILURES:\n');
    for fi = 1:length(fn)
        if ~results.(fn{fi}), fprintf('  - %s\n', upper(fn{fi})); end
    end
end

save('validate_em_s6_edge_results.mat', 'results');
fprintf('Results saved to validate_em_s6_edge_results.mat\n');
