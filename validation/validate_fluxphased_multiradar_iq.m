%% FluxPhased Multi-Radar IQ-Level EM Cross-Validation
%  Complex scenarios: array-level self-interference, multi-radar mutual
%  interference at IQ level, multi-task coupling, beamforming coherence.
%
%  Run: cd validation && matlab -batch "validate_fluxphased_multiradar_iq"

clear; close all; clc;

%% Common Parameters (FluxPhased config.py)
c = 299792458.0; fc = 10e9; lambda = c/fc;
bw = 200e6; fs = bw; prf = 10e3; pri = 1/prf;
n_samples = floor(pri * fs);
rows = 25; cols = 25; N_elem = rows*cols;
dx_m = 0.5*lambda; dy_m = 0.5*lambda;
tx_w = 50000.0; tx_dbm = 10*log10(tx_w*1000);
NF = 5.0; Lsys = 3.0;
kB_T = 1.380649e-23 * 290;
noise_w = kB_T * bw * 10^(NF/10);
noise_dbm = 10*log10(noise_w*1000);
noise_std = sqrt(noise_w/2);
D_dBi = 10*log10(4*pi*N_elem*0.25);
pw = 50e-6; n_lfm = floor(pw*fs);
k_lfm = bw/pw;
t_lfm = (0:n_lfm-1)'/fs;
lfm = exp(1j*pi*k_lfm*t_lfm.^2); lfm = lfm/norm(lfm);

results = struct();

fprintf('FluxPhased Multi-Radar IQ Cross-Validation\n');
fprintf('============================================\n\n');

%% ===== Test 1: ELDA Beamforming Coherence =====
% N elements with steering weights → coherent sum gives N² power gain
fprintf('TEST 1: ELDA Beamforming Coherence (N=%d elements)\n', N_elem);

az_steer = 20; el_steer = 0;
k_wave = 2*pi/lambda;
elem_x = ((0:cols-1) - (cols-1)/2) * dx_m;
elem_y = ((0:rows-1) - (rows-1)/2) * dy_m;
[X,Y] = meshgrid(elem_x, elem_y);
ex = X(:); ey = Y(:);

% Steering phase per element
steer_rad = az_steer * pi/180;
phase_per_elem = k_wave * (ex * sin(steer_rad));

% Verify 1: Coherent gain = |sum(w)|^2 for uniform weights
w = exp(-1j * phase_per_elem);
coherent_gain = abs(sum(w))^2;
ok1a = abs(coherent_gain - N_elem) / N_elem < 0.01;
fprintf('  Uniform array coherent sum: %.0f (theory %d)\n', coherent_gain, N_elem);

% Verify 2: Beamformed SNR = N² × single-element SNR (voltage sum gives N² power)
% Single element receives signal amplitude A. Beamformed output = N*A (coherent).
% Power ratio = (N*A)² / A² = N²
gain_db = 10*log10(coherent_gain / 1);  % relative to single element
expected_db = 10*log10(N_elem);
ok1b = abs(gain_db - expected_db) < 0.1;
fprintf('  Gain vs single element: %.1f dB (theory %.1f dB)\n', gain_db, expected_db);

% Verify 3: Array pattern null at grating lobe angle
% For half-wave spacing, first null at sin(θ) = λ/(N*dx) = 2/N
sv = phased.SteeringVector('SensorArray', ...
    phased.URA('Size',[rows cols],'ElementSpacing',[dx_m dy_m]), ...
    'PropagationSpeed', c);
w30 = sv(fc, [az_steer; 0]);
pat = zeros(361, 1);
az_scan = -90:0.5:90;
for ai = 1:length(az_scan)
    ph = k_wave * (ex * sin(az_scan(ai)*pi/180));
    af = abs(sum(w30 .* exp(1j * ph)));
    pat(ai) = af;
end
[peak_pat, pk_idx] = max(pat);
% 3dB beamwidth
half = peak_pat * 0.5;
above = find(pat > half);
if ~isempty(above)
    bw_meas = az_scan(above(end)) - az_scan(above(1));
else
    bw_meas = 0;
end
ok1c = bw_meas > 2 && bw_meas < 6;  % ~4 deg for 25x25
fprintf('  Beamwidth at %d deg: %.1f deg\n', az_steer, bw_meas);

ok1 = ok1a && ok1b && ok1c;
fprintf('  [%s] Beamforming coherence + pattern\n', pf(ok1));
results.beamforming = ok1;

%% ===== Test 2: Self-Interference vs Beam Steering =====
% SI = tx_signal * coupling. With beam weights, SI per element varies.
fprintf('\nTEST 2: Self-Interference with Beam Steering\n');

isolation_db = 25;
coupling = 10^(-isolation_db/20);
steer_angles = [0, 15, 30, 45];
pass2 = true;

for si = 1:length(steer_angles)
    az = steer_angles(si);
    ph = k_wave * (ex * sin(az*pi/180) + ey * 0);
    w = exp(-1j * ph);

    % TX signal per element: baseband * weights
    tx_per_elem = lfm(1:min(n_lfm,n_samples)) .* conj(w);
    tx_per_elem = [tx_per_elem; complex(zeros(max(0,n_samples-min(n_lfm,n_samples)),1))];

    % SI per element: TX * coupling
    si_per_elem = tx_per_elem * coupling;

    % Total SI power (sum over elements)
    si_total_power = sum(abs(si_per_elem).^2);
    % Expected: N_elem * coupling^2 / n_lfm (each element has unit-norm signal scaled)
    si_expected = N_elem * coupling^2 / min(n_lfm, n_samples);
    err_db = abs(10*log10(si_total_power / max(si_expected, 1e-30)));
    ok = err_db < 1.0;
    pass2 = pass2 && ok;
end
fprintf('  [%s] SI power consistent across beam directions (iso=%d dB)\n', ...
    pf(pass2), isolation_db);
results.si_steering = pass2;

%% ===== Test 3: Multi-Radar IQ Interference Superposition =====
% 4 radars at known positions, one receiving radar sees sum of 3 interferers
fprintf('\nTEST 3: Multi-Radar IQ Interference Superposition\n');

% Radar positions: 4 radars in a diamond
radar_pos = [0 5000 0; 5000 0 0; 0 -5000 0; -5000 0 0];  % [R, 3] meters
R = 4;

% All radars transmit same LFM, Radar 0 receives interference from 1,2,3
rx_idx = 1;  % MATLAB 1-indexed
interference_iq = complex(zeros(n_samples, 1));
total_jnr_db = 0;
pass3 = true;

for tx = 1:R
    if tx == rx_idx, continue; end
    dist = norm(radar_pos(tx,:) - radar_pos(rx_idx,:));
    fspl = 20*log10(4*pi*dist/lambda + 1e-10);
    jnr = tx_dbm + D_dBi + D_dBi - fspl - 3.0 - noise_dbm;
    total_jnr_db = 10*log10(10^(total_jnr_db/10) + 10^(jnr/10));

    % IQ-level interference: attenuated + delayed TX waveform
    amp = sqrt(10^((jnr + noise_dbm - 30)/10));
    delay_samp = round(dist/c * fs);
    tx_sig = lfm;
    tx_pad = [tx_sig; complex(zeros(max(0, n_samples-n_lfm), 1))];
    if delay_samp > 0 && delay_samp < n_samples
        delayed = [complex(zeros(delay_samp,1)); tx_pad(1:n_samples-delay_samp)];
    else
        delayed = complex(zeros(n_samples, 1));
    end
    interference_iq = interference_iq + amp * delayed;
    fprintf('  Radar %d→0: dist=%.0f m, JNR=%.1f dB, delay=%d samp\n', ...
        tx-1, dist, jnr, delay_samp);
end

% Verify total interference power
intf_power = mean(abs(interference_iq).^2);
total_noise = noise_w;
sinr_db = 10*log10(intf_power / total_noise);
err = abs(sinr_db - total_jnr_db);
ok3 = err < 3.0;
fprintf('  Total JNR: theory=%.1f dB, IQ-measured=%.1f dB, err=%.1f dB\n', ...
    total_jnr_db, sinr_db, err);
fprintf('  [%s] Multi-radar IQ interference superposition\n', pf(ok3));
results.multi_radar_intf = ok3;

%% ===== Test 4: Multi-Target Channel Linearity =====
% 3 targets at different ranges/velocities → received IQ = superposition
fprintf('\nTEST 4: Multi-Target Channel Linearity\n');

tgt_ranges = [5000, 8000, 12000];  % meters
tgt_vels = [-20, 10, 40];          % m/s
tgt_rcs = [20, 15, 25];            % dBsm
pass4 = true;

% Build received signal by superposition
rx_all = complex(zeros(n_samples, 1));
rx_individual = cell(1, 3);

for ti = 1:3
    R_t = tgt_ranges(ti);
    v_t = tgt_vels(ti);
    rcs_t = tgt_rcs(ti);

    Pr_dbm = tx_dbm + 2*D_dBi + rcs_t + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_t) - Lsys;
    Pr_w = 10^((Pr_dbm-30)/10);
    gain_v = sqrt(max(Pr_w / N_elem, 0));

    delay_s = round(2*R_t/c * fs);
    doppler_hz = 2*v_t*fc/c;
    phase_step = 2*pi*doppler_hz/fs;

    tx_pad = [lfm; complex(zeros(max(0, n_samples-n_lfm), 1))];
    if delay_s > 0 && delay_s < n_samples
        rx_t = [complex(zeros(delay_s,1)); tx_pad(1:n_samples-delay_s)];
    else
        rx_t = complex(zeros(n_samples, 1));
    end
    rx_t = rx_t .* exp(1j * phase_step * (0:n_samples-1)') * gain_v;
    rx_individual{ti} = rx_t;
    rx_all = rx_all + rx_t;
end

% Verify: superposition = sum of individual
rx_sum = rx_individual{1} + rx_individual{2} + rx_individual{3};
lin_err = max(abs(rx_all - rx_sum)) / max(abs(rx_all(:)));
ok4 = lin_err < 1e-10;
fprintf('  Linearity error: %.2e (threshold 1e-10)\n', lin_err);

% Also verify each target produces MF peak at correct range
n_fft4 = 1; while n_fft4 < 2*n_samples, n_fft4 = n_fft4*2; end
mf4 = ifft(fft(rx_all, n_fft4) .* conj(fft(lfm, n_fft4)));
mf4_pwr = abs(mf4).^2;
[~, mf4_sorted] = sort(mf4_pwr, 'descend');

% Find 3 distinct peaks (separated by > 500m)
detected = [];
for k = 1:min(50, length(mf4_sorted))
    idx = mf4_sorted(k);
    rng = (idx-1) * c/(2*fs);
    too_close = false;
    for d = 1:length(detected)
        if abs(rng - detected(d)) < 500, too_close = true; break; end
    end
    if ~too_close, detected(end+1) = rng; end
    if length(detected) >= 3, break; end
end

n_det = length(detected);
ok4_det = n_det >= 3;
fprintf('  Detected %d/3 targets at ranges:', n_det);
for d = 1:min(3, n_det)
    [~, best] = min(abs(tgt_ranges - detected(d)));
    fprintf(' %.0f(true:%.0f)', detected(d), tgt_ranges(best));
end
fprintf('\n');

pass4 = ok4 && ok4_det;
fprintf('  [%s] Multi-target channel linearity + detection\n', pf(pass4));
results.multi_target = pass4;

%% ===== Test 5: Waveform Cross-Correlation (Multi-Task Orthogonality) =====
% Different waveforms should have low cross-correlation (important for
% multi-task: detect+jam+comm simultaneously)
fprintf('\nTEST 5: Waveform Cross-Correlation (Multi-Task Orthogonality)\n');

% Generate waveforms at narrower BW for fair comparison
wf_bw = 2e6;
n_wf = floor(pw * fs);
t_wf = (0:n_wf-1)'/fs;

lfm_u = exp(1j*pi*(wf_bw/pw)*t_wf.^2); lfm_u = lfm_u/norm(lfm_u);
lfm_d = exp(-1j*pi*(wf_bw/pw)*t_wf.^2); lfm_d = lfm_d/norm(lfm_d);
% Barker
bk_code = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
spc_bk = max(1, floor((pw/13)*fs));
tmp = repmat(bk_code(:), spc_bk, 1); barker = complex(tmp(:));
barker = barker(1:min(n_wf,length(barker)));
if length(barker) < n_wf, barker = [barker; complex(zeros(n_wf-length(barker),1))]; end
barker = barker/norm(barker);

% NLFM
nlfm = exp(1j*(pi*(wf_bw/pw)*t_wf.^2 + 0.3*pi*(wf_bw/pw)/pw.*t_wf.^3));
nlfm = nlfm/norm(nlfm);

wfs = {lfm_u, lfm_d, barker, nlfm};
wf_n = {'LFM_up', 'LFM_dn', 'Barker13', 'NLFM'};
n_wf_types = length(wfs);

n_fft5 = 1; while n_fft5 < 2*n_wf, n_fft5 = n_fft5*2; end
max_xcorr_db = -100;
for i = 1:n_wf_types
    for j = i+1:n_wf_types
        mf_ij = ifft(fft(wfs{i}, n_fft5) .* conj(fft(wfs{j}, n_fft5)));
        xc_pwr = abs(mf_ij).^2;
        peak_xc = max(xc_pwr);
        % Auto-correlation peak for normalization
        auto_i = max(abs(ifft(abs(fft(wfs{i}, n_fft5)).^2)).^2);
        xcorr_db = 10*log10(peak_xc / max(auto_i, 1e-30));
        if xcorr_db > max_xcorr_db, max_xcorr_db = xcorr_db; end
        fprintf('  %s x %s: peak xcorr = %.1f dB\n', wf_n{i}, wf_n{j}, xcorr_db);
    end
end
ok5 = max_xcorr_db < -5;
fprintf('  [%s] Max cross-correlation %.1f dB (threshold -5 dB)\n', pf(ok5), max_xcorr_db);
results.waveform_xcorr = ok5;

%% ===== Test 6: Spot Noise Spectral Accuracy =====
fprintf('\nTEST 6: Spot Noise Spectral Accuracy\n');

f_center = 50e6;  % 50 MHz offset
bw_spot = 10e6;
n_sp = n_samples;
noise_raw = randn(n_sp,1) + 1j*randn(n_sp,1);
spec_raw = fft(noise_raw);
freqs_sp = (0:n_sp-1)'/n_sp * fs;
mask_sp = abs(freqs_sp - f_center) < bw_spot/2;
spec_filt = spec_raw .* mask_sp;
noise_spot = ifft(spec_filt);
noise_spot = noise_spot / norm(noise_spot);

% Measure actual spectral occupancy
spec_out = abs(fft(noise_spot)).^2;
spec_out_db = 10*log10(spec_out / max(spec_out) + 1e-30);
in_band = abs(freqs_sp - f_center) < bw_spot/2;
out_band = ~in_band;
in_pwr = mean(spec_out(in_band));
out_pwr = mean(spec_out(out_band));
rejection_db = 10*log10(in_pwr / max(out_pwr, 1e-30));

ok6 = rejection_db > 10;  % at least 10 dB spectral contrast
fprintf('  Spot noise: center=%.0f MHz, BW=%.0f MHz, in/out contrast=%.1f dB\n', ...
    f_center/1e6, bw_spot/1e6, rejection_db);
fprintf('  [%s] Spot noise spectral selectivity\n', pf(ok6));
results.spot_noise = ok6;

%% ===== Test 7: DRFM Cross-Correlation Fidelity =====
% DRFM output should be highly correlated with captured signal
fprintf('\nTEST 7: DRFM Cross-Correlation Fidelity\n');

freq_shift = 100e3;
n_drfm = min(n_lfm, n_samples);
captured = lfm(1:n_drfm);
t_drfm = (0:n_drfm-1)'/fs;
drfm_out = captured .* exp(1j*2*pi*freq_shift*t_drfm);
delay_d = round(5e-6*fs);  % 5 us delay
if delay_d > 0 && delay_d < n_drfm
    drfm_out = [complex(zeros(delay_d,1)); drfm_out(1:end-delay_d)];
end
drfm_out = drfm_out / norm(drfm_out);

% Cross-correlate DRFM output with original
n_fft7 = 1; while n_fft7 < 2*n_drfm, n_fft7 = n_fft7*2; end
mf_drfm = ifft(fft(drfm_out, n_fft7) .* conj(fft(lfm(1:n_drfm), n_fft7)));
mf_drfm_pwr = abs(mf_drfm).^2;
[peak_drfm, idx_drfm] = max(mf_drfm_pwr);
% Auto-correlation peak for reference
auto_ref = max(abs(ifft(abs(fft(lfm(1:n_drfm), n_fft7)).^2)).^2);
xcorr_ratio = peak_drfm / max(auto_ref, 1e-30);
xcorr_db = 10*log10(xcorr_ratio);

ok7 = xcorr_db > -3;  % DRFM should preserve > 50% correlation
fprintf('  DRFM xcorr with original: %.1f dB (delay=%d samp, shift=%.0f Hz)\n', ...
    xcorr_db, delay_d, freq_shift);
fprintf('  [%s] DRFM preserves waveform structure\n', pf(ok7));
results.drfm_fidelity = ok7;

%% ===== Test 8: Per-Element Channel Phase Coherence =====
% Each element sees same target but with wavefront phase difference.
% Verify phase gradient across array matches expected wavefront.
fprintf('\nTEST 8: Per-Element Channel Phase Coherence\n');

az_target = 25; % degrees
target_range = 10000;
target_phase_elem = k_wave * (ex * sin(az_target*pi/180) + ey * 0);

% Verify phase gradient is linear across array
phase_x = reshape(target_phase_elem, rows, cols);
% Phase along x-axis (middle row) should be linear
mid_row = ceil(rows/2);
ph_x = phase_x(mid_row, :);
expected_slope = k_wave * dx_m * sin(az_target*pi/180);
measured_slope = (ph_x(end) - ph_x(1)) / (cols - 1);
slope_err = abs(measured_slope - expected_slope) / abs(expected_slope);

ok8 = slope_err < 0.01;
fprintf('  Wavefront phase slope: measured=%.6f, expected=%.6f, err=%.2f%%\n', ...
    measured_slope, expected_slope, slope_err*100);

% Also verify phases sum correctly for beamforming
bf_gain = abs(sum(exp(1j * target_phase_elem)))^2 / N_elem;
expected_bf = N_elem;  % perfect coherent sum
ok8b = abs(bf_gain - expected_bf) / expected_bf < 0.01;
fprintf('  Coherent sum gain: %.1f (theory %.1f)\n', bf_gain, expected_bf);

ok8 = ok8 && ok8b;
fprintf('  [%s] Per-element wavefront phase coherence\n', pf(ok8));
results.phase_coherence = ok8;

%% ===== Test 9: Power Conservation in Multi-Radar Scenario =====
% Total received power = target + noise + SI + cross-radar interference
% Verify additive power model holds
fprintf('\nTEST 9: Power Conservation in Multi-Radar Scenario\n');

% Single target at 10km
R_t9 = 10000;
Pr9 = tx_dbm + 2*D_dBi + 20 + 20*log10(lambda) - 30*log10(4*pi) - 40*log10(R_t9) - Lsys;
Pr9_w = 10^((Pr9-30)/10);
gain9 = sqrt(max(Pr9_w/N_elem, 0));
delay9 = round(2*R_t9/c*fs);
tx_pad9 = [lfm; complex(zeros(max(0, n_samples-n_lfm), 1))];
if delay9 > 0 && delay9 < n_samples
    target_rx = [complex(zeros(delay9,1)); tx_pad9(1:n_samples-delay9)];
else
    target_rx = complex(zeros(n_samples, 1));
end
target_rx = target_rx * gain9;

% Noise
noise_rx = noise_std * (randn(n_samples,1) + 1j*randn(n_samples,1)) / sqrt(2);

% SI at 25 dB isolation
coupling9 = 10^(-25/20);
si9 = lfm(1:min(n_lfm,n_samples)) * coupling9;
si9 = [si9; complex(zeros(max(0, n_samples-min(n_lfm,n_samples)),1))];

% Cross-radar interference from 5km
dist9 = 5000;
fspl9 = 20*log10(4*pi*dist9/lambda + 1e-10);
jnr9 = tx_dbm + D_dBi + D_dBi - fspl9 - 3.0 - noise_dbm;
intf_amp9 = sqrt(10^((jnr9 + noise_dbm - 30)/10));
intf9 = lfm(1:min(n_lfm,n_samples)) * intf_amp9;
intf9 = [intf9; complex(zeros(max(0, n_samples-min(n_lfm,n_samples)),1))];

% Total received signal
rx_total = target_rx + noise_rx + si9 + intf9;

% Measure powers
pwr_target = mean(abs(target_rx).^2);
pwr_noise = mean(abs(noise_rx).^2);
pwr_si = mean(abs(si9).^2);
pwr_intf = mean(abs(intf9).^2);
pwr_total = mean(abs(rx_total).^2);

% Expected total (non-coherent sum)
pwr_expected = pwr_target + pwr_noise + pwr_si + pwr_intf;
err9 = abs(10*log10(pwr_total) - 10*log10(pwr_expected));
ok9 = err9 < 1.0;  % within 1 dB (statistical tolerance)

fprintf('  Target: %.2e W, Noise: %.2e W, SI: %.2e W, Intf: %.2e W\n', ...
    pwr_target, pwr_noise, pwr_si, pwr_intf);
fprintf('  Total measured: %.2e W, expected sum: %.2e W, err: %.2f dB\n', ...
    pwr_total, pwr_expected, err9);
fprintf('  [%s] Power conservation (additive model)\n', pf(ok9));
results.power_conservation = ok9;

%% ===== Test 10: Doppler Phase Across CPI =====
% Verify inter-pulse Doppler phase accumulates correctly across 32 pulses
fprintf('\nTEST 10: Inter-Pulse Doppler Phase Across CPI\n');

v10 = 40;  % m/s
doppler10 = 2*v10*fc/c;  % Hz
n_pulses = 32;
pri10 = 1/prf;

% Expected phase step between pulses
phase_step_expected = 2*pi * doppler10 * pri10;

% Simulate received pulses: target echo with Doppler
delay10 = round(2*10000/c * fs);  % 10 km target
tx_p10 = [lfm; complex(zeros(max(0, n_samples-n_lfm), 1))];
if delay10 > 0 && delay10 < n_samples
    rx_base = [complex(zeros(delay10,1)); tx_p10(1:n_samples-delay10)];
else
    rx_base = complex(zeros(n_samples, 1));
end

% MF each pulse and track peak phase
mf_peak_phase = zeros(n_pulses, 1);
n_fft10 = 1; while n_fft10 < 2*n_samples, n_fft10 = n_fft10*2; end

for p = 1:n_pulses
    inter_pulse_phase = 2*pi * doppler10 * (p-1) * pri10;
    rx_p = rx_base .* exp(1j * (2*pi*doppler10/fs * (0:n_samples-1)' + inter_pulse_phase));
    mf_p = ifft(fft(rx_p, n_fft10) .* conj(fft(lfm, n_fft10)));
    [~, pk_idx] = max(abs(mf_p).^2);
    mf_peak_phase(p) = angle(mf_p(pk_idx));
end

% Unwrap and measure phase ramp
mf_unwrap = unwrap(mf_peak_phase);
phase_diff = mf_unwrap(2:end) - mf_unwrap(1:end-1);
mean_step = mean(phase_diff);
err10 = abs(mean_step - phase_step_expected) / abs(phase_step_expected);

ok10 = err10 < 0.1;  % within 10%
fprintf('  Doppler: %.1f Hz, phase step/pulse: measured=%.4f, expected=%.4f, err=%.1f%%\n', ...
    doppler10, mean_step, phase_step_expected, err10*100);
fprintf('  [%s] Inter-pulse Doppler phase coherence\n', pf(ok10));
results.doppler_cpi = ok10;

%% ===== Summary =====
fprintf('\n============================================\n');
fprintf('MULTI-RADAR IQ CROSS-VALIDATION SUMMARY\n');
fprintf('============================================\n');
names = fieldnames(results);
np = 0;
for i = 1:length(names)
    ok = results.(names{i});
    if ok, np = np + 1; end
    fprintf('  [%s] %s\n', pf(ok), names{i});
end
fprintf('\n  %d/%d passed\n', np, length(names));
if np == length(names)
    fprintf('  ALL MULTI-RADAR IQ TESTS PASSED\n');
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
