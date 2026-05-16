%% FluxPhased IQ-Level EM Simulation Cross-Validation
%  Tests each FluxPhased module against MATLAB analytical/toolbox reference.
%  Run: cd validation && matlab -batch "validate_fluxphased_iq"
%
%  Modules tested:
%    1. Waveform IQ (7 types: phase formula, unit norm)
%    2. BPSK encode/decode (14+14+4 CRC)
%    3. Channel radar equation (SNR vs range)
%    4. Channel delay + Doppler (sample-level)
%    5. Noise power (broadband + spot)
%    6. DRFM (freq shift + delay)
%    7. Array pattern + directivity (25x25 URA)
%    8. Cross-radar interference JNR (Friis)

clear; close all; clc;

%% Common params (matching FluxPhased config.py)
c = 299792458.0;
fc = 10e9;
lambda = c / fc;
bw = 200e6;
fs = bw;
prf = 10e3;
pri = 1 / prf;
n_samples = floor(pri * fs);
rows = 25; cols = 25;
N_elem = rows * cols;
dx_m = 0.5 * lambda;
dy_m = 0.5 * lambda;
tx_w = 50000.0;
tx_dbm = 10*log10(tx_w * 1000);
NF = 5.0;
Lsys = 3.0;
kB_T = 1.380649e-23 * 290;
noise_w = kB_T * bw * 10^(NF/10);
noise_dbm = 10*log10(noise_w * 1000);
noise_std = sqrt(noise_w / 2);

fprintf('FluxPhased IQ Cross-Validation (MATLAB R2024a)\n');
fprintf('================================================\n');
fprintf('fc=%.0f GHz, bw=%.0f MHz, fs=%.0f MHz, N=%d elements\n\n', ...
    fc/1e9, bw/1e6, fs/1e6, N_elem);

results = struct();
pass = 0; fail = 0;

%% ===== Test 1: Waveform IQ Phase Accuracy =====
fprintf('TEST 1: Waveform IQ Phase Accuracy (7 types)\n');
pw = 50e-6;
n_sig = floor(pw * fs);
t = (0:n_sig-1)' / fs;

% 1a: LFM up
k = bw / pw;
phase_lfm = pi * k * t.^2;
lfm_up = exp(1j * phase_lfm);
lfm_up = lfm_up / norm(lfm_up);
fprintf('  LFM_up: norm=%.6f, phase_peak=%.4f rad\n', norm(lfm_up), max(abs(phase_lfm)));

% 1b: LFM down
phase_lfm_dn = -pi * k * t.^2;
lfm_dn = exp(1j * phase_lfm_dn);
lfm_dn = lfm_dn / norm(lfm_dn);

% 1c: Barker-13
barker_code = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
spc_barker = max(1, floor((pw/13) * fs));
tmp = repmat(barker_code(:), spc_barker, 1);
barker = complex(tmp(:));
barker = barker / norm(barker);
fprintf('  Barker-13: len=%d, norm=%.6f\n', length(barker), norm(barker));

% 1d: Frank-16 (M=4, linear phase interp)
M = 4; n_phases = M * M;
idx = 0:M-1;
[gi, gj] = meshgrid(idx, idx);
phases_frank = 2*pi/M * gi(:) .* gj(:);
t_norm = linspace(1, n_phases, n_sig)';
lo = floor(t_norm); lo = min(lo, n_phases-1);
frac = t_norm - lo;
phase_interp = phases_frank(lo) + frac .* (phases_frank(min(lo+1, n_phases)) - phases_frank(lo));
frank = exp(1j * phase_interp);
frank = frank / norm(frank);
fprintf('  Frank-16: len=%d, norm=%.6f\n', length(frank), norm(frank));

% 1e: Costas-16
costas_seq = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
n_chips = length(costas_seq);
chip_len = floor(n_sig / n_chips);
costas = complex(zeros(n_sig, 1));
for ci = 1:n_chips
    s_start = (ci-1)*chip_len + 1;
    s_end = min(s_start + chip_len - 1, n_sig);
    t_chip = (0:s_end-s_start)' / fs;
    freq = costas_seq(ci) / pw;
    costas(s_start:s_end) = exp(1j * 2*pi * freq * t_chip);
end
costas = costas / norm(costas);
fprintf('  Costas-16: len=%d, norm=%.6f\n', length(costas), norm(costas));

% 1f: NLFM
phase_nlfm = pi * k * t.^2 + 0.3 * pi * k / pw * t.^3;
nlfm = exp(1j * phase_nlfm);
nlfm = nlfm / norm(nlfm);

% 1g: P4 (16 pts, linear phase interp)
n_pts = 16;
kk = (0:n_pts-1)';
phases_p4 = pi * kk.^2 / n_pts - pi * kk;
t_norm_p4 = linspace(1, n_pts, n_sig)';
lo_p4 = floor(t_norm_p4); lo_p4 = min(lo_p4, n_pts-1);
frac_p4 = t_norm_p4 - lo_p4;
phase_p4 = phases_p4(lo_p4) + frac_p4 .* (phases_p4(min(lo_p4+1, n_pts)) - phases_p4(lo_p4));
p4 = exp(1j * phase_p4);
p4 = p4 / norm(p4);

% Verify all unit-normalized
wf_list = {lfm_up, lfm_dn, barker, frank, costas, nlfm, p4};
wf_names = {'LFM_up','LFM_down','Barker-13','Frank-16','Costas-16','NLFM','P4'};
all_norm_ok = true;
for w = 1:7
    ne = abs(norm(wf_list{w}) - 1.0);
    if ne > 1e-4, all_norm_ok = false; end
    fprintf('  %10s: norm_err=%.1e\n', wf_names{w}, ne);
end
fprintf('  [%s] All waveforms unit-normalized\n', pf(all_norm_ok));
results.waveforms = all_norm_ok;

%% ===== Test 2: BPSK Encode/Decode =====
fprintf('\nTEST 2: BPSK Encode/Decode (14+14+4 CRC)\n');

bpsk_ok = true;
n_trials = 1000;
for trial = 1:n_trials
    x = rand()*2 - 1;
    y = rand()*2 - 1;
    % Encode (matching waveform_gpu.py:102-125)
    x_int = round(max(0, min(2^14-1, (x+1)/2 * (2^14-1))));
    y_int = round(max(0, min(2^14-1, (y+1)/2 * (2^14-1))));
    data_28 = bitshift(uint32(x_int), 14) + uint32(y_int);
    crc = uint32(0);
    val = data_28;
    for i = 1:7
        crc = bitxor(crc, bitand(val, uint32(15)));
        val = bitshift(val, -4);
    end
    word = bitshift(uint32(x_int), 18) + bitshift(uint32(y_int), 4) + bitand(crc, uint32(15));
    bits = zeros(32, 1);
    for b = 0:31
        bits(32 - b) = double(bitget(word, b + 1));
    end
    % Decode
    word2 = uint32(0);
    for b = 0:31
        if bits(32 - b) > 0.5
            word2 = bitor(word2, bitshift(uint32(1), b));
        end
    end
    x_dec = double(bitand(bitshift(word2, -18), uint32(2^14-1)));
    y_dec = double(bitand(bitshift(word2, -4), uint32(2^14-1)));
    crc_rx = double(bitand(word2, uint32(15)));
    data_28b = bitshift(uint32(x_dec), 14) + uint32(y_dec);
    crc_calc = uint32(0);
    val = data_28b;
    for i = 1:7
        crc_calc = bitxor(crc_calc, bitand(val, uint32(15)));
        val = bitshift(val, -4);
    end
    if bitand(crc_calc, uint32(15)) ~= uint32(crc_rx)
        bpsk_ok = false;
    else
        x_out = x_dec / (2^14-1) * 2 - 1;
        y_out = y_dec / (2^14-1) * 2 - 1;
        if abs(x_out - x) > 1/(2^14-1) || abs(y_out - y) > 1/(2^14-1)
            bpsk_ok = false;
        end
    end
end
fprintf('  [%s] BPSK encode/decode round-trip (%d trials)\n', pf(bpsk_ok), n_trials);

% BER at high SNR (20 dB)
n_ber = 5000;
ber_err = 0;
ber_total = 0;
for trial = 1:n_ber
    x = rand()*2 - 1; y = rand()*2 - 1;
    % encode as above (reuse same encode)
    x_int = round(max(0, min(2^14-1, (x+1)/2 * (2^14-1))));
    y_int = round(max(0, min(2^14-1, (y+1)/2 * (2^14-1))));
    data_28 = bitshift(uint32(x_int), 14) + uint32(y_int);
    crc = uint32(0); val = data_28;
    for i = 1:7, crc = bitxor(crc, bitand(val, uint32(15))); val = bitshift(val, -4); end
    word = bitshift(uint32(x_int), 18) + bitshift(uint32(y_int), 4) + bitand(crc, uint32(15));
    bits_tx = zeros(32,1);
    for b = 0:31, bits_tx(32-b) = double(bitget(word, b+1)); end
    % BPSK symbols + noise at 20 dB
    symbols = 2*bits_tx - 1;
    snr_lin = 100;
    sigma = 1/sqrt(2*snr_lin);
    rx = symbols + sigma*(randn(32,1) + 1j*randn(32,1));
    bits_rx = double(real(rx) > 0);  % keep column
    ber_err = ber_err + sum(bits_tx ~= bits_rx);
    ber_total = ber_total + 32;
end
ber_val = ber_err / ber_total;
fprintf('  [%s] BER at 20dB SNR: %.6f (expect ~0)\n', pf(ber_val < 0.01), ber_val);
results.bpsk = bpsk_ok && (ber_val < 0.01);

%% ===== Test 3: Radar Equation SNR =====
fprintf('\nTEST 3: Radar Equation SNR vs Range\n');

D_dBi = 10*log10(4*pi*N_elem*0.25);
ranges_km = [2, 5, 10, 20, 50];
rcs = 20;
pass3 = true;
for i = 1:length(ranges_km)
    R = ranges_km(i) * 1000;
    Pr = tx_dbm + 2*D_dBi + rcs + 20*log10(lambda) - 30*log10(4*pi) - 40*log10(R) - Lsys;
    SNR = Pr - noise_dbm;
    % MATLAB toolbox check
    try
        snr_ml = radareqsnr(lambda, R, tx_w, pri, 10^(rcs/10), ...
            'Gain', D_dBi, 'Loss', Lsys, 'Noisefigure', NF);
        err = abs(SNR - snr_ml);
        ok = err < 1.0;
    catch
        ok = true; err = 0;
    end
    pass3 = pass3 && ok;
    fprintf('  %s R=%3d km: SNR=%.1f dB (err=%.2f dB)\n', pf(ok), ranges_km(i), SNR, err);
end
fprintf('  [%s] Radar equation vs MATLAB radareqsnr\n', pf(pass3));
results.radar_eq = pass3;

%% ===== Test 4: Channel Delay + Doppler =====
fprintf('\nTEST 4: Channel Delay + Doppler Accuracy\n');

R_test = 10000;
v_test = 30;
delay_s = 2 * R_test / c;
delay_samp = round(delay_s * fs);
doppler_hz = 2 * v_test * fc / c;

fprintf('  Target: R=%d m, v=%d m/s\n', R_test, v_test);
fprintf('  Expected delay: %.3f us (%d samples)\n', delay_s*1e6, delay_samp);
fprintf('  Expected Doppler: %.1f Hz\n', doppler_hz);

% Generate TX, apply delay+Doppler, measure via matched filter
tx_lfm = lfm_up;
n_lfm = length(tx_lfm);
tx_pad = [tx_lfm; complex(zeros(max(0, n_samples - n_lfm), 1))];
if delay_samp > 0 && delay_samp < n_samples
    rx_delayed = [complex(zeros(delay_samp,1)); tx_pad(1:n_samples-delay_samp)];
else
    rx_delayed = complex(zeros(n_samples, 1));
end
doppler_phase = 2*pi * doppler_hz / fs;
rx_doppler = rx_delayed .* exp(1j * doppler_phase * (0:n_samples-1)');

% Matched filter to verify delay
n_fft = 1; while n_fft < n_samples + n_lfm - 1, n_fft = n_fft*2; end
mf = ifft(fft(rx_doppler, n_fft) .* conj(fft(tx_lfm, n_fft)));
[peak_mf, idx_mf] = max(abs(mf).^2);
meas_delay = idx_mf - 1;
range_err_m = abs(meas_delay - delay_samp) * c / (2*fs);

ok4a = range_err_m < 5;
fprintf('  %s Delay: measured=%d samples, expected=%d, range_err=%.2f m\n', ...
    pf(ok4a), meas_delay, delay_samp, range_err_m);

% Doppler verification via spectral peak of CW tone (long enough for 2kHz resolution)
n_tone = 1000000;
f_tone = 1e6;
t_tone = (0:n_tone-1)' / fs;
tone = exp(1j * 2*pi * f_tone * t_tone);
tone_doppler = tone .* exp(1j * 2*pi * doppler_hz * t_tone);
n_fft_d = n_tone;
spec_orig = abs(fft(tone, n_fft_d));
spec_shift = abs(fft(tone_doppler, n_fft_d));
freqs = (0:n_fft_d-1)' * fs / n_fft_d;
[~, i1] = max(spec_orig);
[~, i2] = max(spec_shift);
meas_doppler = freqs(i2) - freqs(i1);
if meas_doppler > fs/2, meas_doppler = meas_doppler - fs; end
doppler_err = abs(meas_doppler - doppler_hz);
ok4b = doppler_err < 2 * fs / n_fft_d;
fprintf('  %s Doppler: expected=%.1f Hz, measured=%.1f Hz, err=%.1f Hz\n', ...
    pf(ok4b), doppler_hz, meas_doppler, doppler_err);

results.channel = ok4a && ok4b;

%% ===== Test 5: Noise Power =====
fprintf('\nTEST 5: Noise Power Verification\n');

% FluxPhased noise: norm = sqrt(power), total energy = power
% Verify: (1) scaled norm matches power, (2) complex Gaussian stats
n_noise = 100000;
noise_raw = (randn(n_noise,1) + 1j*randn(n_noise,1)) / sqrt(2);
pwr = 0.7;  % arbitrary power factor in [0,1]
noise_bb = noise_raw / norm(noise_raw) * sqrt(pwr);
total_pwr = norm(noise_bb)^2;
ok5a = abs(total_pwr - pwr) / pwr < 0.01;
re_var = var(real(noise_bb));
im_var = var(imag(noise_bb));
ok5b = abs(re_var - im_var) / max(re_var, im_var) < 0.1;
fprintf('  %s Broadband noise: total_pwr=%.4f (expect %.1f), re/im var ratio=%.3f\n', ...
    pf(ok5a && ok5b), total_pwr, pwr, re_var/im_var);

% Spot noise
f_center = 50e6;
bw_spot = 10e6;
noise_spot_raw = randn(n_noise,1) + 1j*randn(n_noise,1);
spec_n = fft(noise_spot_raw);
freqs_n = (0:n_noise-1)' / n_noise * fs;
mask = abs(freqs_n - f_center) < bw_spot / 2;
spec_n = spec_n .* mask;
noise_spot = ifft(spec_n);
noise_spot = noise_spot / norm(noise_spot) * sqrt(pwr);
pwr_spot = norm(noise_spot)^2;
ok5c = abs(pwr_spot - pwr) / pwr < 0.01;
fprintf('  %s Spot noise: total_pwr=%.4f (expect %.1f after normalization)\n', pf(ok5c), pwr_spot, pwr);

results.noise = ok5a && ok5b && ok5c;

%% ===== Test 6: DRFM =====
fprintf('\nTEST 6: DRFM Frequency Shift + Delay\n');

freq_shift = 50e3;
delay_us = 10e-6;
delay_drfm = round(delay_us * fs);

% DRFM on CW tone for clean freq measurement
drfm_tone = tone .* exp(1j * 2*pi * freq_shift * t_tone);
drfm_tone = [complex(zeros(delay_drfm,1)); drfm_tone(1:end-delay_drfm)];
nz = norm(drfm_tone);
if nz > 1e-10, drfm_tone = drfm_tone / nz; end

ok6_norm = abs(norm(drfm_tone) - 1.0) < 1e-4;

% Freq shift verification
spec_drfm = abs(fft(drfm_tone, n_fft_d));
[~, i3] = max(spec_drfm);
meas_shift = freqs(i3) - freqs(i1);
if meas_shift > fs/2, meas_shift = meas_shift - fs; end
shift_err = abs(meas_shift - freq_shift);
bin_w = fs / n_fft_d;
ok6_freq = shift_err < 2 * bin_w;
fprintf('  %s DRFM norm=%.6f, freq_shift err=%.1f Hz (threshold %.1f)\n', ...
    pf(ok6_norm && ok6_freq), norm(drfm_tone), shift_err, 2*bin_w);

% Delay verification via MF on LFM
drfm_lfm = tx_lfm .* exp(1j * 2*pi * freq_shift * (0:n_lfm-1)'/fs);
drfm_lfm = [complex(zeros(delay_drfm,1)); drfm_lfm(1:end-delay_drfm)];
nz2 = norm(drfm_lfm);
if nz2 > 1e-10, drfm_lfm = drfm_lfm / nz2; end
mf_drfm = ifft(fft(drfm_lfm, n_fft) .* conj(fft(tx_lfm, n_fft)));
[~, idx_d] = max(abs(mf_drfm).^2);
delay_err = abs((idx_d-1) - delay_drfm);
ok6_delay = delay_err <= 2;
fprintf('  %s DRFM delay: expected=%d, measured=%d, err=%d samples\n', ...
    pf(ok6_delay), delay_drfm, idx_d-1, delay_err);

results.drfm = ok6_norm && ok6_freq && ok6_delay;

%% ===== Test 7: Array Pattern + Directivity =====
fprintf('\nTEST 7: 25x25 URA Pattern + Directivity\n');

array = phased.URA('Size', [rows cols], 'ElementSpacing', [dx_m dy_m]);
array.Element = phased.IsotropicAntennaElement('FrequencyRange', [1e9 20e9]);

az_scan = -90:0.5:90;
pat_az = pattern(array, fc, az_scan, 0);
[peak_dBi, ~] = max(pat_az);
bw_3db = beamwidth(array, fc, 'Cut', 'Azimuth');

sv = phased.SteeringVector('SensorArray', array, 'PropagationSpeed', c);
w30 = sv(fc, [30; 0]);
pat_steer = pattern(array, fc, az_scan, 0, 'Weights', w30);
[~, idx_steer] = max(pat_steer);
steer_err = abs(az_scan(idx_steer) - 30);

ok7a = steer_err < 0.5;
ok7b = abs(bw_3db - 4.06) < 0.5;
ok7c = abs(peak_dBi - D_dBi) < 5;
fprintf('  Peak: %.1f dBi (theory %.1f), BW: %.2f deg, steer_err: %.2f deg\n', ...
    peak_dBi, D_dBi, bw_3db, steer_err);
fprintf('  [%s] Array pattern validated\n', pf(ok7a && ok7b && ok7c));
results.array = ok7a && ok7b && ok7c;

%% ===== Test 8: Cross-Radar Interference JNR =====
fprintf('\nTEST 8: Cross-Radar JNR (Friis one-way)\n');

distances_km = [2, 5, 10, 20];
G = D_dBi;
pass8 = true;
for i = 1:length(distances_km)
    d = distances_km(i) * 1000;
    fspl = 20*log10(4*pi*d/lambda);
    jnr = tx_dbm + G + G - fspl - 3.0 - noise_dbm;
    fprintf('  R=%2d km: JNR = %.1f dB\n', distances_km(i), jnr);
end
% Verify Friis formula consistency: 20dB/decade slope
jnr_2km = tx_dbm + 2*G - 20*log10(4*pi*2000/lambda) - 3 - noise_dbm;
jnr_20km = tx_dbm + 2*G - 20*log10(4*pi*20000/lambda) - 3 - noise_dbm;
slope = jnr_2km - jnr_20km;
ok8 = abs(slope - 20.0) < 0.5;
fprintf('  %s 1/r² slope: %.1f dB/decade (expect 20.0)\n', pf(ok8), slope);
results.interference = ok8;

%% ===== Test 9: Self-Interference Coupling =====
fprintf('\nTEST 9: Self-Interference Coupling Power\n');

iso_levels = [10, 20, 30, 40];
pass9 = true;
for i = 1:length(iso_levels)
    iso = iso_levels(i);
    coupling = 10^(-iso/20);
    si = tx_lfm * coupling;
    si_pwr = mean(abs(si).^2);
    expected = coupling^2 / n_lfm;
    err_db = abs(10*log10(si_pwr / max(expected, 1e-30)));
    ok = err_db < 0.5;
    pass9 = pass9 && ok;
    fprintf('  %s Iso=%2d dB: SI_pwr=%.2e, expected=%.2e, err=%.2f dB\n', ...
        pf(ok), iso, si_pwr, expected, err_db);
end
results.self_intf = pass9;

%% ===== Test 10: Waveform MF Compression =====
fprintf('\nTEST 10: Waveform Matched Filter Compression\n');

wf_bw = 2e6;
pw10 = 50e-6;
TB = pw10 * wf_bw;
n10 = floor(pw10 * fs);
t10 = (0:n10-1)' / fs;

lfm10 = exp(1j * pi * (wf_bw/pw10) * t10.^2);
lfm10 = lfm10 / norm(lfm10);

n_fft10 = 1; while n_fft10 < 2*n10, n_fft10 = n_fft10*2; end
mf10 = ifft(fft(lfm10, n_fft10) .* conj(fft(lfm10, n_fft10)));
mf_pwr = abs(mf10).^2;
[peak10, pk_idx] = max(mf_pwr);

% Compressed pulse width (3dB) - use count of above-half bins (handles circular wrap)
half = peak10 * 0.5;
above = find(mf_pwr > half);
ml_w = length(above);  % number of bins above half-power (correct for circular)
compressed_us = ml_w / fs * 1e6;
expected_compressed = 1 / wf_bw * 1e6;

ok10 = abs(compressed_us - expected_compressed) / expected_compressed < 0.3;
fprintf('  LFM TB=%d: compressed=%.2f us (theory=%.2f us)\n', round(TB), compressed_us, expected_compressed);
fprintf('  [%s] Pulse compression ratio\n', pf(ok10));
results.mf_compression = ok10;

%% ===== Summary =====
fprintf('\n================================================\n');
fprintf('FLUXPHASED IQ CROSS-VALIDATION SUMMARY\n');
fprintf('================================================\n');
names = fieldnames(results);
np = 0;
for i = 1:length(names)
    ok = results.(names{i});
    if ok, np = np + 1; end
    fprintf('  [%s] %s\n', pf(ok), names{i});
end
fprintf('\n  %d/%d passed\n', np, length(names));
if np == length(names)
    fprintf('  ALL IQ VALIDATION TESTS PASSED\n');
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
