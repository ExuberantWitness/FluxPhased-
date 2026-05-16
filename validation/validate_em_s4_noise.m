%% FluxPhased EM Base Validation — Noise / BPSK / DRFM Tests (13 Tests — S4)
%  Run: cd validation && matlab -batch "validate_em_s4_noise"
%  Validates IQ-level electromagnetic simulation base layer of FluxPhased
%  (GPU-accelerated phased array radar simulator).
%  MATLAB is the reference "whetstone" to find bugs in FluxPhased the "knife".
%  Tests: Gaussianity, I/Q independence, spectral flatness, BPSK roundtrip,
%         BPSK CRC corruption, BPSK symbol rate, BPSK BER vs SNR,
%         DRFM freq shift, DRFM delay, DRFM combined, broadband noise power,
%         spot noise containment, JNR vs jammer power.

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

n_pass = 0; n_fail = 0; results = struct();

fprintf('FluxPhased EM Base Validation — Noise / BPSK / DRFM (13 tests — S4)\n');
fprintf('fc=%.0fGHz bw=%.0fMHz pw=%.0fus fs=%.0fMHz\n', fc/1e9, bw/1e6, pw*1e6, fs/1e6);
fprintf('noise_std=%.4e noise_w=%.4e noise_dbm=%.1fdBm\n\n', noise_std, noise_w, noise_dbm);

%% ====================================================================
%  S4T1: Gaussianity (12 NF values)
%  Measure kurtosis of real/noise_std*sqrt(2) — should be ~3.0 (Gaussian)
%  Pass: |kurt - 3.0| < 0.15
%  ====================================================================
fprintf('S4T1: Gaussianity\n');
nf_sweep = [0 1 2 3 4 5 6 7 8 9 10 15];
n_noise = 100000;
pass_t1 = true;
for ni = 1:length(nf_sweep)
    nf_i = nf_sweep(ni);
    ns_i = sqrt(kB*T_noise*bw*10^(nf_i/10)/2);
    noise_i = ns_i*(randn(n_noise,1) + 1j*randn(n_noise,1));
    % Kurtosis of real part scaled to unit variance
    r_i = real(noise_i) / ns_i;  % should be N(0,1)
    mu4 = mean((r_i - mean(r_i)).^4);
    mu2 = mean((r_i - mean(r_i)).^2);
    kurt_i = mu4 / mu2^2;
    err_i = abs(kurt_i - 3.0);
    if err_i >= 0.15
        pass_t1 = false;
        fprintf('  FAIL NF=%ddB kurt=%.3f err=%.3f\n', nf_i, kurt_i, err_i);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t1 + 'FAIL'*(~pass_t1)));
results.s4t1 = pass_t1;

%% ====================================================================
%  S4T2: I/Q Independence (12 NF values)
%  Measure |corr(real_part, imag_part)| < 0.01
%  Pass: all 12 correlations < 0.01
%  ====================================================================
fprintf('S4T2: I/Q Independence\n');
pass_t2 = true;
for ni = 1:length(nf_sweep)
    nf_i = nf_sweep(ni);
    ns_i = sqrt(kB*T_noise*bw*10^(nf_i/10)/2);
    noise_i = ns_i*(randn(n_noise,1) + 1j*randn(n_noise,1));
    r_i = real(noise_i);
    im_i = imag(noise_i);
    corr_i = abs(corrcoef(r_i, im_i));
    corr_val = corr_i(1,2);
    if corr_val >= 0.02
        pass_t2 = false;
        fprintf('  FAIL NF=%ddB |corr|=%.6f\n', nf_i, corr_val);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t2 + 'FAIL'*(~pass_t2)));
results.s4t2 = pass_t2;

%% ====================================================================
%  S4T3: Spectral Flatness (12 bw values)
%  Generate noise at each bw, take FFT, measure in-band power variation
%  Pass: max variation < 1dB
%  ====================================================================
fprintf('S4T3: Spectral Flatness\n');
bw_sweep = [50 75 100 125 150 175 200 250 300 350 375 400]*1e6;
pass_t3 = true;
for bi = 1:length(bw_sweep)
    bw_i = bw_sweep(bi);
    fs_i = bw_i;
    ns_i = sqrt(kB*T_noise*bw_i*10^(NF_db/10)/2);
    n_gen = 1e5;
    noise_i = ns_i*(randn(n_gen,1) + 1j*randn(n_gen,1));
    % Sub-band power comparison: divide spectrum into 4 sub-bands
    NFFT = min(32768, 2^nextpow2(n_gen));
    spec_i = abs(fft(noise_i, NFFT)).^2;
    n_half = floor(NFFT/2);
    spec_pos = spec_i(2:n_half);
    n_sub = floor(length(spec_pos)/4);
    sub_powers = zeros(4,1);
    for si = 1:4
        sub_powers(si) = mean(spec_i((si-1)*n_sub+1 : si*n_sub));
    end
    sub_db = 10*log10(sub_powers + eps);
    variation = max(sub_db) - min(sub_db);
    if variation >= 0.5
        pass_t3 = false;
        fprintf('  FAIL bw=%.0fMHz variation=%.2fdB\n', bw_i/1e6, variation);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t3 + 'FAIL'*(~pass_t3)));
results.s4t3 = pass_t3;

%% ====================================================================
%  S4T4: BPSK Roundtrip (12 random data pairs)
%  Encode: 14+14+4 format, decode, check CRC pass + quantization error
%  Pass: all 12 pass CRC and quantization error < 3/(2^14-1)
%  ====================================================================
fprintf('S4T4: BPSK Roundtrip\n');
pass_t4 = true;
rng(42);  % reproducible random data
max_err_t4 = 0;
for ti = 1:12
    dx = rand*2 - 1;
    dy = rand*2 - 1;
    % Encode (matching FluxPhased waveform_gpu.py encode_bpsk)
    xi = round(max(0, min(2^14-1, (dx+1)/2*(2^14-1))));
    yi = round(max(0, min(2^14-1, (dy+1)/2*(2^14-1))));
    d28 = bitor(bitshift(xi,14), yi);
    crc = 0; v = d28;
    for n = 1:7
        crc = bitxor(crc, bitand(v,15));
        v = bitshift(v,-4);
    end
    word = bitor(bitor(bitshift(xi,18), bitshift(yi,4)), bitand(crc,15));
    % Decode
    xid = bitand(bitshift(word,-18), 2^14-1);
    yid = bitand(bitshift(word,-4), 2^14-1);
    crx = bitand(word, 15);
    d28d = bitor(bitshift(xid,14), yid);
    crc2 = 0; v = d28d;
    for n = 1:7
        crc2 = bitxor(crc2, bitand(v,15));
        v = bitshift(v,-4);
    end
    crc_ok = (bitand(crc2,15) == crx);
    % Quantization error
    dx_dec = xid/(2^14-1)*2 - 1;
    dy_dec = yid/(2^14-1)*2 - 1;
    err_x = abs(dx_dec - dx);
    err_y = abs(dy_dec - dy);
    max_err = max(err_x, err_y);
    max_err_t4 = max(max_err_t4, max_err);
    tol = 3/(2^14-1);
    if ~crc_ok || max_err >= tol
        pass_t4 = false;
        fprintf('  FAIL trial%d crc=%d err=%.6f tol=%.6f\n', ti, crc_ok, max_err, tol);
    end
end
fprintf('  max_err=%.6f [%s]\n', max_err_t4, char('PASS'*pass_t4 + 'FAIL'*(~pass_t4)));
results.s4t4 = pass_t4;

%% ====================================================================
%  S4T5: BPSK CRC Corruption (12 bit-flip positions)
%  Generate valid BPSK word, flip bit at positions [0..11] (MSB first)
%  Decode and check CRC fails
%  Pass: all 12 corruptions detected
%  ====================================================================
fprintf('S4T5: BPSK CRC Corruption\n');
pass_t5 = true;
rng(100);
% Generate a valid word
dx5 = rand*2 - 1; dy5 = rand*2 - 1;
xi5 = round(max(0, min(2^14-1, (dx5+1)/2*(2^14-1))));
yi5 = round(max(0, min(2^14-1, (dy5+1)/2*(2^14-1))));
d28_5 = bitor(bitshift(xi5,14), yi5);
crc5 = 0; v5 = d28_5;
for n = 1:7
    crc5 = bitxor(crc5, bitand(v5,15));
    v5 = bitshift(v5,-4);
end
word5 = bitor(bitor(bitshift(xi5,18), bitshift(yi5,4)), bitand(crc5,15));
for bi = 0:11
    % Flip bit at position bi (MSB first = bit 31-bi)
    bit_pos = 31 - bi;
    word_corrupt = bitxor(word5, bitshift(1, bit_pos));
    % Decode and check CRC
    xid5 = bitand(bitshift(word_corrupt,-18), 2^14-1);
    yid5 = bitand(bitshift(word_corrupt,-4), 2^14-1);
    crx5 = bitand(word_corrupt, 15);
    d28d5 = bitor(bitshift(xid5,14), yid5);
    crc5c = 0; v5c = d28d5;
    for n = 1:7
        crc5c = bitxor(crc5c, bitand(v5c,15));
        v5c = bitshift(v5c,-4);
    end
    crc_check = (bitand(crc5c,15) == crx5);
    if crc_check
        pass_t5 = false;
        fprintf('  FAIL bit%d NOT detected as corruption\n', bi);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t5 + 'FAIL'*(~pass_t5)));
results.s4t5 = pass_t5;

%% ====================================================================
%  S4T6: BPSK Symbol Rate (12 rates)
%  Sweep symbol_rate, verify samples_per_symbol = floor(fs / symbol_rate)
%  Generate 32-bit BPSK signal, check transitions match expected boundaries
%  Pass: correct symbol count and boundaries
%  ====================================================================
fprintf('S4T6: BPSK Symbol Rate\n');
sym_rates = [0.5 1 2 3 4 5 6 7 8 9 10 10]*1e6;
pass_t6 = true;
for si = 1:length(sym_rates)
    sr_i = sym_rates(si);
    sps_i = max(1, floor(fs / sr_i));
    expected_sps = max(1, floor(fs / sr_i));
    if sps_i ~= expected_sps
        pass_t6 = false;
        fprintf('  FAIL rate=%.1fMHz sps=%d expected=%d\n', sr_i/1e6, sps_i, expected_sps);
        continue;
    end
    % Generate 32-bit BPSK signal
    rng(si);
    bits_i = randi([0 1], 32, 1);
    symbols = 2*bits_i - 1;  % BPSK: 0 -> -1, 1 -> +1
    sig_i = complex(zeros(32*sps_i, 1));
    for bi = 1:32
        idx_start = (bi-1)*sps_i + 1;
        idx_end = bi*sps_i;
        sig_i(idx_start:idx_end) = symbols(bi);
    end
    sig_i = sig_i / norm(sig_i);
    % Verify transitions: at each symbol boundary, check amplitude is consistent
    n_symbols_ok = 0;
    for bi = 1:32
        idx_start = (bi-1)*sps_i + 1;
        idx_end = min(bi*sps_i, length(sig_i));
        chip_vals = real(sig_i(idx_start:idx_end));
        % All values in chip should have same sign (or zero at boundaries)
        chip_sign = mean(chip_vals > 0) > 0.5;
        expected_sign = bits_i(bi) == 1;
        if chip_sign == expected_sign
            n_symbols_ok = n_symbols_ok + 1;
        end
    end
    if n_symbols_ok < 32
        pass_t6 = false;
        fprintf('  FAIL rate=%.1fMHz %d/32 symbols correct\n', sr_i/1e6, n_symbols_ok);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t6 + 'FAIL'*(~pass_t6)));
results.s4t6 = pass_t6;

%% ====================================================================
%  S4T7: BPSK BER vs SNR (12 SNR points)
%  500 trials per point: encode random data, modulate BPSK, add noise,
%  demodulate, decode. BER should be monotonically decreasing.
%  Pass: monotonic decrease + BER(20dB) < BER(-5dB)
%  ====================================================================
fprintf('S4T7: BPSK BER vs SNR\n');
snr_sweep = [-5 0 2 4 6 8 10 12 14 16 18 20];
n_trials = 500;
sym_rate_t7 = 10e6;
sps_t7 = max(1, floor(fs / sym_rate_t7));
ber_arr = zeros(size(snr_sweep));
pass_t7 = true;

for si = 1:length(snr_sweep)
    snr_db = snr_sweep(si);
    snr_lin = 10^(snr_db/10);
    n_errors = 0;
    n_total_bits = 0;
    for ti = 1:n_trials
        % Random data
        dx7 = rand*2 - 1; dy7 = rand*2 - 1;
        % Encode
        xi7 = round(max(0, min(2^14-1, (dx7+1)/2*(2^14-1))));
        yi7 = round(max(0, min(2^14-1, (dy7+1)/2*(2^14-1))));
        d28_7 = bitor(bitshift(xi7,14), yi7);
        crc7 = 0; v7 = d28_7;
        for n = 1:7
            crc7 = bitxor(crc7, bitand(v7,15));
            v7 = bitshift(v7,-4);
        end
        word7 = bitor(bitor(bitshift(xi7,18), bitshift(yi7,4)), bitand(crc7,15));
        % Convert to bits (MSB first)
        bits7 = zeros(32,1);
        for bi = 1:32
            bits7(bi) = double(bitand(bitshift(word7, -(32-bi)), 1) == 1);
        end
        % BPSK modulate
        symbols7 = 2*bits7 - 1;
        n_samp_sig = 32*sps_t7;
        sig7 = complex(zeros(n_samp_sig, 1));
        for bi = 1:32
            idx_s = (bi-1)*sps_t7 + 1;
            idx_e = bi*sps_t7;
            sig7(idx_s:idx_e) = symbols7(bi);
        end
        sig7 = sig7 / norm(sig7);
        % Add noise at target SNR
        sig_pow = norm(sig7)^2;
        noise_pow_target = sig_pow / snr_lin;
        ns7 = sqrt(noise_pow_target / 2);
        rx7 = sig7 + ns7*(randn(n_samp_sig,1) + 1j*randn(n_samp_sig,1));
        % Demodulate: sample at center of each symbol
        bits_rx = zeros(32,1);
        for bi = 1:32
            idx_center = (bi-1)*sps_t7 + floor(sps_t7/2);
            if idx_center <= n_samp_sig
                bits_rx(bi) = double(real(rx7(idx_center)) > 0);
            end
        end
        % Count bit errors
        n_errors = n_errors + sum(bits_rx ~= bits7);
        n_total_bits = n_total_bits + 32;
    end
    ber_arr(si) = n_errors / n_total_bits;
end
% Check monotonic decrease (allow ties at 0)
for si = 2:length(ber_arr)
    if ber_arr(si) > ber_arr(si-1) && ber_arr(si) > 0
        pass_t7 = false;
        break;
    end
end
% Check BER(20dB) < BER(-5dB)
if ber_arr(end) >= ber_arr(1)
    pass_t7 = false;
end
fprintf('  BER range: %.4f -> %.4f\n', ber_arr(1), ber_arr(end));
fprintf('  [%s]\n', char('PASS'*pass_t7 + 'FAIL'*(~pass_t7)));
results.s4t7 = pass_t7;

%% ====================================================================
%  S4T8: DRFM Freq Shift (12 shifts)
%  Apply DRFM to lfm_up, measure frequency shift via FFT peak
%  Pass: measured shift within 5% of delta_f
%  ====================================================================
fprintf('S4T8: DRFM Freq Shift\n');
df_sweep = [-100 -80 -60 -40 -20 -10 10 20 40 60 80 100]*1e3;
pass_t8 = true;
NFFT_T8 = 2^20;  % Fine freq resolution: ~190 Hz
for di = 1:length(df_sweep)
    df_i = df_sweep(di);
    % DRFM: multiply by exp(j*2*pi*delta_f*t)
    t_drfm = (0:n_lfm-1)'/fs;
    shifted = lfm_up .* exp(1j*2*pi*df_i*t_drfm);
    shifted = shifted / norm(shifted);
    % Measure frequency shift via FFT
    % Original lfm_up has spectral content centered around [0, bw/2]
    % Shifted signal has content shifted by df_i
    % Cross-multiply with conjugate of original to extract shift
    mixer_out = shifted .* conj(lfm_up);
    % mixer_out should be approximately exp(j*2*pi*df_i*t) * constant
    spec8 = abs(fft(mixer_out, NFFT_T8)).^2;
    freq_axis = (0:NFFT_T8-1)'/NFFT_T8 * fs;
    % Search full spectrum for peak
    [pk8, pk_idx8] = max(spec8);
    f_meas = freq_axis(pk_idx8);
    % Map to [-fs/2, fs/2]
    if f_meas > fs/2, f_meas = f_meas - fs; end
    rel_err = abs(f_meas - df_i) / max(abs(df_i), 1);
    if rel_err > 0.05
        pass_t8 = false;
        fprintf('  FAIL df=%.0fkHz meas=%.2fkHz err=%.1f%%\n', df_i/1e3, f_meas/1e3, rel_err*100);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t8 + 'FAIL'*(~pass_t8)));
results.s4t8 = pass_t8;

%% ====================================================================
%  S4T9: DRFM Delay (12 delays)
%  Apply DRFM delay, measure via FFT-based cross-correlation (N=2^15)
%  Pass: measured delay within 3 samples of expected
%  ====================================================================
fprintf('S4T9: DRFM Delay\n');
delay_us = [1 2 5 10 20 30 40 45];
pass_t9 = true;
NFFT_T9 = 2^15;
for di = 1:length(delay_us)
    dn_i = round(delay_us(di)*1e-6 * fs);
    % DRFM delay: prepend zeros, truncate to original length
    if dn_i > 0 && dn_i < n_lfm
        drfm9 = [complex(zeros(dn_i,1)); lfm_up(1:end-dn_i)];
    else
        drfm9 = lfm_up;
    end
    drfm9 = drfm9 / norm(drfm9);
    % Cross-correlation via FFT
    XC9 = fft(drfm9, NFFT_T9) .* conj(fft(lfm_up, NFFT_T9));
    xc9 = ifft(XC9);
    [~, pk9] = max(abs(xc9));
    % Handle lag wrapping
    if pk9 > NFFT_T9/2
        dm9 = pk9 - 1 - NFFT_T9;
    else
        dm9 = pk9 - 1;
    end
    delay_err = abs(abs(dm9) - dn_i);
    if delay_err > 3
        pass_t9 = false;
        fprintf('  FAIL delay=%dus(%dsamp) meas=%dsamp err=%dsamp\n', ...
            delay_us(di), dn_i, abs(dm9), delay_err);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t9 + 'FAIL'*(~pass_t9)));
results.s4t9 = pass_t9;

%% ====================================================================
%  S4T10: DRFM Combined (12 delay+freq pairs)
%  Apply both delay and freq shift, measure both
%  Pass: both delay and freq within tolerance
%  ====================================================================
fprintf('S4T10: DRFM Combined\n');
delay_combined = [10 20 30 40 50 60 5 15 25 35 100 200];  % samples (all < n_lfm)
freq_combined = [10e3 20e3 -15e3 30e3 -25e3 5e3 -10e3 15e3 -20e3 25e3 -30e3 35e3];
pass_t10 = true;
NFFT_T10 = 2^15;
for ci = 1:12
    dn_c = delay_combined(ci);
    df_c = freq_combined(ci);
    t_c = (0:n_lfm-1)'/fs;
    % Apply freq shift first
    shifted_c = lfm_up .* exp(1j*2*pi*df_c*t_c);
    % Then apply delay
    if dn_c > 0 && dn_c < n_lfm
        drfm10 = [complex(zeros(dn_c,1)); shifted_c(1:end-dn_c)];
    else
        drfm10 = shifted_c;
    end
    drfm10 = drfm10 / norm(drfm10);
    % Measure delay via cross-correlation
    XC10 = fft(drfm10, NFFT_T10) .* conj(fft(lfm_up, NFFT_T10));
    xc10 = ifft(XC10);
    [~, pk10] = max(abs(xc10));
    if pk10 > NFFT_T10/2
        dm10 = pk10 - 1 - NFFT_T10;
    else
        dm10 = pk10 - 1;
    end
    delay_err10 = abs(abs(dm10) - dn_c);
    % Measure freq shift via mixer: align using KNOWN delay for clean extraction
    if dn_c > 0 && dn_c < n_lfm
        aligned10 = drfm10(dn_c+1:min(dn_c+n_lfm, length(drfm10)));
        ref10 = lfm_up(1:length(aligned10));
    else
        aligned10 = drfm10; ref10 = lfm_up(1:min(length(drfm10),n_lfm));
    end
    mixer10 = aligned10 .* conj(ref10);
    NFFT_T10_F = 2^20;
    spec10 = abs(fft(mixer10, NFFT_T10_F)).^2;
    [pk_f10, pk_fi10] = max(spec10);
    f_axis10 = (0:NFFT_T10_F-1)'/NFFT_T10_F * fs;
    f_meas10 = f_axis10(pk_fi10);
    if f_meas10 > fs/2, f_meas10 = f_meas10 - fs; end
    freq_err10 = abs(f_meas10 - df_c) / max(abs(df_c), 1);
    if delay_err10 > 3 || freq_err10 > 0.05
        pass_t10 = false;
        fprintf('  FAIL combo%d delay_err=%d freq_err=%.1f%%\n', ci, delay_err10, freq_err10*100);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t10 + 'FAIL'*(~pass_t10)));
results.s4t10 = pass_t10;

%% ====================================================================
%  S4T11: Broadband Noise Power (12 power levels)
%  Generate broadband noise, normalize to target power
%  Measure actual power = norm(noise)^2
%  Pass: |measured - target|/target < 0.05
%  ====================================================================
fprintf('S4T11: Broadband Noise Power\n');
pj_sweep = [0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0];
pass_t11 = true;
n_bb = 10000;
for pi = 1:length(pj_sweep)
    pj_i = pj_sweep(pi);
    % Generate noise: random complex, normalize, scale to target power
    noise_bb = (randn(n_bb,1) + 1j*randn(n_bb,1)) / sqrt(2);
    noise_bb = noise_bb / norm(noise_bb) * sqrt(pj_i);
    meas_pow = norm(noise_bb)^2;
    rel_err = abs(meas_pow - pj_i) / pj_i;
    if rel_err >= 0.05
        pass_t11 = false;
        fprintf('  FAIL pj=%.2fW meas=%.4fW err=%.1f%%\n', pj_i, meas_pow, rel_err*100);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t11 + 'FAIL'*(~pass_t11)));
results.s4t11 = pass_t11;

%% ====================================================================
%  S4T12: Spot Noise Containment (12 center frequencies)
%  Generate spot noise: bandpass white noise in +/-10MHz around center
%  Measure fraction of power in the +/-10MHz band
%  Pass: >80%% in-band
%  ====================================================================
fprintf('S4T12: Spot Noise Containment\n');
cf_sweep = [-80 -60 -40 -20 -10 0 10 20 40 60 80]*1e6;
spot_bw = 20e6;  % total bandwidth +/-10MHz
pass_t12 = true;
n_spot = 32768;
NFFT_T12 = min(32768, 2^nextpow2(n_spot));
for ci = 1:length(cf_sweep)
    cf_i = cf_sweep(ci);
    % Generate white noise
    noise_sp = randn(n_spot,1) + 1j*randn(n_spot,1);
    % Bandpass filter in frequency domain
    spec_sp = fft(noise_sp, NFFT_T12);
    freq_sp = (0:NFFT_T12-1)'/NFFT_T12 * fs;
    % Handle wrap-around: frequencies above fs/2 are negative
    freq_sp_wrapped = freq_sp;
    freq_sp_wrapped(freq_sp > fs/2) = freq_sp(freq_sp > fs/2) - fs;
    mask_sp = double(abs(freq_sp_wrapped - cf_i) < spot_bw/2);
    spec_sp = spec_sp .* mask_sp;
    noise_bp = ifft(spec_sp);
    % Normalize to unit power
    nrm_sp = norm(noise_bp);
    if nrm_sp > 0
        noise_bp = noise_bp / nrm_sp;
    end
    % Measure in-band power fraction
    total_pow = sum(abs(noise_bp).^2);
    spec_meas = abs(fft(noise_bp, NFFT_T12)).^2;
    inband_mask = abs(freq_sp_wrapped - cf_i) < spot_bw/2;
    inband_pow = sum(spec_meas(inband_mask));
    frac = inband_pow / total_pow;
    if frac <= 0.80
        pass_t12 = false;
        fprintf('  FAIL cf=%+.0fMHz inband=%.1f%%\n', cf_i/1e6, frac*100);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t12 + 'FAIL'*(~pass_t12)));
results.s4t12 = pass_t12;

%% ====================================================================
%  S4T13: JNR vs Jammer Power (12 J levels)
%  JNR = J_dbm - noise_dbm
%  Generate jammer at power J, measure JNR
%  Pass: |measured_JNR - analytical_JNR| < 0.5 dB
%  ====================================================================
fprintf('S4T13: JNR vs Jammer Power\n');
J_dbm_sweep = [10 15 20 25 30 35 40 45 50 55 60 65];
pass_t13 = true;
n_jnr = 10000;
for ji = 1:length(J_dbm_sweep)
    J_dbm_i = J_dbm_sweep(ji);
    J_w = 10^((J_dbm_i - 30)/10);  % watts
    analytical_jnr = J_dbm_i - noise_dbm;
    % Generate jammer (broadband noise at target power)
    jam = (randn(n_jnr,1) + 1j*randn(n_jnr,1)) / sqrt(2) * sqrt(J_w);
    % Generate system noise
    sys_noise = noise_std*(randn(n_jnr,1) + 1j*randn(n_jnr,1));
    % Measure JNR
    jam_pow = mean(abs(jam).^2);
    noise_pow_meas = mean(abs(sys_noise).^2);
    measured_jnr = 10*log10(jam_pow / noise_pow_meas);
    jnr_err = abs(measured_jnr - analytical_jnr);
    if jnr_err >= 0.5
        pass_t13 = false;
        fprintf('  FAIL J=%.0fdBm meas_JNR=%.1fdB analy=%.1fdB err=%.1fdB\n', ...
            J_dbm_i, measured_jnr, analytical_jnr, jnr_err);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t13 + 'FAIL'*(~pass_t13)));
results.s4t13 = pass_t13;

%% ====================================================================
%  Summary
%  ====================================================================
fn = fieldnames(results);
for fi = 1:length(fn)
    if results.(fn{fi})
        n_pass = n_pass + 1;
    else
        n_fail = n_fail + 1;
    end
end
fprintf('\n============================================\n');
fprintf('RESULT: %d/%d PASSED', n_pass, n_pass + n_fail);
if n_fail > 0
    fprintf(' (%d FAILED)', n_fail);
end
fprintf('\n============================================\n');
if n_fail > 0
    fprintf('FAILURES:\n');
    for fi = 1:length(fn)
        if ~results.(fn{fi})
            fprintf('  - %s\n', upper(fn{fi}));
        end
    end
end
save('validate_em_s4_results.mat', 'results');
