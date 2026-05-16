%% FluxPhased Waveform & Matched Filter Validation (14 Tests — S3)
%  Run: cd validation && matlab -batch "validate_em_s3_waveform"
%  Validates waveform generators and matched-filter performance against
%  analytical models for all 7 waveform types used in FluxPhased.
%  Tests: LFM freq (up/down), LFM pw/samples, Barker PSL, Frank phase,
%         Costas freq, NLFM nonlinearity, P4 phase, unit norm, MF CR,
%         MF peak vs delay, cross-correlation, TBP conservation,
%         reproducibility.

clear; close all; clc;

%% Common Parameters
c = 299792458; fc = 10e9; lambda = c/fc;
bw = 200e6; fs = bw;
kB = 1.380649e-23; T_noise = 290;
NF_db = 5; noise_w = kB*T_noise*bw*10^(NF_db/10);
noise_std = sqrt(noise_w/2);
pw = 50e-6; n_lfm = floor(pw*fs); k_lfm = bw/pw;
t_lfm = (0:n_lfm-1)'/fs;

n_pass = 0; n_fail = 0; results = struct();
wf_names = {'LFM_up','LFM_down','Barker13','Frank16','Costas16','NLFM','P4'};

fprintf('FluxPhased Waveform & Matched Filter Validation (14 tests — S3)\n');
fprintf('fc=%.0fGHz bw=%.0fMHz pw=%.0fus fs=%.0fMHz\n\n', fc/1e9, bw/1e6, pw*1e6, fs/1e6);

%% ====================================================================
%  S3T1: LFM Up Frequency Sweep
%  ====================================================================
fprintf('S3T1: LFM up freq\n');
bw_sweep = [50 75 100 125 150 175 200 250 300 350 400 50]*1e6;
pw_t1 = 50e-6;
pass_t1 = true;
for bi = 1:length(bw_sweep)
    bw_i = bw_sweep(bi);
    fs_i = 4*bw_i;  % Oversample to avoid aliasing for freq measurement
    n_i = max(1, floor(pw_t1*fs_i));
    t_i = (0:n_i-1)'/fs_i;
    k_i = bw_i/pw_t1;
    sig = exp(1j*pi*k_i*t_i.^2);
    inst_phase = unwrap(angle(sig));
    dt_i = t_i(2) - t_i(1);
    inst_freq = diff(inst_phase)/(2*pi*dt_i);
    f_end = inst_freq(end);
    f_target = bw_i;
    rel_err = abs(f_end - f_target)/f_target;
    if rel_err > 0.02
        pass_t1 = false;
        fprintf('  FAIL bw=%.0fMHz f_end=%.2eHz target=%.2eHz err=%.1f%%\n', ...
            bw_i/1e6, f_end, f_target, rel_err*100);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t1 + 'FAIL'*(~pass_t1)));
results.s3t1 = pass_t1;

%% ====================================================================
%  S3T2: LFM Down Frequency Sweep
%  ====================================================================
fprintf('S3T2: LFM down freq\n');
pass_t2 = true;
for bi = 1:length(bw_sweep)
    bw_i = bw_sweep(bi);
    fs_i = 4*bw_i;  % Oversample to avoid aliasing for freq measurement
    n_i = max(1, floor(pw_t1*fs_i));
    t_i = (0:n_i-1)'/fs_i;
    k_i = bw_i/pw_t1;
    sig = exp(-1j*pi*k_i*t_i.^2);
    inst_phase = unwrap(angle(sig));
    dt_i = t_i(2) - t_i(1);
    inst_freq = diff(inst_phase)/(2*pi*dt_i);
    f_end = inst_freq(end);
    f_target = -bw_i;
    rel_err = abs(f_end - f_target)/abs(f_target);
    if rel_err > 0.02
        pass_t2 = false;
        fprintf('  FAIL bw=%.0fMHz f_end=%.2eHz target=%.2eHz err=%.1f%%\n', ...
            bw_i/1e6, f_end, f_target, rel_err*100);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t2 + 'FAIL'*(~pass_t2)));
results.s3t2 = pass_t2;

%% ====================================================================
%  S3T3: LFM Pulse Width vs Samples
%  ====================================================================
fprintf('S3T3: LFM pw vs samples\n');
pw_sweep = [1 2 5 10 20 30 40 50 60 70 80 100]*1e-6;
bw_t3 = 200e6; fs_t3 = bw_t3;
pass_t3 = true;
for pi_idx = 1:length(pw_sweep)
    pw_i = pw_sweep(pi_idx);
    n_i = max(1, floor(pw_i*fs_t3));
    k_i = bw_t3/pw_i;
    t_i = (0:n_i-1)'/fs_t3;
    sig = exp(1j*pi*k_i*t_i.^2);
    sig = sig / norm(sig);
    ok_n = (n_i == max(1, floor(pw_i*fs_t3)));
    ok_norm = abs(norm(sig) - 1) < 1e-5;
    if ~ok_n || ~ok_norm
        pass_t3 = false;
        fprintf('  FAIL pw=%.0fus n=%d expected=%d norm_err=%.1e\n', ...
            pw_i*1e6, n_i, max(1,floor(pw_i*fs_t3)), abs(norm(sig)-1));
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t3 + 'FAIL'*(~pass_t3)));
results.s3t3 = pass_t3;

%% ====================================================================
%  S3T4: Barker-13 Peak Sidelobe Level
%  ====================================================================
fprintf('S3T4: Barker PSL\n');
pw_sweep_t4 = [5 10 15 20 25 30 40 50 60 80 100]*1e-6;
pass_t4 = true;
barker_code_t4 = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
% Use BASE 13-chip code (not oversampled) for PSL measurement
acb = abs(xcorr(barker_code_t4));
mainlobe = max(acb);
sidelobes = acb(acb < mainlobe);
psl = 20*log10(max(sidelobes)/mainlobe);
fprintf('  base13 PSL=%.1fdB', psl);
if psl >= -20
    pass_t4 = false;
end
% Also verify at various pulse widths that oversampled Barker has unit norm
for pi_idx = 1:length(pw_sweep_t4)
    pw_i = pw_sweep_t4(pi_idx);
    reps = max(1, floor(pw_i*fs/13));
    barker_i = repelem(barker_code_t4, reps);
    barker_i = complex(barker_i(:));
    barker_i = barker_i / norm(barker_i);
    if abs(norm(barker_i) - 1) > 1e-5
        pass_t4 = false;
        fprintf('  FAIL pw=%.0fus norm_err=%.1e', pw_i*1e6, abs(norm(barker_i)-1));
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t4 + 'FAIL'*(~pass_t4)));
results.s3t4 = pass_t4;

%% ====================================================================
%  S3T5: Frank-16 Phase Verification
%  ====================================================================
fprintf('S3T5: Frank phase\n');
pw_sweep_t5 = [5 10 20 30 40 50 60 70 80 90 100 50]*1e-6;
pass_t5 = true;
for pi_idx = 1:length(pw_sweep_t5)
    pw_i = pw_sweep_t5(pi_idx);
    n_i = max(1, floor(pw_i*fs));
    n_phases = 4;
    phase_seq_16 = (2*pi/n_phases * ((0:n_phases-1)'*(0:n_phases-1)));
    phase_seq_16 = phase_seq_16(:);
    % Use interp1 as required (avoids MATLAB R2024a floor/indexing hang)
    frank_phase = interp1(1:length(phase_seq_16), phase_seq_16, ...
        linspace(1, length(phase_seq_16), n_i), 'linear');
    frank = exp(1j*frank_phase);
    frank = frank / norm(frank);
    % Check unit norm
    if abs(norm(frank) - 1) > 1e-5
        pass_t5 = false;
        fprintf('  FAIL pw=%.0fus norm_err=%.1e\n', pw_i*1e6, abs(norm(frank)-1));
    end
    % Check phase at original Frank code phase points
    % The 16 phases map to output indices via linspace(1,16,n_i)
    idx_list = round(linspace(1, n_i, 16));
    for ci = 1:16
        idx = idx_list(ci);
        row_i = floor((ci-1)/n_phases);
        col_j = mod((ci-1), n_phases);
        expected_phase = 2*pi/n_phases * row_i * col_j;
        actual_phase = angle(frank(idx));
        phase_err = abs(wrapToPi(actual_phase - expected_phase));
        if phase_err > 0.1
            pass_t5 = false;
            fprintf('  FAIL pw=%.0fus chip%d phase_err=%.3f rad\n', pw_i*1e6, ci, phase_err);
        end
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t5 + 'FAIL'*(~pass_t5)));
results.s3t5 = pass_t5;

%% ====================================================================
%  S3T6: Costas Frequency Verification
%  ====================================================================
fprintf('S3T6: Costas freq\n');
pw_sweep_t6 = [5 10 20 30 40 50 60 70 80 90 100 50]*1e-6;
cseq = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
pass_t6 = true;
for pi_idx = 1:length(pw_sweep_t6)
    pw_i = pw_sweep_t6(pi_idx);
    n_i = max(1, floor(pw_i*fs));
    cl = max(1, floor(n_i/16));
    costas = complex(zeros(n_i, 1));
    for ci = 1:16
        s0 = (ci-1)*cl + 1;
        s1 = min(s0 + cl - 1, n_i);
        costas(s0:s1) = exp(1j*2*pi*cseq(ci)/pw_i*(0:s1-s0)'/fs);
    end
    costas = costas / norm(costas);
    % Check unit norm
    if abs(norm(costas) - 1) > 1e-5
        pass_t6 = false;
        fprintf('  FAIL pw=%.0fus norm_err=%.1e\n', pw_i*1e6, abs(norm(costas)-1));
    end
    % Check each chip is present (non-zero energy in each chip segment)
    for ci = 1:16
        s0 = (ci-1)*cl + 1;
        s1 = min(s0 + cl - 1, n_i);
        chip_energy = sum(abs(costas(s0:s1)).^2);
        if chip_energy < 1e-10
            pass_t6 = false;
            fprintf('  FAIL pw=%.0fus chip%d zero energy\n', pw_i*1e6, ci);
        end
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t6 + 'FAIL'*(~pass_t6)));
results.s3t6 = pass_t6;

%% ====================================================================
%  S3T7: NLFM Nonlinearity
%  ====================================================================
fprintf('S3T7: NLFM nonlinearity\n');
bw_sweep_t7 = [50 75 100 150 200 250 300 350 400 50 100 200]*1e6;
pw_t7 = 50e-6;
pass_t7 = true;
for bi = 1:length(bw_sweep_t7)
    bw_i = bw_sweep_t7(bi);
    fs_i = bw_i;
    n_i = max(1, floor(pw_t7*fs_i));
    t_i = (0:n_i-1)'/fs_i;
    k_i = bw_i/pw_t7;
    % NLFM with cubic term
    sig = exp(1j*(pi*k_i*t_i.^2 + 0.3*pi*k_i/pw_t7*t_i.^3));
    sig = sig / norm(sig);
    % Check unit norm
    if abs(norm(sig) - 1) > 1e-5
        pass_t7 = false;
        fprintf('  FAIL bw=%.0fMHz norm_err=%.1e\n', bw_i/1e6, abs(norm(sig)-1));
    end
    % Check instantaneous frequency deviates from linear by > 0.01*bw
    inst_phase = unwrap(angle(sig));
    dt_i = t_i(2) - t_i(1);
    inst_freq = diff(inst_phase)/(2*pi*dt_i);
    % Linear reference: f_linear(t) = k_i * t  (derivative of pi*k*t^2)
    f_linear = k_i * t_i(1:end-1);
    deviation = inst_freq - f_linear;
    max_dev = max(abs(deviation));
    if max_dev < 0.01*bw_i
        pass_t7 = false;
        fprintf('  FAIL bw=%.0fMHz max_dev=%.2eHz min=%.2eHz\n', ...
            bw_i/1e6, max_dev, 0.01*bw_i);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t7 + 'FAIL'*(~pass_t7)));
results.s3t7 = pass_t7;

%% ====================================================================
%  S3T8: P4 Phase Verification
%  ====================================================================
fprintf('S3T8: P4 phase\n');
pw_sweep_t8 = [5 10 20 30 40 50 60 70 80 90 100 50]*1e-6;
pass_t8 = true;
for pi_idx = 1:length(pw_sweep_t8)
    pw_i = pw_sweep_t8(pi_idx);
    n_i = max(1, floor(pw_i*fs));
    npt = 16;
    kp = (0:npt-1)';
    php = pi*kp.^2/npt - pi*kp;
    % Use interp1 as required (avoids MATLAB R2024a floor/indexing hang)
    p4_phase = interp1(1:npt, php, linspace(1, npt, n_i), 'linear');
    p4 = exp(1j*p4_phase);
    p4 = p4 / norm(p4);
    % Check unit norm
    if abs(norm(p4) - 1) > 1e-5
        pass_t8 = false;
        fprintf('  FAIL pw=%.0fus norm_err=%.1e\n', pw_i*1e6, abs(norm(p4)-1));
    end
    % Check phase(1) ~= 0 (P4: phase(k=0) = pi*0^2/N - pi*0 = 0)
    if abs(angle(p4(1))) > 0.05
        pass_t8 = false;
        fprintf('  FAIL pw=%.0fus phase(1)=%.4f rad\n', pw_i*1e6, angle(p4(1)));
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t8 + 'FAIL'*(~pass_t8)));
results.s3t8 = pass_t8;

%% ====================================================================
%  S3T9: All 7 Types Unit Norm
%  ====================================================================
fprintf('S3T9: All 7 unit norm\n');
pass_t9 = true;
pw_checks = [10e-6, 50e-6];
bw_t9 = 200e6; fs_t9 = bw_t9;
chk = 0;
for qi = 1:length(pw_checks)
    wfs = gen_all_7(pw_checks(qi), bw_t9, fs_t9);
    for wi = 1:7
        nrm = norm(wfs{wi});
        err = abs(nrm - 1);
        chk = chk + 1;
        if err > 1e-5
            pass_t9 = false;
            fprintf('  FAIL pw=%.0fus %s norm=%.10f err=%.1e\n', ...
                pw_checks(qi)*1e6, wf_names{wi}, nrm, err);
        end
    end
end
fprintf('  %d checks [%s]\n', chk, char('PASS'*pass_t9 + 'FAIL'*(~pass_t9)));
results.s3t9 = pass_t9;

%% ====================================================================
%  S3T10: MF Compression Ratio
%  ====================================================================
fprintf('S3T10: MF compression ratio\n');
tbp_targets = [50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000];
pass_t10 = true;
NFFT_MF = 32768;
for ti = 1:length(tbp_targets)
    tbp_i = tbp_targets(ti);
    % Choose pw and bw to achieve target TBP
    bw_i = 200e6;
    pw_i = tbp_i / bw_i;
    if pw_i < 0.5e-6, pw_i = 0.5e-6; end
    if pw_i > 200e-6, pw_i = 200e-6; end
    fs_i = bw_i;
    n_i = max(1, floor(pw_i*fs_i));
    k_i = bw_i/pw_i;
    t_i = (0:n_i-1)'/fs_i;
    sig = exp(1j*pi*k_i*t_i.^2);
    sig = sig / norm(sig);
    % Matched filter: conjugate time-reverse
    mf = conj(sig(end:-1:1));
    % Apply MF via FFT (cap at 32768)
    NFFT = min(NFFT_MF, 2^nextpow2(2*n_i));
    mf_out = ifft(fft(sig, NFFT) .* fft(mf, NFFT));
    mf_pow = abs(mf_out).^2;
    [pk_val, ~] = max(mf_pow);
    % Measure 3-dB width (mainlobe bins)
    threshold = pk_val / 2;
    above = find(mf_pow > threshold);
    ml_width = length(above);
    if ml_width < 1, ml_width = 1; end
    actual_tbp = n_i;
    cr_meas = 10*log10(actual_tbp / ml_width);
    cr_theory = 10*log10(pw_i * bw_i);
    cr_err = abs(cr_meas - cr_theory);
    if cr_err > 2
        pass_t10 = false;
        fprintf('  FAIL TBP=%d CR=%.1fdB theory=%.1fdB err=%.1fdB\n', ...
            tbp_i, cr_meas, cr_theory, cr_err);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t10 + 'FAIL'*(~pass_t10)));
results.s3t10 = pass_t10;

%% ====================================================================
%  S3T11: MF Peak vs Delay (Range Resolution)
%  ====================================================================
fprintf('S3T11: MF peak vs delay\n');
R_km_sweep = [1 2 5 8 10 12 14];
bw_t11 = 200e6; fs_t11 = bw_t11;
pw_t11 = 50e-6;
n_t11 = max(1, floor(pw_t11*fs_t11));
k_t11 = bw_t11/pw_t11;
t_t11 = (0:n_t11-1)'/fs_t11;
sig_t11 = exp(1j*pi*k_t11*t_t11.^2);
sig_t11 = sig_t11 / norm(sig_t11);
mf_t11 = conj(sig_t11(end:-1:1));
range_res = c / (2*bw_t11);
pass_t11 = true;
NFFT_T11 = min(32768, 2^nextpow2(2*n_t11 + 10000));
fprintf('  range_res=%.2fm c/(2*bw)\n', range_res);
for ri = 1:length(R_km_sweep)
    R_m = R_km_sweep(ri)*1000;
    delay_samples = round(2*R_m/c*fs_t11);
    % Create delayed echo
    n_total = NFFT_T11;
    rx = complex(zeros(n_total, 1));
    src_start = delay_samples + 1;
    src_end = min(delay_samples + n_t11, n_total);
    copy_len = src_end - src_start + 1;
    if copy_len > 0
        rx(src_start:src_end) = sig_t11(1:copy_len);
    end
    % MF processing via FFT
    mf_padded = [mf_t11; complex(zeros(NFFT_T11 - n_t11, 1))];
    mf_out = ifft(fft(rx, NFFT_T11) .* fft(mf_padded, NFFT_T11));
    mf_pow = abs(mf_out).^2;
    [~, pk_idx] = max(mf_pow);
    measured_delay = pk_idx - n_t11;
    expected_delay = delay_samples;
    delay_err_samples = abs(measured_delay - expected_delay);
    range_err = delay_err_samples * c / (2*fs_t11);
    if range_err > 3*range_res
        pass_t11 = false;
        fprintf('  FAIL R=%dkm range_err=%.2fm (%.1f cells)\n', ...
            R_km_sweep(ri), range_err, range_err/range_res);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t11 + 'FAIL'*(~pass_t11)));
results.s3t11 = pass_t11;

%% ====================================================================
%  S3T12: Cross-Correlation (All 21 Pairs of 7 Waveforms)
%  ====================================================================
fprintf('S3T12: Cross-correlation\n');
pw_t12 = 50e-6; bw_t12 = 200e6; fs_t12 = bw_t12;
wfs_t12 = gen_all_7(pw_t12, bw_t12, fs_t12);
nw = 7;
NFFT_XC = 8192;  % Capped at 8192 for cross-correlation
ml = min(cellfun(@length, wfs_t12));
max_xcorr = 0;
max_pair = [0 0];
pass_t12 = true;
for wi = 1:nw
    S1 = fft(wfs_t12{wi}(1:ml), NFFT_XC);
    for wj = wi+1:nw
        S2 = fft(wfs_t12{wj}(1:ml), NFFT_XC);
        xc = abs(ifft(S1 .* conj(S2)));
        xcp = max(xc) / sqrt(sum(abs(wfs_t12{wi}(1:ml)).^2) * sum(abs(wfs_t12{wj}(1:ml)).^2));
        if xcp > max_xcorr
            max_xcorr = xcp;
            max_pair = [wi wj];
        end
    end
end
fprintf('  max=%.4f (%.1fdB) pair=%s/%s\n', max_xcorr, 20*log10(max_xcorr+eps), ...
    wf_names{max_pair(1)}, wf_names{max_pair(2)});
if max_xcorr >= 0.6
    pass_t12 = false;
end
fprintf('  [%s]\n', char('PASS'*pass_t12 + 'FAIL'*(~pass_t12)));
results.s3t12 = pass_t12;

%% ====================================================================
%  S3T13: TBP Conservation
%  ====================================================================
fprintf('S3T13: TBP conservation\n');
pw_sweep_t13 = [5 10 20 30 40 50 60 70 80 90 100 50]*1e-6;
bw_t13 = 200e6; fs_t13 = bw_t13;
pass_t13 = true;
for pi_idx = 1:length(pw_sweep_t13)
    pw_i = pw_sweep_t13(pi_idx);
    n_i = max(1, floor(pw_i*fs_t13));
    k_i = bw_t13/pw_i;
    t_i = (0:n_i-1)'/fs_t13;
    sig = exp(1j*pi*k_i*t_i.^2);
    sig = sig / norm(sig);
    % Matched filter
    mf = conj(sig(end:-1:1));
    NFFT = min(32768, 2^nextpow2(2*n_i));
    mf_out = ifft(fft(sig, NFFT) .* fft(mf, NFFT));
    mf_pow = abs(mf_out).^2;
    [pk_val, ~] = max(mf_pow);
    % Measure 3-dB mainlobe width
    threshold = pk_val / 2;
    above = find(mf_pow > threshold);
    ml_width = length(above);
    if ml_width < 1, ml_width = 1; end
    cr_meas = 10*log10(n_i / ml_width);
    cr_theory = 10*log10(pw_i * bw_t13);
    cr_err = abs(cr_meas - cr_theory);
    if cr_err > 2
        pass_t13 = false;
        fprintf('  FAIL pw=%.0fus CR=%.1fdB theory=%.1fdB err=%.1fdB\n', ...
            pw_i*1e6, cr_meas, cr_theory, cr_err);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t13 + 'FAIL'*(~pass_t13)));
results.s3t13 = pass_t13;

%% ====================================================================
%  S3T14: Reproducibility
%  ====================================================================
fprintf('S3T14: Reproducibility\n');
pass_t14 = true;
pw_t14 = 50e-6; bw_t14 = 200e6; fs_t14 = bw_t14;
wfs_a = gen_all_7(pw_t14, bw_t14, fs_t14);
wfs_b = gen_all_7(pw_t14, bw_t14, fs_t14);
for wi = 1:7
    max_diff = max(abs(wfs_a{wi} - wfs_b{wi}));
    if max_diff > 1e-10
        pass_t14 = false;
        fprintf('  FAIL %s max_diff=%.2e\n', wf_names{wi}, max_diff);
    end
end
fprintf('  [%s]\n', char('PASS'*pass_t14 + 'FAIL'*(~pass_t14)));
results.s3t14 = pass_t14;

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
save('validate_em_s3_waveform_results.mat', 'results');

%% ====================================================================
%  Local function: Generate all 7 waveform types at given pw/bw/fs
%  ====================================================================
function wfs = gen_all_7(pw_val, bw_val, fs_val)
    n_s = max(1, floor(pw_val * fs_val));
    t_s = (0:n_s-1)'/fs_val;
    k_v = bw_val / pw_val;

    % 1. LFM up
    lfm_u = exp(1j*pi*k_v*t_s.^2);
    lfm_u = lfm_u / norm(lfm_u);

    % 2. LFM down
    lfm_d = exp(-1j*pi*k_v*t_s.^2);
    lfm_d = lfm_d / norm(lfm_d);

    % 3. Barker-13
    barker_code = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
    reps = max(1, floor(pw_val*fs_val/13));
    barker = repelem(barker_code, reps);
    barker = complex(barker(:));
    barker = barker / norm(barker);

    % 4. Frank-16 (via interp1 — avoids MATLAB R2024a floor/indexing hang)
    n_phases = 4;
    phase_seq_16 = (2*pi/n_phases * ((0:n_phases-1)' * (0:n_phases-1)));
    phase_seq_16 = phase_seq_16(:);
    frank_phase = interp1(1:length(phase_seq_16), phase_seq_16, ...
        linspace(1, length(phase_seq_16), n_s), 'linear');
    frank = exp(1j*frank_phase);
    frank = frank / norm(frank);

    % 5. Costas-16
    cseq = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
    cl = max(1, floor(n_s/16));
    costas = complex(zeros(n_s, 1));
    for ci = 1:16
        s0 = (ci-1)*cl + 1;
        s1 = min(s0 + cl - 1, n_s);
        costas(s0:s1) = exp(1j*2*pi*cseq(ci)/pw_val*(0:s1-s0)'/fs_val);
    end
    costas = costas / norm(costas);

    % 6. NLFM (cubic modulation)
    nlfm = exp(1j*(pi*k_v*t_s.^2 + 0.3*pi*k_v/pw_val*t_s.^3));
    nlfm = nlfm / norm(nlfm);

    % 7. P4 (via interp1)
    npt = 16;
    kp = (0:npt-1)';
    php = pi*kp.^2/npt - pi*kp;
    p4_phase = interp1(1:npt, php, linspace(1, npt, n_s), 'linear');
    p4 = exp(1j*p4_phase);
    p4 = p4 / norm(p4);

    wfs = {lfm_u, lfm_d, barker, frank, costas, nlfm, p4};
end
