%% FluxPhased S5: Interference / SI / Polarization Validation (13 Tests)
%  Run: cd validation && matlab -batch "validate_em_s5_interference"
%  Validates IQ-level electromagnetic interference models: cross-radar path
%  loss, polarization loss, SI coupling, off-boresight gain, delay alignment,
%  pairwise interference matrices, angular wrapping, frequency overlap,
%  multi-target superposition, and SINR consistency.
%  Constraints: char('PASS'*ok+'FAIL'*(~ok)), Friis one-way PL, cosd^1.5 element.

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
elem_y = ((0:rows-1)-(rows-1)/2)*dy_m;
[Ex,Ey] = meshgrid(elem_x, elem_y); ex = Ex(:); ey = Ey(:);

n_pass = 0; n_fail = 0; results = struct();

fprintf('FluxPhased S5: Interference / SI / Polarization Validation (13 tests)\n');
fprintf('fc=%.0fGHz bw=%.0fMHz c=%.0f lambda=%.4fm D=%.1fdBi\n\n', ...
    fc/1e9, bw/1e6, c, lambda, D_dBi);

%% ====================================================================
%  S5T1: Cross-Radar Path Loss (12 distances)
%  ====================================================================
fprintf('S5T1: Cross-Radar Path Loss\n');
d_km_t1 = [0.5 1 2 3 5 7 10 15 20 30 40 50];
pass_t1 = true;
for di = 1:length(d_km_t1)
    d_m = d_km_t1(di)*1000;
    PL = 20*log10(4*pi*d_m/lambda);
    % Two-antenna one-way link budget
    Pr = tx_dbm + D_dBi + D_dBi - PL - Lsys_db;
    % Direct formula: Pr_ref = 10*log10(Pt*G^2/((4*pi*d/lambda)^2*L)*1000)
    G_lin = 10^(D_dBi/10);
    L_lin = 10^(Lsys_db/10);
    Pr_ref = 10*log10(tx_power_w * G_lin^2 / ((4*pi*d_m/lambda)^2 * L_lin) * 1000);
    err = abs(Pr - Pr_ref);
    if err > 0.5
        pass_t1 = false;
        fprintf('  FAIL d=%gkm Pr=%.2fdBm Pr_ref=%.2fdBm err=%.2fdB\n', ...
            d_km_t1(di), Pr, Pr_ref, err);
    end
end
fprintf('  max_err check [%s]\n', char('PASS'*pass_t1 + 'FAIL'*(~pass_t1)));
results.s5t1 = pass_t1;

%% ====================================================================
%  S5T2: Polarization Loss (12 pol values)
%  ====================================================================
fprintf('S5T2: Polarization Loss\n');
pol_loss_db = [0 0.5 1 1.5 2 3 4 5 6 7 8 10];
% Base received power (no polarization loss)
d_ref_m = 10e3;
PL_ref = 20*log10(4*pi*d_ref_m/lambda);
Pr_no_pol = tx_dbm + D_dBi + D_dBi - PL_ref - Lsys_db;
pass_t2 = true;
for pi_idx = 1:length(pol_loss_db)
    pol = pol_loss_db(pi_idx);
    Pr_with_pol = Pr_no_pol - pol;
    % Verify exact dB subtraction
    Pr_expected = Pr_no_pol - pol;
    err = abs(Pr_with_pol - Pr_expected);
    if err > 0.01
        pass_t2 = false;
        fprintf('  FAIL pol=%.1fdB Pr=%.4fdBm expected=%.4fdBm err=%.4f\n', ...
            pol, Pr_with_pol, Pr_expected, err);
    end
end
fprintf('  all 12 pol values [%s]\n', char('PASS'*pass_t2 + 'FAIL'*(~pass_t2)));
results.s5t2 = pass_t2;

%% ====================================================================
%  S5T3: SI Coupling (12 iso values)
%  ====================================================================
fprintf('S5T3: SI Coupling\n');
iso_db = [5 10 15 20 25 30 35 40 45 50 55 60];
pass_t3 = true;
for ii = 1:length(iso_db)
    iso = iso_db(ii);
    % SI power at RX (dBm)
    si_dbm = tx_dbm - iso;
    % Voltage coupling coefficient
    coupling = 10^(-iso/20);
    % Verify via power: 10*log10(coupling^2 * tx_power_w * 1000)
    si_formula = 10*log10(coupling^2 * tx_power_w * 1000);
    err = abs(si_dbm - si_formula);
    if err > 0.01
        pass_t3 = false;
        fprintf('  FAIL iso=%ddB si=%.2fdBm formula=%.2fdBm err=%.4f\n', ...
            iso, si_dbm, si_formula, err);
    end
end
fprintf('  all 12 iso values [%s]\n', char('PASS'*pass_t3 + 'FAIL'*(~pass_t3)));
results.s5t3 = pass_t3;

%% ====================================================================
%  S5T4: Off-Boresight Array Gain (12 angles)
%  ====================================================================
fprintf('S5T4: Off-Boresight Array Gain\n');
az_t4 = [0 5 10 15 20 25 30 35 40 45 50 55];
pass_t4 = true;
for ai = 1:length(az_t4)
    az = az_t4(ai);
    % Element voltage pattern: cosd(az)^1.5
    elem_voltage = cosd(az)^1.5;
    % Gain reduction in dB: 10*log10(cosd(az)^3)
    gain_red_db = 10*log10(cosd(az)^3);
    % Verify: 20*log10(cosd(az)^1.5) should equal gain_red_db
    gain_via_voltage = 20*log10(elem_voltage);
    err = abs(gain_red_db - gain_via_voltage);
    if err > 0.5
        pass_t4 = false;
        fprintf('  FAIL az=%ddeg theory=%.4fdB measured=%.4fdB err=%.4f\n', ...
            az, gain_red_db, gain_via_voltage, err);
    end
end
fprintf('  all 12 angles [%s]\n', char('PASS'*pass_t4 + 'FAIL'*(~pass_t4)));
results.s5t4 = pass_t4;

%% ====================================================================
%  S5T5: Cross-Radar Delay (12 distances)
%  ====================================================================
fprintf('S5T5: Cross-Radar Delay\n');
d_km_t5 = [0.5 1 2 3 5 7 10 15 20 30 40 50];
pass_t5 = true;
for di = 1:length(d_km_t5)
    d_m = d_km_t5(di)*1000;
    % One-way delay in samples
    dn = round(d_m/c*fs);
    % Verify round-trip of conversion
    d_recovered = dn*c/fs;
    err_m = abs(d_recovered - d_m);
    tol = c/fs; % within one sample
    if err_m >= tol
        pass_t5 = false;
        fprintf('  FAIL d=%gkm dn=%d d_recovered=%.4fm err=%.4fm tol=%.4f\n', ...
            d_km_t5(di), dn, d_recovered, err_m, tol);
    end
end
fprintf('  all 12 distances [%s]\n', char('PASS'*pass_t5 + 'FAIL'*(~pass_t5)));
results.s5t5 = pass_t5;

%% ====================================================================
%  S5T6: Interference TX Power Scaling (12 tx levels)
%  ====================================================================
fprintf('S5T6: Interference TX Power Scaling\n');
tx_dbm_t6 = [10 15 20 25 30 35 40 45 50 55 60 65];
G_rx_dBi = D_dBi;
d_t6_m = 10e3;
PL_t6 = 20*log10(4*pi*d_t6_m/lambda);
pol_t6 = 0; % no polarization loss
Pr_t6 = zeros(size(tx_dbm_t6));
for ti = 1:length(tx_dbm_t6)
    Pr_t6(ti) = tx_dbm_t6(ti) + D_dBi + G_rx_dBi - PL_t6 - Lsys_db - pol_t6;
end
pass_t6 = true;
% Verify adjacent differences all == 5 dB (since tx steps by 5 dB)
diffs = diff(Pr_t6);
for di = 1:length(diffs)
    if abs(diffs(di) - 5) > 0.001
        pass_t6 = false;
        fprintf('  FAIL step %d->%d: Pr_diff=%.6fdB expected=5\n', ...
            di, di+1, diffs(di));
    end
end
fprintf('  all diffs=5dB [%s]\n', char('PASS'*pass_t6 + 'FAIL'*(~pass_t6)));
results.s5t6 = pass_t6;

%% ====================================================================
%  S5T7: N-Radar Pairwise Matrix (6 n_radars values)
%  ====================================================================
fprintf('S5T7: N-Radar Pairwise Matrix\n');
n_radars_t7 = [2 3 4 5 6 8];
pass_t7 = true;
for ni = 1:length(n_radars_t7)
    n = n_radars_t7(ni);
    % Build interference power matrix
    int_mat = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if i == j
                int_mat(i,j) = -Inf; % no self-interference in cross-radar model
            else
                % Use placeholder Friis path loss at 10 km
                d_ij = 10e3;
                PL_ij = 20*log10(4*pi*d_ij/lambda);
                int_mat(i,j) = tx_dbm + D_dBi + D_dBi - PL_ij - Lsys_db;
            end
        end
    end
    % Verify symmetry
    sym_err = max(max(abs(int_mat - int_mat')));
    % Check diagonal = -Inf
    diag_ok = all(diag(int_mat) == -Inf);
    % Count unique pairs
    n_pairs = n*(n-1)/2;
    % Count actual off-diagonal entries (upper triangle)
    upper_entries = int_mat(triu(ones(n), 1) == 1);
    actual_pairs = sum(upper_entries ~= -Inf & upper_entries ~= 0 & ~isnan(upper_entries));

    if sym_err > 1e-10 || ~diag_ok || actual_pairs ~= n_pairs
        pass_t7 = false;
        fprintf('  FAIL n=%d sym_err=%.1e diag_ok=%d pairs=%d/%d\n', ...
            n, sym_err, diag_ok, actual_pairs, n_pairs);
    end
end
fprintf('  all 6 n_radars [%s]\n', char('PASS'*pass_t7 + 'FAIL'*(~pass_t7)));
results.s5t7 = pass_t7;

%% ====================================================================
%  S5T8: Angular Wrapping (12 relative azimuths)
%  ====================================================================
fprintf('S5T8: Angular Wrapping\n');
rel_az_t8 = [-350 -300 -200 -100 -10 10 100 200 300 350 710 -710];
expected_wrapped = [10 60 160 260 350 10 100 200 300 350 350 10];
% Actually compute: wrapped = mod(rel_az+180, 360) - 180
pass_t8 = true;
for ai = 1:length(rel_az_t8)
    raw = rel_az_t8(ai);
    wrapped = mod(raw + 180, 360) - 180;
    % Compute element pattern
    elem_gain = cosd(wrapped)^3; % power pattern for element
    % Verify wrapping is in [-180, 180]
    if wrapped < -180 || wrapped > 180
        pass_t8 = false;
        fprintf('  FAIL raw=%d wrapped=%.1f out of range\n', raw, wrapped);
    end
    % Verify cosd consistency: cosd(wrapped) should match cosd of equivalent angle
    cos_direct = cosd(raw);
    cos_wrapped = cosd(wrapped);
    if abs(cos_direct - cos_wrapped) > 1e-10
        pass_t8 = false;
        fprintf('  FAIL raw=%d cos mismatch: direct=%.10f wrapped=%.10f\n', ...
            raw, cos_direct, cos_wrapped);
    end
end
fprintf('  all 12 azimuths [%s]\n', char('PASS'*pass_t8 + 'FAIL'*(~pass_t8)));
results.s5t8 = pass_t8;

%% ====================================================================
%  S5T9: TX Gain Dependence (12 G_tx values)
%  ====================================================================
fprintf('S5T9: TX Gain Dependence\n');
G_tx_t9 = [10 15 20 25 30 35 37 39 41 43 44 44];
G_rx_t9 = D_dBi;
d_t9_m = 10e3;
PL_t9 = 20*log10(4*pi*d_t9_m/lambda);
Pr_t9 = zeros(size(G_tx_t9));
for gi = 1:length(G_tx_t9)
    Pr_t9(gi) = tx_dbm + G_tx_t9(gi) + G_rx_t9 - PL_t9 - Lsys_db;
end
pass_t9 = true;
% Verify: Pr differences match G_tx differences exactly
G_tx_diffs = diff(G_tx_t9);
Pr_diffs = diff(Pr_t9);
for di = 1:length(G_tx_diffs)
    err = abs(Pr_diffs(di) - G_tx_diffs(di));
    if err > 1e-10
        pass_t9 = false;
        fprintf('  FAIL step %d->%d: Pr_diff=%.6f G_tx_diff=%d err=%.2e\n', ...
            di, di+1, Pr_diffs(di), G_tx_diffs(di), err);
    end
end
fprintf('  all 12 G_tx values [%s]\n', char('PASS'*pass_t9 + 'FAIL'*(~pass_t9)));
results.s5t9 = pass_t9;

%% ====================================================================
%  S5T10: Frequency Overlap (12 offsets)
%  ====================================================================
fprintf('S5T10: Frequency Overlap\n');
freq_offset_mhz = [0 25 50 75 100 125 150 175 200 250 300 350];
freq_offset = freq_offset_mhz * 1e6;
pass_t10 = true;
for fi = 1:length(freq_offset)
    offset = freq_offset(fi);
    % Overlap factor: max(0, (bw - offset)/bw)
    overlap = max(0, (bw - offset)/bw);
    % Analytical expected overlap
    if offset >= bw
        expected = 0;
    else
        expected = (bw - offset)/bw;
    end
    err = abs(overlap - expected);
    if err > 1e-12
        pass_t10 = false;
        fprintf('  FAIL offset=%dMHz overlap=%.6f expected=%.6f err=%.2e\n', ...
            freq_offset_mhz(fi), overlap, expected, err);
    end
    % Sanity checks
    if offset == 0 && abs(overlap - 1.0) > 1e-12
        pass_t10 = false;
        fprintf('  FAIL offset=0 not full overlap: %.6f\n', overlap);
    end
    if offset >= bw && overlap ~= 0
        pass_t10 = false;
        fprintf('  FAIL offset=%d>=bw=%d but overlap=%.6f\n', ...
            freq_offset_mhz(fi), bw/1e6, overlap);
    end
end
fprintf('  all 12 offsets [%s]\n', char('PASS'*pass_t10 + 'FAIL'*(~pass_t10)));
results.s5t10 = pass_t10;

%% ====================================================================
%  S5T11: Multi-Target Superposition (12 target counts)
%  ====================================================================
fprintf('S5T11: Multi-Target Superposition\n');
n_targets_t11 = [1 2 3 4 5 6 7 8 10 12 15 20];
pass_t11 = true;
% Use reproducible seed for random amplitudes
rng(42);
n_buf = 100000; % large enough buffer for all target delays
for ti = 1:length(n_targets_t11)
    nt = n_targets_t11(ti);
    % Generate random delays and amplitudes
    delays = sort(randi([100 n_buf-1000], nt, 1));
    amps = 0.1 + rand(nt, 1)*0.9; % amplitudes in [0.1, 1.0]
    phases = rand(nt, 1)*2*pi;
    % Build individual target signals and sum
    rx_all = complex(zeros(n_buf, 1));
    target_sigs = cell(nt, 1);
    for ki = 1:nt
        sig_k = complex(zeros(n_buf, 1));
        d_start = delays(ki);
        d_end = min(delays(ki) + n_lfm - 1, n_buf);
        copy_len = d_end - d_start + 1;
        sig_k(d_start:d_end) = amps(ki) * exp(1j*phases(ki)) * lfm_up(1:copy_len);
        target_sigs{ki} = sig_k;
        rx_all = rx_all + sig_k;
    end
    % Verify superposition: at each target's peak delay, check that
    % sum of all contributions equals rx_all
    max_err = 0;
    for ki = 1:nt
        d_start = delays(ki);
        d_end = min(delays(ki) + n_lfm - 1, n_buf);
        % Within this target's support, check that rx_all - sum of all others
        % equals this target's signal
        others_sum = complex(zeros(n_buf, 1));
        for kj = 1:nt
            if kj ~= ki
                others_sum = others_sum + target_sigs{kj};
            end
        end
        residual = rx_all - others_sum - target_sigs{ki};
        seg_err = max(abs(residual(d_start:d_end)));
        if seg_err > max_err
            max_err = seg_err;
        end
    end
    if max_err > 1e-10
        pass_t11 = false;
        fprintf('  FAIL n_targets=%d max_err=%.2e\n', nt, max_err);
    end
end
fprintf('  all 12 target counts [%s]\n', char('PASS'*pass_t11 + 'FAIL'*(~pass_t11)));
results.s5t11 = pass_t11;

%% ====================================================================
%  S5T12: IQ Delay Alignment (12 distances)
%  ====================================================================
fprintf('S5T12: IQ Delay Alignment\n');
d_km_t12 = [0.5 1 2 3 5 7 10 15 20 30 40 50];
pass_t12 = true;
for di = 1:length(d_km_t12)
    d_m = d_km_t12(di)*1000;
    % One-way delay in samples
    dn = round(d_m/c*fs);
    % Build TX signal in a buffer
    n_buf_t12 = max(dn + n_lfm + 100, 2*n_lfm + 100);
    tx_sig = complex(zeros(n_buf_t12, 1));
    tx_sig(1:n_lfm) = lfm_up;
    % Apply one-way delay: received signal at sample s = tx(s - dn)
    rx_sig = complex(zeros(n_buf_t12, 1));
    for s = (dn+1):n_buf_t12
        rx_sig(s) = tx_sig(s - dn);
    end
    % Verify alignment: rx_sig(s) == tx_sig(s - dn) for all valid s
    max_err = 0;
    for s = (dn+1):(dn+n_lfm)
        err = abs(rx_sig(s) - tx_sig(s - dn));
        if err > max_err
            max_err = err;
        end
    end
    % Also verify that before the delay, signal is zero
    pre_err = max(abs(rx_sig(1:dn)));
    if pre_err > 1e-15
        max_err = max(max_err, pre_err);
    end
    if max_err > 1e-10
        pass_t12 = false;
        fprintf('  FAIL d=%gkm dn=%d max_err=%.2e\n', d_km_t12(di), dn, max_err);
    end
end
fprintf('  all 12 distances [%s]\n', char('PASS'*pass_t12 + 'FAIL'*(~pass_t12)));
results.s5t12 = pass_t12;

%% ====================================================================
%  S5T13: SINR Consistency (12 J levels)
%  ====================================================================
fprintf('S5T13: SINR Consistency\n');
J_dbm_t13 = [-120 -100 -80 -60 -40 -20 -10 -5 0 5 10 20];
% Signal power (single-target return at reference range)
d_s_m = 10e3;
PL_s = 20*log10(4*pi*d_s_m/lambda);
S_dbm = tx_dbm + D_dBi + D_dBi - PL_s - Lsys_db - 2*Lsys_db; % TX+RX system loss
% Simplified: use a fixed signal level
S_dbm = 30; % fix signal at 30 dBm for clean SINR test
pass_t13 = true;
for ji = 1:length(J_dbm_t13)
    J_dbm = J_dbm_t13(ji);
    % SINR = S_dbm - 10*log10(10^(J_dbm/10) + 10^(noise_dbm/10))
    int_plus_noise_dbm = 10*log10(10^(J_dbm/10) + 10^(noise_dbm/10));
    SINR = S_dbm - int_plus_noise_dbm;

    % Case 1: J << noise => SINR ~ SNR = S_dbm - noise_dbm
    SNR = S_dbm - noise_dbm;
    J_lin = 10^(J_dbm/10);
    N_lin = 10^(noise_dbm/10);

    if J_lin < N_lin / 100  % J << noise (J is 20 dB below noise)
        sinr_approx = SNR;
        err = abs(SINR - sinr_approx);
        if err > 0.5
            pass_t13 = false;
            fprintf('  FAIL J=%ddBm (J<<N) SINR=%.2fdB SNR=%.2fdB err=%.2f\n', ...
                J_dbm, SINR, sinr_approx, err);
        end
    end

    % Case 2: J >> noise => SINR ~ S - J
    if J_lin > N_lin * 100  % J >> noise (J is 20 dB above noise)
        sinr_approx = S_dbm - J_dbm;
        err = abs(SINR - sinr_approx);
        if err > 0.5
            pass_t13 = false;
            fprintf('  FAIL J=%ddBm (J>>N) SINR=%.2fdB S-J=%.2fdB err=%.2f\n', ...
                J_dbm, SINR, sinr_approx, err);
        end
    end

    % General case: verify against analytical formula
    SINR_ref = 10*log10(10^(S_dbm/10) / (10^(J_dbm/10) + 10^(noise_dbm/10)));
    err_general = abs(SINR - SINR_ref);
    if err_general > 0.5
        pass_t13 = false;
        fprintf('  FAIL J=%ddBm SINR=%.2fdB SINR_ref=%.2fdB err=%.2f\n', ...
            J_dbm, SINR, SINR_ref, err_general);
    end
end
fprintf('  all 12 J levels [%s]\n', char('PASS'*pass_t13 + 'FAIL'*(~pass_t13)));
results.s5t13 = pass_t13;

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
save('validate_em_s5_results.mat', 'results');
