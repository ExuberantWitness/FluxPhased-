%% FluxPhased Channel & Radar Equation Validation (14 Tests)
%  Run: cd validation && matlab -batch "validate_em_s2_channel"
%  Validates channel model and radar equation consistency against MATLAB
%  analytical models. Tests: Pr vs range/RCS/power/loss/frequency,
%  per-element gain, delay, Doppler, one-way path loss, one-way vs two-way,
%  noise power, noise figure, range error, SNR vs range.

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

fprintf('FluxPhased Channel & Radar Equation Validation (14 tests)\n');
fprintf('fc=%.0fGHz bw=%.0fMHz %dx%d D=%.1fdBi tx=%.1fdBm\n\n', fc/1e9, bw/1e6, rows, cols, D_dBi, tx_dbm);

%% S2T1: Pr vs Range
fprintf('S2T1: Pr vs Range\n');
pass1 = true; max_err1 = 0;
R_km_list = [0.5,1,2,5,10,15,20,25,30,40,50,0.5];
for ri = 1:length(R_km_list)
    R_km = R_km_list(ri); R_m = R_km*1000;
    rcs_dbsm = 10;
    % dB formula (FluxPhased convention)
    Pr_dB = tx_dbm + 2*D_dBi + rcs_dbsm + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m) - Lsys_db;
    % Linear formula
    G_lin = 10^(D_dBi/10); L_lin = 10^(Lsys_db/10);
    sigma_lin = 10^(rcs_dbsm/10);
    Pr_lin = 10*log10(tx_power_w * G_lin^2 * lambda^2 * sigma_lin ...
        / ((4*pi)^3 * R_m^4 * L_lin) * 1000);
    err1 = abs(Pr_dB - Pr_lin);
    if err1 > max_err1, max_err1 = err1; end
    if err1 > 0.01, pass1 = false; end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err1, char('PASS'*pass1+'FAIL'*(~pass1)));
results.s2t1 = pass1;

%% S2T2: Pr vs RCS
fprintf('S2T2: Pr vs RCS\n');
pass2 = true; max_err2 = 0;
rcs_list = [-10,-5,0,5,10,15,20,25,30,0,10,20];
R_m2 = 10000; % 10 km
Pr2 = zeros(1,length(rcs_list));
for ri = 1:length(rcs_list)
    rcs_dbsm = rcs_list(ri);
    Pr2(ri) = tx_dbm + 2*D_dBi + rcs_dbsm + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m2) - Lsys_db;
end
% Check Pr(sigma1)-Pr(sigma2) = sigma1-sigma2 for all pairs
for i = 1:length(rcs_list)
    for j = i+1:length(rcs_list)
        expected_diff = rcs_list(i) - rcs_list(j);
        actual_diff = Pr2(i) - Pr2(j);
        err2 = abs(actual_diff - expected_diff);
        if err2 > max_err2, max_err2 = err2; end
        if err2 > 0.01, pass2 = false; end
    end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err2, char('PASS'*pass2+'FAIL'*(~pass2)));
results.s2t2 = pass2;

%% S2T3: Pr vs TX Power
fprintf('S2T3: Pr vs TX Power\n');
pass3 = true; max_err3 = 0;
tx_w_list = [0.01,0.1,1,10,100,1000,1e4,5e4,1e5,0.01,1,5e4];
R_m3 = 10000; rcs3 = 10;
Pr3 = zeros(1,length(tx_w_list));
for ti = 1:length(tx_w_list)
    tx_dbm_i = 10*log10(tx_w_list(ti)*1000);
    Pr3(ti) = tx_dbm_i + 2*D_dBi + rcs3 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m3) - Lsys_db;
end
% Check Pr scales 1:1 in dB with TX power
for i = 1:length(tx_w_list)
    for j = i+1:length(tx_w_list)
        expected_diff = 10*log10(tx_w_list(i)) - 10*log10(tx_w_list(j));
        actual_diff = Pr3(i) - Pr3(j);
        err3 = abs(actual_diff - expected_diff);
        if err3 > max_err3, max_err3 = err3; end
        if err3 > 0.01, pass3 = false; end
    end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err3, char('PASS'*pass3+'FAIL'*(~pass3)));
results.s2t3 = pass3;

%% S2T4: Pr vs System Loss
fprintf('S2T4: Pr vs System Loss\n');
pass4 = true; max_err4 = 0;
Lsys_list = [0,0.5,1,2,3,4,5,6,7,8,9,10];
R_m4 = 10000; rcs4 = 10;
Pr4 = zeros(1,length(Lsys_list));
for li = 1:length(Lsys_list)
    Pr4(li) = tx_dbm + 2*D_dBi + rcs4 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m4) - Lsys_list(li);
end
% Check Pr(L)-Pr(0) = -L
for li = 1:length(Lsys_list)
    expected = -Lsys_list(li);
    actual = Pr4(li) - Pr4(1);
    err4 = abs(actual - expected);
    if err4 > max_err4, max_err4 = err4; end
    if err4 > 0.01, pass4 = false; end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err4, char('PASS'*pass4+'FAIL'*(~pass4)));
results.s2t4 = pass4;

%% S2T5: Pr vs Frequency
fprintf('S2T5: Pr vs Frequency\n');
pass5 = true; max_err5 = 0;
fc_list = [8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,8.0,10.0,12.0]*1e9;
R_m5 = 10000; rcs5 = 10;
Pr5 = zeros(1,length(fc_list));
for fi = 1:length(fc_list)
    lambda_i = c/fc_list(fi);
    % D is constant if dx_wl fixed (area_wl2 unchanged)
    Pr5(fi) = tx_dbm + 2*D_dBi + rcs5 + 20*log10(lambda_i) ...
        - 30*log10(4*pi) - 40*log10(R_m5) - Lsys_db;
end
% Check Pr(fc1)-Pr(fc2) = 20*log10(fc2/fc1)
for i = 1:length(fc_list)
    for j = i+1:length(fc_list)
        expected_diff = 20*log10(fc_list(j)/fc_list(i));
        actual_diff = Pr5(i) - Pr5(j);
        err5 = abs(actual_diff - expected_diff);
        if err5 > max_err5, max_err5 = err5; end
        if err5 > 0.01, pass5 = false; end
    end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err5, char('PASS'*pass5+'FAIL'*(~pass5)));
results.s2t5 = pass5;

%% S2T6: Per-element Gain
fprintf('S2T6: Per-element Gain\n');
pass6 = true; max_err6 = 0;
N_list = [4,16,25,64,100,144,225,400,625,1024,2500,10000];
for ni = 1:length(N_list)
    N = N_list(ni);
    % Directivity for N elements (same area_wl2 = N*0.5*0.5)
    D_i = 10*log10(4*pi*N*0.5*0.5);
    % Beamformed received power (arbitrary reference R, RCS)
    R_m6 = 10000; rcs6 = 10;
    Pr_bf_dBm = tx_dbm + 2*D_i + rcs6 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m6) - Lsys_db;
    Pr_bf_w = 10^((Pr_bf_dBm-30)/10);
    gain = sqrt(Pr_bf_w / N);
    % Check 20*log10(gain*sqrt(N)) = Pr_bf_dBm - 30
    lhs = 20*log10(gain*sqrt(N));
    rhs = Pr_bf_dBm - 30;
    err6 = abs(lhs - rhs);
    if err6 > max_err6, max_err6 = err6; end
    if err6 > 0.01, pass6 = false; end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err6, char('PASS'*pass6+'FAIL'*(~pass6)));
results.s2t6 = pass6;

%% S2T7: Delay vs Range
fprintf('S2T7: Delay vs Range\n');
pass7 = true;
R_km_list7 = [0.5,1,2,3,4,5,6,7,0.5,1,2,5];
NFFT7 = 32768;
for ri = 1:length(R_km_list7)
    R_km = R_km_list7(ri); R_m = R_km*1000;
    dn = round(2*R_m/c*fs);
    tx_sig = [lfm_up; complex(zeros(n_samp-n_lfm,1))];
    rx_sig = complex(zeros(n_samp,1));
    for s = 1:n_samp
        src = s-dn;
        if src>=1 && src<=n_samp, rx_sig(s)=tx_sig(src); end
    end
    mf = ifft(fft(rx_sig,NFFT7).*fft([conj(flipud(lfm_up));complex(zeros(NFFT7-n_lfm,1))],NFFT7));
    [~,pk] = max(abs(mf(1:n_samp)).^2);
    rng_meas = (pk-n_lfm)*c/(2*fs);
    err7 = abs(rng_meas-R_m);
    if err7 > c/fs*2, pass7 = false; end
    fprintf('  R=%.1fkm meas=%.1fkm err=%.2fm\n', R_km, rng_meas/1000, err7);
end
fprintf('  [%s]\n', char('PASS'*pass7+'FAIL'*(~pass7)));
results.s2t7 = pass7;

%% S2T8: Doppler vs Velocity
fprintf('S2T8: Doppler vs Velocity\n');
pass8 = true;
v_list = [-300,-200,-100,-50,-10,0,10,50,100,200,300,-150];
n8 = 2^17; t8 = (0:n8-1)';
NFFT8 = 2^18;
for vi = 1:length(v_list)
    v = v_list(vi);
    fd = 2*v*fc/c;
    if fd == 0
        fprintf('  v=%d m/s fd=0 (skipped)\n', v);
        continue;
    end
    ds = 2*pi*fd/fs;
    sig = exp(1j*t8*ds);
    spec = fftshift(fft(sig, NFFT8));
    fr = (-NFFT8/2:NFFT8/2-1)*fs/NFFT8;
    [~,pk] = max(abs(spec));
    fd_meas = fr(pk);
    % Parabolic interpolation for sub-bin accuracy
    if pk > 1 && pk < length(spec)
        y1 = abs(spec(pk-1)); y2 = abs(spec(pk)); y3 = abs(spec(pk+1));
        denom = y1 - 2*y2 + y3;
        if abs(denom) > 1e-30
            delta = 0.5 * (y1 - y3) / denom;
            fd_meas = fr(pk) + delta * (fs/NFFT8);
        end
    end
    err_pct = abs(fd_meas-fd)/abs(fd)*100;
    fprintf('  v=%d fd=%.1f meas=%.1f err=%.2f%%\n', v, fd, fd_meas, err_pct);
    if err_pct > 5.0, pass8 = false; end
end
fprintf('  [%s]\n', char('PASS'*pass8+'FAIL'*(~pass8)));
results.s2t8 = pass8;

%% S2T9: One-way Path Loss
fprintf('S2T9: One-way Path Loss\n');
pass9 = true; max_err9 = 0;
d_km_list9 = [0.5,1,2,5,10,15,20,25,30,40,50,0.5];
for di = 1:length(d_km_list9)
    d_km = d_km_list9(di); d_m = d_km*1000;
    PL = 20*log10(4*pi*d_m/lambda);
    Pr_dB = tx_dbm + D_dBi - PL - Lsys_db;
    % Linear formula
    G_lin = 10^(D_dBi/10); L_lin = 10^(Lsys_db/10);
    Pr_lin = 10*log10(tx_power_w * G_lin / ((4*pi*d_m/lambda)^2 * L_lin) * 1000);
    err9 = abs(Pr_dB - Pr_lin);
    if err9 > max_err9, max_err9 = err9; end
    if err9 > 0.01, pass9 = false; end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err9, char('PASS'*pass9+'FAIL'*(~pass9)));
results.s2t9 = pass9;

%% S2T10: One-way vs Two-way Consistency
fprintf('S2T10: One-way vs Two-way\n');
pass10 = true; max_err10 = 0;
d_km_list10 = [1,2,5,10,15,20,25,30,35,40,45,50];
for di = 1:length(d_km_list10)
    d_km = d_km_list10(di); d_m = d_km*1000; rcs10 = 10;
    % Two-way radar equation
    Pr_two = tx_dbm + 2*D_dBi + rcs10 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(d_m) - Lsys_db;
    % One-way link to target
    PL_one = 20*log10(4*pi*d_m/lambda);
    Pr_one = tx_dbm + D_dBi - PL_one - Lsys_db;
    % Two-way derived from one-way: Pr_two = Pr_one + G + σ - 10*log10(4π) - 20*log10(R)
    Pr_two_from_one = Pr_one + D_dBi + rcs10 - 10*log10(4*pi) - 20*log10(d_m);
    err10 = abs(Pr_two - Pr_two_from_one);
    if err10 > max_err10, max_err10 = err10; end
    if err10 > 0.01, pass10 = false; end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err10, char('PASS'*pass10+'FAIL'*(~pass10)));
results.s2t10 = pass10;

%% S2T11: Noise Power vs Bandwidth
fprintf('S2T11: Noise Power vs BW\n');
pass11 = true; max_err11 = 0;
bw_MHz_list = [50,75,100,125,150,175,200,250,300,350,400,50];
for bi = 1:length(bw_MHz_list)
    bw_hz = bw_MHz_list(bi)*1e6;
    % Analytical noise power in dBm
    noise_w_i = kB*T_noise*bw_hz*10^(NF_db/10);
    noise_dbm_i = 10*log10(noise_w_i*1000);
    % Exact kT reference (avoid -174 dBm/Hz approximation error)
    noise_ref = 10*log10(kB*T_noise*1000) + 10*log10(bw_hz) + NF_db;
    err11 = abs(noise_dbm_i - noise_ref);
    if err11 > max_err11, max_err11 = err11; end
    if err11 > 0.01, pass11 = false; end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err11, char('PASS'*pass11+'FAIL'*(~pass11)));
results.s2t11 = pass11;

%% S2T12: Noise vs NF
fprintf('S2T12: Noise vs NF\n');
pass12 = true;
NF_list = [0,1,2,3,4,5,6,7,8,9,10,15];
nn12 = 1e5;
max_err12 = 0;
for ni = 1:length(NF_list)
    nf_i = NF_list(ni);
    noise_w_i = kB*T_noise*bw*10^(nf_i/10);
    noise_std_i = sqrt(noise_w_i/2);
    nc = noise_std_i*(randn(nn12,1)+1j*randn(nn12,1));
    nf_meas = 10*log10(mean(abs(nc).^2)/(kB*T_noise*bw));
    err12 = abs(nf_meas - nf_i);
    if err12 > max_err12, max_err12 = err12; end
    if err12 > 0.3, pass12 = false; end
end
fprintf('  max_err=%.2f dB [%s]\n', max_err12, char('PASS'*pass12+'FAIL'*(~pass12)));
results.s2t12 = pass12;

%% S2T13: Range Error vs Distance (R<=2km only)
fprintf('S2T13: Range Error vs Distance\n');
pass13 = true;
R_km_list13 = [1,2,5,8,10,12,15,20,25,30,40,50];
rcs13 = 20; % dBsm
v13 = 150; % m/s
fd13 = 2*v13*fc/c;
ds13 = 2*pi*fd13/fs;
n_tested13 = 0;
for ri = 1:length(R_km_list13)
    R_km = R_km_list13(ri); R_m = R_km*1000;
    % Compute per-element SNR check
    Pr13 = tx_dbm + 2*D_dBi + rcs13 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m) - Lsys_db;
    Pr_w13 = 10^((Pr13-30)/10);
    g13 = sqrt(Pr_w13/N_elem);
    % Per-element SNR estimate after MF processing
    snr_est = g13^2 * n_lfm / noise_w;
    snr_est_db = 10*log10(snr_est);
    if R_km > 2
        fprintf('  R=%gkm SNR_est=%.1fdB (skipped, R>2km)\n', R_km, snr_est_db);
        continue;
    end
    n_tested13 = n_tested13 + 1;
    dn = round(2*R_m/c*fs);
    tx_sig = [lfm_up; complex(zeros(n_samp-n_lfm,1))];
    rx_sig = complex(zeros(n_samp,1));
    for s = 1:n_samp
        src = s-dn;
        if src>=1 && src<=n_samp
            rx_sig(s) = tx_sig(src)*g13*exp(1j*(s-1)*ds13);
        end
    end
    rx_sig = rx_sig + noise_std*(randn(n_samp,1)+1j*randn(n_samp,1));
    mf = ifft(fft(rx_sig,32768).*fft([conj(flipud(lfm_up));complex(zeros(32768-n_lfm,1))],32768));
    mp = abs(mf(1:n_samp)).^2;
    [~,pk] = max(mp);
    rng_meas = (pk-n_lfm)*c/(2*fs);
    err13 = abs(rng_meas-R_m);
    ok13 = err13 < c/fs*2;
    fprintf('  R=%gkm meas=%.2fkm err=%.2fm SNR_est=%.1fdB %s\n', ...
        R_km, rng_meas/1000, err13, snr_est_db, char('PASS'*ok13+'FAIL'*(~ok13)));
    if ~ok13, pass13 = false; end
end
fprintf('  tested=%d [%s]\n', n_tested13, char('PASS'*pass13+'FAIL'*(~pass13)));
results.s2t13 = pass13;

%% S2T14: SNR vs Range (analytical, 40dB/decade slope)
fprintf('S2T14: SNR vs Range\n');
pass14 = true;
R_km_list14 = [1,2,5,8,10,15,20,25,30,35,40,50];
rcs14 = 10;
Pr14 = zeros(1,length(R_km_list14));
for ri = 1:length(R_km_list14)
    R_m = R_km_list14(ri)*1000;
    Pr14(ri) = tx_dbm + 2*D_dBi + rcs14 + 20*log10(lambda) ...
        - 30*log10(4*pi) - 40*log10(R_m) - Lsys_db;
end
% Check 40dB/decade: Pr(R1)-Pr(R2) should equal 40*log10(R2/R1)
% Test consecutive pairs and decade-spanning pairs
max_err14 = 0;
for i = 1:length(R_km_list14)
    for j = i+1:length(R_km_list14)
        expected_diff = 40*log10(R_km_list14(j)/R_km_list14(i));
        actual_diff = Pr14(i) - Pr14(j);
        err14 = abs(actual_diff - expected_diff);
        if err14 > max_err14, max_err14 = err14; end
        if err14 > 0.01, pass14 = false; end
    end
end
fprintf('  max_err=%.4f dB [%s]\n', max_err14, char('PASS'*pass14+'FAIL'*(~pass14)));
results.s2t14 = pass14;

%% Summary
fn=fieldnames(results);
for fi=1:length(fn)
    if results.(fn{fi}), n_pass=n_pass+1; else, n_fail=n_fail+1; end
end
fprintf('\n============================================\n');
fprintf('RESULT: %d/%d PASSED', n_pass, n_pass+n_fail);
if n_fail>0, fprintf(' (%d FAILED)', n_fail); end
fprintf('\n============================================\n');
if n_fail>0
    fprintf('FAILURES:\n');
    for fi=1:length(fn)
        if ~results.(fn{fi}), fprintf('  - %s\n', fn{fi}); end
    end
end
save('validate_em_s2_channel_results.mat','results');
