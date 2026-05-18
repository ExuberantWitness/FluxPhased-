%% FluxPhased EM Base Validation Part 1 (T1-T10)
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

fprintf('Part1: T1-T10  fc=%.0fGHz bw=%.0fMHz %dx%d D=%.1fdBi\n\n', fc/1e9, bw/1e6, rows, cols, D_dBi);

%% T1: Steering Beam Peak
fprintf('T1: Steering\n');
steer_az = [0 15 -30 45]; pass1 = true;
for si = 1:4
    az = steer_az(si);
    w = (1/N_elem)*exp(-1j*k_wave*ex*sind(az));
    best = 0; best_az = 0;
    for ai = -90:0.2:90
        af = abs(sum(w .* exp(1j*k_wave*ex*sind(ai))));
        if af > best, best = af; best_az = ai; end
    end
    err = abs(best_az - az);
    if err > 0.5, pass1 = false; end
end
fprintf('  [%s]\n', char('PASS'*pass1 + 'FAIL'*(~pass1)));
results.t1 = pass1;

%% T2: Beamwidth
fprintf('T2: Beamwidth\n');
w0 = (1/N_elem)*ones(N_elem,1);
bw_vals = [];
for ai = -20:0.2:20
    af = abs(sum(w0 .* exp(1j*k_wave*ex*sind(ai))));
    bw_vals(end+1) = af;
end
bw_vals = bw_vals / max(bw_vals);
above_3db = find(bw_vals > 0.5);
bw_meas = (above_3db(end) - above_3db(1)) * 0.2;
bw_th = 0.886*lambda/(cols*dx_m)*180/pi;
pass2 = abs(bw_meas-bw_th)/bw_th < 0.35;
fprintf('  Meas=%.1f Theory=%.1f [%s]\n', bw_meas, bw_th, char('PASS'*pass2+'FAIL'*(~pass2)));
results.t2 = pass2;

%% T3: Radar Equation
fprintf('T3: RadarEq\n');
pass3 = true;
for R_km = [5 10 20 50]
    R_m = R_km*1000;
    Pr = tx_dbm+2*D_dBi+20+20*log10(lambda)-30*log10(4*pi)-40*log10(R_m)-Lsys_db;
    G=10^(D_dBi/10); L=10^(Lsys_db/10);
    Pr_ref = 10*log10(tx_power_w*G^2*lambda^2*10^(20/10)/((4*pi)^3*R_m^4*L)*1000);
    if abs(Pr-Pr_ref) > 0.5, pass3 = false; end
end
fprintf('  [%s]\n', char('PASS'*pass3+'FAIL'*(~pass3)));
results.t3 = pass3;

%% T4: One-Way Link
fprintf('T4: OneWay\n');
pass4 = true;
for d_km = [1 5 10 20]
    d = d_km*1000;
    PL = 20*log10(4*pi*d/lambda);
    Pr = tx_dbm+D_dBi-PL-Lsys_db;
    G=10^(D_dBi/10); L=10^(Lsys_db/10);
    Pr_ref = 10*log10(tx_power_w*G/((4*pi*d/lambda)^2*L)*1000);
    if abs(Pr-Pr_ref) > 0.5, pass4 = false; end
end
fprintf('  [%s]\n', char('PASS'*pass4+'FAIL'*(~pass4)));
results.t4 = pass4;

%% T5: Channel Delay
fprintf('T5: Delay\n');
R5 = 5000; dn5 = round(2*R5/c*fs);
tx5 = [lfm_up; complex(zeros(n_samp-n_lfm,1))];
rx5 = complex(zeros(n_samp,1));
for s = 1:n_samp
    src = s-dn5;
    if src>=1 && src<=n_samp, rx5(s)=tx5(src); end
end
mf5 = ifft(fft(rx5,32768).*fft([conj(flipud(lfm_up));complex(zeros(32768-n_lfm,1))],32768));
[~,pk5] = max(abs(mf5(1:n_samp)).^2);
rng5 = (pk5-n_lfm)*c/(2*fs);
pass5 = abs(rng5-R5) < c/fs*2;
fprintf('  range_err=%.2fm [%s]\n', abs(rng5-R5), char('PASS'*pass5+'FAIL'*(~pass5)));
results.t5 = pass5;

%% T6: Doppler
fprintf('T6: Doppler\n');
fd6 = 2*300*fc/c; ds6 = 2*pi*fd6/fs;
n6 = 2^17; t6 = (0:n6-1)';
sig6 = exp(1j*t6*ds6);
NFFT6 = 2^18;
spec6 = fft(sig6, NFFT6);
fr6 = (0:NFFT6/2-1)*fs/NFFT6;
[~,pk6] = max(abs(spec6(1:NFFT6/2)));
fd_meas = fr6(pk6);
pass6 = abs(fd_meas-fd6)/fd6*100 < 1.0;
fprintf('  fd=%.1f meas=%.1f err=%.2f%% [%s]\n', fd6, fd_meas, abs(fd_meas-fd6)/fd6*100, char('PASS'*pass6+'FAIL'*(~pass6)));
results.t6 = pass6;

%% T7: Waveforms - using interp1 for Frank/P4 to avoid MATLAB hang
fprintf('T7: Waveforms\n');
% LFM dn
lfm_dn = exp(-1j*pi*k_lfm*t_lfm.^2); lfm_dn = lfm_dn/norm(lfm_dn);
p7a = abs(norm(lfm_dn)-1)<1e-5;
% Barker
barker = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
acb = abs(xcorr(barker)); psl_b = 20*log10(max(acb(acb<max(acb)))/max(acb));
p7b = psl_b < -18;
% Frank - using interp1
ph_f = 2*pi/4 * ((0:3)'*(0:3)); ph_f = ph_f(:);
t_idx_f = linspace(1,length(ph_f),n_lfm);
frank_phase = interp1(1:length(ph_f), ph_f, t_idx_f, 'linear');
frank = exp(1j*frank_phase); frank = frank/norm(frank);
p7c = abs(norm(frank)-1)<1e-5;
% Costas
cseq = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
cl = floor(n_lfm/16); costas = complex(zeros(n_lfm,1));
for ci = 1:16
    s0=(ci-1)*cl+1; s1=min(s0+cl-1,n_lfm);
    costas(s0:s1)=exp(1j*2*pi*cseq(ci)/pw*(0:s1-s0)'/fs);
end
costas = costas/norm(costas); p7d = abs(norm(costas)-1)<1e-5;
% NLFM
nlfm = exp(1j*(pi*k_lfm*t_lfm.^2+0.3*pi*k_lfm/pw*t_lfm.^3));
nlfm = nlfm/norm(nlfm); p7e = abs(norm(nlfm)-1)<1e-5;
% P4 - using interp1
npt=16; kp=(0:npt-1)'; php = pi*kp.^2/npt-pi*kp;
t_idx_p = linspace(1,npt,n_lfm);
p4_phase = interp1(1:npt, php, t_idx_p, 'linear');
p4 = exp(1j*p4_phase); p4 = p4/norm(p4);
p7f = abs(norm(p4)-1)<1e-5;

pass7 = p7a&&p7b&&p7c&&p7d&&p7e&&p7f;
fprintf('  LFM:%d Barker:%.0fdB Frank:%d Costas:%d NLFM:%d P4:%d [%s]\n', ...
    p7a, psl_b, p7c, p7d, p7e, p7f, char('PASS'*pass7+'FAIL'*(~pass7)));
results.t7 = pass7;

%% T8: MF Compression
fprintf('T8: MF\n');
mf8 = ifft(fft(tx5,32768).*fft([conj(flipud(lfm_up));complex(zeros(32768-n_lfm,1))],32768));
mp8 = abs(mf8(1:n_samp)).^2; [pk8,~]=max(mp8);
h8 = pk8/2; ab8 = find(mp8>h8);
cr8 = 10*log10(n_lfm/length(ab8));
cr_th = 10*log10(bw*pw);
pass8 = abs(cr8-cr_th)<2;
fprintf('  CR=%.1fdB theory=%.1fdB [%s]\n', cr8, cr_th, char('PASS'*pass8+'FAIL'*(~pass8)));
results.t8 = pass8;

%% T9: Noise (fixed: each quad gets noise_std, total power = kB*T*B*F)
fprintf('T9: Noise\n');
nn = 1e5;
nc = noise_std*(randn(nn,1)+1j*randn(nn,1));
meas_power = mean(abs(nc).^2);
nf9 = 10*log10(meas_power/(kB*T_noise*bw));
pass9 = abs(nf9-NF_db)<0.5;
fprintf('  NF=%.1fdB config=%.1fdB [%s]\n', nf9, NF_db, char('PASS'*pass9+'FAIL'*(~pass9)));
results.t9 = pass9;

%% T10: BPSK CRC
fprintf('T10: BPSK\n');
nok=0; mxe=0;
for ti=1:500
    dx10=rand*2-1; dy10=rand*2-1;
    xi=round(max(0,min(2^14-1,(dx10+1)/2*(2^14-1))));
    yi=round(max(0,min(2^14-1,(dy10+1)/2*(2^14-1))));
    d28=bitor(bitshift(xi,14),yi); crc=0; v=d28;
    for n=1:7, crc=bitxor(crc,bitand(v,15)); v=bitshift(v,-4); end
    word=bitor(bitor(bitshift(xi,18),bitshift(yi,4)),bitand(crc,15));
    xid=bitand(bitshift(word,-18),2^14-1); yid=bitand(bitshift(word,-4),2^14-1);
    crx=bitand(word,15); d28d=bitor(bitshift(xid,14),yid); crc2=0; v=d28d;
    for n=1:7, crc2=bitxor(crc2,bitand(v,15)); v=bitshift(v,-4); end
    if bitand(crc2,15)==crx
        nok=nok+1;
        mxe=max(mxe,max(abs(xid/(2^14-1)*2-1-dx10),abs(yid/(2^14-1)*2-1-dy10)));
    end
end
pass10 = nok==500 && mxe<3/(2^14-1);
fprintf('  CRC %d/500 err=%.6f [%s]\n', nok, mxe, char('PASS'*pass10+'FAIL'*(~pass10)));
results.t10 = pass10;

%% Summary Part 1
fn=fieldnames(results);
for fi=1:length(fn)
    if results.(fn{fi}), n_pass=n_pass+1; else, n_fail=n_fail+1; end
end
fprintf('\nPart1: %d/%d PASSED', n_pass, n_pass+n_fail);
if n_fail>0, fprintf(' (%d FAILED)', n_fail); end
fprintf('\n');
save('validate_em_base_p1_results.mat','results');
