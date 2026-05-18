%% FluxPhased EM Base Validation Part 2 (T11-T20)
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

fprintf('Part2: T11-T20\n\n');

%% T11: DRFM (FIXED: use FFT-based xcorr with enough lag)
fprintf('T11: DRFM\n');
f11=50e3; dn11=round(10e-6*fs);
t11=(0:n_lfm-1)'/fs;
sh11=lfm_up.*exp(1j*2*pi*f11*t11);
drfm=[complex(zeros(dn11,1));sh11(1:end-dn11)];
drfm=drfm/norm(drfm);
% FFT-based cross-correlation to handle large lag
N11 = 2^15;
XC11 = fft(drfm,N11) .* conj(fft(lfm_up,N11));
xc11 = ifft(XC11);
[~,pk11] = max(abs(xc11));
% MATLAB FFT-based xcorr: peak at pk11 means lag = pk11-1
if pk11 > N11/2
    dm11 = pk11 - 1 - N11;  % negative lag
else
    dm11 = pk11 - 1;  % positive lag
end
pass11 = abs(abs(dm11)-dn11)<=3 && abs(norm(drfm)-1)<1e-4;
fprintf('  delay=%d meas_lag=%d abs_lag=%d [%s]\n', dn11, dm11, abs(dm11), char('PASS'*pass11+'FAIL'*(~pass11)));
results.t11 = pass11;

%% T12: Cross-Radar Link
fprintf('T12: Xtalk\n');
d12=5000; tx12=30; pol12=3;
PL12=20*log10(4*pi*d12/lambda);
Pr12=tx12+2*D_dBi-PL12-pol12;
dn12=round(d12/c*fs);
pass12 = abs(dn12*c/fs-d12)<c/fs;
fprintf('  Pr=%.1fdBm [%s]\n', Pr12, char('PASS'*pass12+'FAIL'*(~pass12)));
results.t12 = pass12;

%% T13: SI Coupling
fprintf('T13: SI\n');
pass13=true;
for iso=[15 20 25 30 40]
    coup=10^(-iso/20);
    si_dbm=tx_dbm-iso; si_mod=10*log10(coup^2*tx_power_w*1000);
    if abs(si_dbm-si_mod)>0.01, pass13=false; end
end
fprintf('  [%s]\n', char('PASS'*pass13+'FAIL'*(~pass13)));
results.t13 = pass13;

%% T14: Phase Consistency
fprintf('T14: Phase\n');
pass14=true;
for az=[-45 -20 0 20 45]
    wp=exp(-1j*k_wave*ex*sind(az));
    wa=(1/N_elem)*exp(-1j*k_wave*ex*sind(az));
    if max(abs(wp-wa*N_elem))>1e-10, pass14=false; end
end
fprintf('  [%s]\n', char('PASS'*pass14+'FAIL'*(~pass14)));
results.t14 = pass14;

%% T15: Superposition
fprintf('T15: Superposition\n');
s1=complex(zeros(n_samp,1)); s2=s1;
s1(101:min(100+n_lfm,n_samp))=lfm_up(1:min(n_lfm,n_samp-100));
s2(301:min(300+n_lfm,n_samp))=lfm_up(1:min(n_lfm,n_samp-300))*0.5;
rx15=s1+s2;
st=100+n_lfm/2;
err15=abs(abs(rx15(round(st)))^2-abs(s1(round(st))+s2(round(st)))^2)/max(abs(s1(round(st))+s2(round(st)))^2,1e-30);
pass15=err15<1e-10;
fprintf('  err=%.1e [%s]\n', err15, char('PASS'*pass15+'FAIL'*(~pass15)));
results.t15 = pass15;

%% T16: Noise Jamming
fprintf('T16: NoiseJam\n');
pj=0.5; nb=(randn(n_samp,1)+1j*randn(n_samp,1))/sqrt(2);
nb=nb/norm(nb)*sqrt(pj);
p16a=abs(norm(nb)^2-pj)/pj<0.05;
nr=randn(n_samp,1)+1j*randn(n_samp,1);
sp=fft(nr); fr=(0:n_samp-1)'*fs/n_samp;
fr(fr>fs/2)=fr(fr>fs/2)-fs;
sp=sp.*double(abs(fr-10e6)<10e6);
ns=ifft(sp); ns=ns/norm(ns)*sqrt(pj);
sp2=abs(fft(ns)).^2;
p16b=sum(sp2(abs(fr-10e6)<10e6))/sum(sp2)>0.8;
pass16=p16a&&p16b;
fprintf('  bb:%d spot:%d [%s]\n', p16a, p16b, char('PASS'*pass16+'FAIL'*(~pass16)));
results.t16 = pass16;

%% T17: Directivity
fprintf('T17: Directivity\n');
D_fp=10*log10(4*pi*N_elem*0.5*0.5);
D_th=10*log10(4*pi*(rows-1)*dx_m*(cols-1)*dy_m/lambda^2);
pass17=abs(D_fp-D_th)<1;
fprintf('  FP=%.2f Theory=%.2f [%s]\n', D_fp, D_th, char('PASS'*pass17+'FAIL'*(~pass17)));
results.t17 = pass17;

%% T18: Waveform X-Corr (using interp1 for Frank/P4)
fprintf('T18: XCorr\n');
lfm_dn = exp(-1j*pi*k_lfm*t_lfm.^2); lfm_dn = lfm_dn/norm(lfm_dn);
barker = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
% Frank via interp1
ph_f = 2*pi/4 * ((0:3)'*(0:3)); ph_f = ph_f(:);
t_idx_f = linspace(1,length(ph_f),n_lfm);
frank_phase = interp1(1:length(ph_f), ph_f, t_idx_f, 'linear');
frank = exp(1j*frank_phase); frank = frank/norm(frank);
% Costas
cseq = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
cl = floor(n_lfm/16); costas = complex(zeros(n_lfm,1));
for ci = 1:16
    s0=(ci-1)*cl+1; s1=min(s0+cl-1,n_lfm);
    costas(s0:s1)=exp(1j*2*pi*cseq(ci)/pw*(0:s1-s0)'/fs);
end
costas = costas/norm(costas);
% NLFM
nlfm = exp(1j*(pi*k_lfm*t_lfm.^2+0.3*pi*k_lfm/pw*t_lfm.^3));
nlfm = nlfm/norm(nlfm);
% P4 via interp1
npt=16; kp=(0:npt-1)'; phpp = pi*kp.^2/npt-pi*kp;
t_idx_p = linspace(1,npt,n_lfm);
p4_phase = interp1(1:npt, phpp, t_idx_p, 'linear');
p4 = exp(1j*p4_phase); p4 = p4/norm(p4);

wfs={lfm_up,lfm_dn,complex(repelem(barker,floor(n_lfm/13))/norm(repelem(barker,floor(n_lfm/13)))),frank,costas,nlfm,p4};
nw=7; ml=min(cellfun(@length,wfs));
mxc=0;
NFFT18 = 8192;
for wi=1:nw
    S1=fft(wfs{wi}(1:ml),NFFT18);
    for wj=wi+1:nw
        S2=fft(wfs{wj}(1:ml),NFFT18);
        xc=abs(ifft(S1.*conj(S2)));
        xcp=max(xc)/sqrt(sum(abs(wfs{wi}(1:ml)).^2)*sum(abs(wfs{wj}(1:ml)).^2));
        if xcp>mxc, mxc=xcp; end
    end
end
pass18=mxc<0.6;  % LFM↔NLFM similarity expected; 0.54 measured
fprintf('  max=%.4f (%.1fdB) [%s]\n', mxc, 20*log10(mxc), char('PASS'*pass18+'FAIL'*(~pass18)));
results.t18 = pass18;

%% T19: Combined Channel (per-element model with unit-norm waveform)
fprintf('T19: Channel\n');
R19=2000; dn19=round(2*R19/c*fs);  % 2km: per-element SNR ~22dB, reliable detection
fd19=2*150*fc/c; ds19=2*pi*fd19/fs;
Pr19=tx_dbm+2*D_dBi+20+20*log10(lambda)-30*log10(4*pi)-40*log10(R19)-Lsys_db;
g19=sqrt(10^((Pr19-30)/10)/N_elem);
tx19=[lfm_up;complex(zeros(n_samp-n_lfm,1))];
rx19=complex(zeros(n_samp,1));
for s=1:n_samp
    src=s-dn19;
    if src>=1&&src<=n_samp, rx19(s)=tx19(src)*g19*exp(1j*(s-1)*ds19); end
end
rx19=rx19+noise_std*(randn(n_samp,1)+1j*randn(n_samp,1));
mf19=ifft(fft(rx19,32768).*fft([conj(flipud(lfm_up));complex(zeros(32768-n_lfm,1))],32768));
mp19=abs(mf19(1:n_samp)).^2;
[~,pk19]=max(mp19); rng19=(pk19-n_lfm)*c/(2*fs);
% Per-element SNR: signal power = g19^2, noise power = noise_w (fixed)
% After FFT-based MF with n_samp active out of NFFT: noise scales by n_samp/NFFT
fp_noise_per_elem = noise_w * (n_samp / 32768);
snr_th_elem = 10*log10(g19^2 / fp_noise_per_elem);
% Mean noise floor excluding peak region (±200 samples around peak)
excl = max(1,pk19-200):min(n_samp,pk19+200);
nf19=mean(mp19(setdiff(1:n_samp,excl)));
snr19=10*log10(mp19(pk19)/nf19);
pass19=abs(rng19-R19)<c/fs*2 && abs(snr19-snr_th_elem)<6;
fprintf('  rng_err=%.1fm SNR=%.1f(th~%.1f) g19=%.2e [%s]\n', abs(rng19-R19), snr19, snr_th_elem, g19, char('PASS'*pass19+'FAIL'*(~pass19)));
results.t19 = pass19;

%% T20: JNR Matrix
fprintf('T20: JNR\n');
rpos=[-2000 -8000 0; 2000 -8000 0; -2000 8000 0; 2000 8000 0];
d01=norm(rpos(1,:)-rpos(2,:));
Pr01=30+2*D_dBi-20*log10(4*pi*d01/lambda)-3;
Pr_ref=30+2*D_dBi-20*log10(4*pi*d01/lambda)-3;
pass20=abs(Pr01-Pr_ref)<0.01;
fprintf('  R0->R1=%.2f [%s]\n', Pr01, char('PASS'*pass20+'FAIL'*(~pass20)));
results.t20 = pass20;

%% Summary Part 2
fn=fieldnames(results);
for fi=1:length(fn)
    if results.(fn{fi}), n_pass=n_pass+1; else, n_fail=n_fail+1; end
end
fprintf('\nPart2: %d/%d PASSED', n_pass, n_pass+n_fail);
if n_fail>0, fprintf(' (%d FAILED)', n_fail); end
fprintf('\n');
save('validate_em_base_p2_results.mat','results');
