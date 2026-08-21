%% Debug T7: Find exact hang point
clear; close all; clc;

c = 299792458; fc = 10e9; lambda = c/fc;
bw = 200e6; fs = bw; pw = 50e-6;
n_lfm = floor(pw*fs);
k_lfm = bw/pw; t_lfm = (0:n_lfm-1)'/fs;
lfm_up = exp(1j*pi*k_lfm*t_lfm.^2); lfm_up = lfm_up/norm(lfm_up);

fprintf('n_lfm=%d\n', n_lfm);
fprintf('Starting T7 debug...\n');

% LFM dn
fprintf('  Computing LFM down...\n');
lfm_dn = exp(-1j*pi*k_lfm*t_lfm.^2); lfm_dn = lfm_dn/norm(lfm_dn);
p7a = abs(norm(lfm_dn)-1)<1e-5;
fprintf('  LFM dn done: p7a=%d\n', p7a);

% Barker
fprintf('  Computing Barker...\n');
barker = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
fprintf('  xcorr...\n');
acb = abs(xcorr(barker));
fprintf('  psl...\n');
psl_b = 20*log10(max(acb(acb<max(acb)))/max(acb));
p7b = psl_b < -18;
fprintf('  Barker done: psl=%.1fdB p7b=%d\n', psl_b, p7b);

% Frank
fprintf('  Computing Frank...\n');
ph_f = 2*pi/4 * ((0:3)'*(0:3)); ph_f = ph_f(:);
fprintf('  Frank phase seq: %d elements\n', length(ph_f));
tn = linspace(0,15,n_lfm);
fprintf('  tn range: %.1f to %.1f\n', tn(1), tn(end));
lo = min(floor(tn),14); fr = tn-lo;
plo = ph_f(lo+1); phi = ph_f(min(lo+1,15)+1);
frank = exp(1j*(plo+fr.*(phi-plo))); frank = frank/norm(frank);
p7c = abs(norm(frank)-1)<1e-5;
fprintf('  Frank done: p7c=%d\n', p7c);

% Costas
fprintf('  Computing Costas...\n');
cseq = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
cl = floor(n_lfm/16); costas = complex(zeros(n_lfm,1));
for ci = 1:16
    s0=(ci-1)*cl+1; s1=min(s0+cl-1,n_lfm);
    costas(s0:s1)=exp(1j*2*pi*cseq(ci)/pw*(0:s1-s0)'/fs);
end
costas = costas/norm(costas); p7d = abs(norm(costas)-1)<1e-5;
fprintf('  Costas done: p7d=%d\n', p7d);

% NLFM
fprintf('  Computing NLFM...\n');
nlfm = exp(1j*(pi*k_lfm*t_lfm.^2+0.3*pi*k_lfm/pw*t_lfm.^3));
nlfm = nlfm/norm(nlfm); p7e = abs(norm(nlfm)-1)<1e-5;
fprintf('  NLFM done: p7e=%d\n', p7e);

% P4
fprintf('  Computing P4...\n');
npt=16; kp=(0:npt-1)'; php = pi*kp.^2/npt-pi*kp;
tp = linspace(0,npt-1,n_lfm); lop=min(floor(tp),npt-2); frp=tp-lop;
p4 = exp(1j*(php(lop+1)+frp.*(php(min(lop+1,npt-2)+1)-php(lop+1))));
p4 = p4/norm(p4); p7f = abs(norm(p4)-1)<1e-5;
fprintf('  P4 done: p7f=%d\n', p7f);

pass7 = p7a&&p7b&&p7c&&p7d&&p7e&&p7f;
fprintf('T7 RESULT: LFM:%d Barker:%.0fdB Frank:%d Costas:%d NLFM:%d P4:%d => %s\n', ...
    p7a, psl_b, p7c, p7d, p7e, p7f, char('PASS'*pass7+'FAIL'*(~pass7)));
fprintf('T7 debug complete.\n');
