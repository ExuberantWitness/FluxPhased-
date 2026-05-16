function [signal, mf_ref] = gen_lfm(pw, bw, fs, direction)
% Generate LFM chirp matching FluxPhased waveform_gpu.py:18-26
% direction: 'up' or 'down'
    n = max(1, floor(pw * fs));
    t = (0:n-1)' / fs;
    k = bw / pw;
    if strcmp(direction, 'down'), s = -1; else, s = 1; end
    phase = s * pi * k * t.^2;
    signal = exp(1j * phase);
    signal = signal / norm(signal);
    mf_ref = conj(signal);  % matched filter = conjugate (auto-corr)
end

function signal = gen_barker13(pw, fs)
% Generate Barker-13 matching FluxPhased waveform_gpu.py:29-40
    code = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
    n_chips = length(code);
    chip_width = pw / n_chips;
    spc = max(1, floor(chip_width * fs));
    samples = zeros(n_chips * spc, 1);
    for i = 1:n_chips
        samples((i-1)*spc+1 : i*spc) = code(i);
    end
    signal = complex(samples);
    signal = signal / norm(signal);
end

function signal = gen_frank16(pw, fs)
% Generate Frank-16 matching FluxPhased waveform_gpu.py:43-52
    M = 4; n_phases = M * M;
    phases = zeros(n_phases, 1);
    for i = 0:M-1
        for j = 0:M-1
            phases(i*M + j + 1) = 2*pi/M * i * j;
        end
    end
    n_target = floor(pw * fs);
    t_norm = linspace(1, n_phases, n_target)';
    phase_interp = interp1(1:n_phases, phases, t_norm, 'linear');
    signal = exp(1j * phase_interp);
    signal = signal / norm(signal);
end

function signal = gen_costas16(pw, fs)
% Generate Costas-16 matching FluxPhased waveform_gpu.py:55-73
    costas = [3 9 10 13 5 15 11 16 14 8 7 4 12 2 6 1];
    n_chips = length(costas);
    chip_dur = pw / n_chips;
    spc = max(1, floor(chip_dur * fs));
    n = n_chips * spc;
    signal = zeros(n, 1);
    for c = 1:n_chips
        fi = costas(c);  % 1-indexed
        t_chip = (0:spc-1)' / fs;
        idx_start = (c-1)*spc + 1;
        signal(idx_start:idx_start+spc-1) = exp(1j * 2*pi * (fi/pw) * t_chip);
    end
    signal = signal / norm(signal);
end

function signal = gen_nlfm(pw, bw, fs)
% Generate NLFM matching FluxPhased waveform_gpu.py:76-83
    n = max(1, floor(pw * fs));
    t = (0:n-1)' / fs;
    k = bw / pw;
    phase = pi * k * t.^2 + 0.3 * pi * k / pw * t.^3;
    signal = exp(1j * phase);
    signal = signal / norm(signal);
end

function signal = gen_p4(pw, fs)
% Generate P4 code matching FluxPhased waveform_gpu.py:86-95
    n_stages = 4;
    n_pts = n_stages * n_stages;
    phases = zeros(n_pts, 1);
    for k = 0:n_pts-1
        phases(k+1) = pi * k^2 / n_pts - pi * k;
    end
    n_target = floor(pw * fs);
    t_norm = linspace(1, n_pts, n_target)';
    phase_interp = interp1(1:n_pts, phases, t_norm, 'linear');
    signal = exp(1j * phase_interp);
    signal = signal / norm(signal);
end

function mf_out = mf_fft(signal, ref, n_out)
% Matched filter in frequency domain matching FluxPhased vec_receiver.py
% signal: received [n,1], ref: reference waveform [m,1], n_out: output length
    n_sig = length(signal);
    n_ref = length(ref);
    n_fft = 1;
    while n_fft < n_sig + n_ref - 1
        n_fft = n_fft * 2;
    end
    % FluxPhased: ifft(fft(signal) * conj(fft(ref)))
    sig_fft = fft(signal, n_fft);
    ref_fft = fft(ref, n_fft);
    mf_out = ifft(sig_fft .* conj(ref_fft));
    mf_out = mf_out(1:n_out);
end

function noise_bb = gen_noise_broadband(n, power)
% Broadband noise matching FluxPhased waveform_gpu.py:255-267
    noise_bb = (randn(n,1) + 1j*randn(n,1)) / sqrt(2);
    noise_bb = noise_bb / norm(noise_bb) * sqrt(power);
end

function noise_spot = gen_noise_spot(n, center_freq, bw_noise, fs, power)
% Spot noise matching FluxPhased waveform_gpu.py:270-294
    noise = randn(n,1) + 1j*randn(n,1);
    spectrum = fft(noise);
    freqs = (0:n-1)' / n * fs;
    mask = abs(freqs - center_freq) < bw_noise / 2;
    spectrum = spectrum .* mask;
    noise_spot = ifft(spectrum);
    noise_spot = noise_spot / norm(noise_spot) * sqrt(power);
end

function shifted = gen_drfm(captured, freq_shift, fs, delay_samples)
% DRFM matching FluxPhased waveform_gpu.py:297-320
    n = length(captured);
    t = (0:n-1)' / fs;
    shifted = captured .* exp(1j * 2*pi * freq_shift * t);
    if delay_samples > 0 && delay_samples < n
        shifted = [zeros(delay_samples,1); shifted(1:end-delay_samples)];
    end
    nz = norm(shifted);
    if nz > 1e-10
        shifted = shifted / nz;
    end
end

function bits = encode_bpsk_flux(data_x, data_y)
% BPSK encode matching FluxPhased waveform_gpu.py:102-125
% Layout: [X:14 bits | Y:14 bits | CRC:4 bits] = 32 bits MSB first
    x_int = round(max(0, min(1, (data_x + 1) / 2)) * (2^14 - 1));
    y_int = round(max(0, min(1, (data_y + 1) / 2)) * (2^14 - 1));
    data_28 = bitshift(uint32(x_int), 14) + uint32(y_int);
    % CRC: XOR of 7 nibbles
    crc = uint32(0);
    val = data_28;
    for i = 1:7
        crc = bitxor(crc, bitand(val, uint32(15)));
        val = bitshift(val, -4);
    end
    word = bitshift(uint32(x_int), 18) + bitshift(uint32(y_int), 4) + bitand(crc, uint32(15));
    % Extract bits MSB first
    bits = zeros(32, 1);
    for b = 0:31
        bits(32 - b) = double(bitget(word, b + 1));
    end
end

function [data_x, data_y, crc_ok] = decode_bpsk_flux(bits)
% BPSK decode matching FluxPhased waveform_gpu.py:128-156
    if length(bits) < 32
        data_x = 0; data_y = 0; crc_ok = false; return;
    end
    % Reconstruct word from bits MSB first
    word = uint32(0);
    for b = 0:31
        if bits(32 - b) > 0.5
            word = bitor(word, bitshift(uint32(1), b));
        end
    end
    x_int = double(bitand(bitshift(word, -18), uint32(2^14 - 1)));
    y_int = double(bitand(bitshift(word, -4), uint32(2^14 - 1)));
    crc_received = double(bitand(word, uint32(15)));
    % CRC check
    data_28 = bitshift(uint32(x_int), 14) + uint32(y_int);
    crc_computed = uint32(0);
    val = data_28;
    for i = 1:7
        crc_computed = bitxor(crc_computed, bitand(val, uint32(15)));
        val = bitshift(val, -4);
    end
    crc_ok = (bitand(crc_computed, uint32(15)) == uint32(crc_received));
    if crc_ok
        data_x = x_int / (2^14 - 1) * 2 - 1;
        data_y = y_int / (2^14 - 1) * 2 - 1;
    else
        data_x = 0; data_y = 0;
    end
end

function s = pass_str(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
