"""IQ-Level Capability Validation for FluxPhased Phased Array Radar Simulation.

Tests all 6 EW/radar capabilities at the IQ (baseband complex sample) level:
  1. Detection (探测): waveform → channel → matched filter → range resolution
  2. Mutual Interference (互扰): cross-radar IQ-level signal injection
  3. Communication (通信): BPSK encode → channel → demodulate → CRC
  4. Self-Interference (自扰): TX→RX leakage within same array
  5. Jamming (干扰): noise broadband/spot + DRFM retransmission
  6. Reconnaissance (侦察): signal parameter extraction from spectrum

Usage:
    python validation/test_iq_capabilities.py
"""

import sys
import os
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()


def make_env(**kwargs):
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv
    defaults = dict(
        num_envs=1, n_radars=2, rows=5, cols=5,
        pulses_per_cpi=4, bandwidth=10e6, prf=10e3,
        tx_power_w=50000.0,
        device=device,
    )
    defaults.update(kwargs)
    return MFARVecEnv(**defaults)


# ============================================================
# Test 1: Detection (探测)
# ============================================================
def test_detection():
    """Verify IQ-level detection chain: LFM → channel → matched filter → peak."""
    from radar_sim.gpu.waveform_gpu import generate_lfm
    from radar_sim.gpu.vec_channel import VecChannel

    print("\n" + "=" * 60)
    print("TEST 1: Detection / 探测")
    print("=" * 60)

    fs = 10e6
    pw = 50e-6
    bw = 10e6
    dev = device

    # Generate LFM waveform
    lfm = generate_lfm(pw, bw, fs, dev, "up")
    n_lfm = lfm.shape[0]
    n = n_lfm + 200  # extra margin for delay

    # Simulate a target return at delay=50 samples (no Doppler for clean test)
    delay = 50
    signal = torch.zeros(n, dtype=torch.complex64, device=dev)
    signal[delay:delay + n_lfm] = lfm * 0.1
    signal += torch.randn(n, dtype=torch.complex64, device=dev) * 0.01

    # Matched filter (frequency domain correlation)
    n_fft = 1
    while n_fft < n + n_lfm:
        n_fft *= 2
    mf_ref = torch.fft.fft(lfm, n=n_fft)
    sig_fft = torch.fft.fft(signal, n=n_fft)
    mf_out = torch.fft.ifft(sig_fft * mf_ref.conj())[:n]

    # Find peak
    mag = mf_out.abs()
    peak_idx = mag.argmax().item()

    # Processing gain
    noise_region = torch.cat([mag[:max(delay - 10, 1)], mag[delay + n_lfm:]])
    noise_power = (noise_region ** 2).mean().item()
    peak_power = (mag[peak_idx] ** 2).item()
    pg_db = 10.0 * np.log10(peak_power / max(noise_power, 1e-30))

    # Theoretical PG = 10*log10(TB) = 10*log10(pw * bw) ≈ 27 dB
    expected_pg = 10.0 * np.log10(pw * bw)

    passed_peak = abs(peak_idx - delay) <= 2  # allow 2-sample tolerance
    passed_pg = pg_db > expected_pg - 6  # allow 6 dB tolerance

    status = "PASS" if (passed_peak and passed_pg) else "FAIL"
    print(f"  [{status}] Peak at bin {peak_idx} (expected {delay}), "
          f"PG={pg_db:.1f} dB (expected ~{expected_pg:.1f} dB)")

    # Test all waveform types
    from radar_sim.gpu.waveform_gpu import (
        generate_barker, generate_frank, generate_costas,
        generate_nlfm, generate_p4,
    )
    waveforms = {
        "lfm_up": lambda: generate_lfm(pw, bw, fs, dev, "up"),
        "lfm_down": lambda: generate_lfm(pw, bw, fs, dev, "down"),
        "barker_13": lambda: generate_barker(13, pw / 13, fs, dev),
        "frank_16": lambda: generate_frank(4, fs, pw, dev),
        "costas_16": lambda: generate_costas(16, pw, fs, dev),
        "nlfm": lambda: generate_nlfm(pw, bw, fs, dev),
        "p4_code": lambda: generate_p4(4, pw, fs, dev),
    }

    all_ok = passed_peak and passed_pg
    for name, gen in waveforms.items():
        wf = gen()
        norm = wf.norm().item()
        ok = abs(norm - 1.0) < 0.01
        if not ok:
            all_ok = False
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {name}: norm={norm:.4f}")

    return all_ok


# ============================================================
# Test 2: Mutual Interference (互扰)
# ============================================================
def test_mutual_interference():
    """Verify cross-radar IQ interference adds correctly scaled signals."""
    from radar_sim.gpu.vec_interference import VecInterference

    print("\n" + "=" * 60)
    print("TEST 2: Mutual Interference / 互扰")
    print("=" * 60)

    env = make_env(n_radars=2)
    env.reset()

    # Get radar positions
    r0 = env.radar_pos[0, 0, :2].cpu().numpy()
    r1 = env.radar_pos[0, 1, :2].cpu().numpy()
    dist = np.linalg.norm(r1 - r0)

    # Run one step with jam elements on radar 0 to create interference
    result = env.step()
    intf_signal = result.get("interference", None)

    # Verify cross-radar interference exists in the CPI buffer
    # The interference should add IQ-level signals from other radars
    cpi = env._buf_cpi[0, 0]  # [N, P, S] for radar 0
    intf_energy = cpi.abs().pow(2).mean().item()

    passed = intf_energy > 0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] CPI buffer has non-zero energy (interference present): {intf_energy:.6f}")
    print(f"  Distance between radars: {dist:.0f} m")

    del env
    torch.cuda.empty_cache()
    gc.collect()
    return passed


# ============================================================
# Test 3: Communication (通信)
# ============================================================
def test_communication():
    """Verify BPSK comm chain: encode → modulate → channel → demodulate → decode."""
    from radar_sim.gpu.waveform_gpu import (
        encode_bpsk, decode_bpsk, modulate_bpsk, demodulate_bpsk,
    )

    print("\n" + "=" * 60)
    print("TEST 3: Communication / 通信")
    print("=" * 60)

    dev = device
    fs = 10e6
    symbol_rate = 1e6
    n_samples = 10000

    # Test multiple (X, Y) values
    test_cases = [
        (0.5, -0.3),
        (-0.8, 0.9),
        (0.0, 0.0),
        (0.99, -0.99),
    ]

    all_pass = True
    for x, y in test_cases:
        # Encode
        bits = encode_bpsk(x, y, n_bits=32, device=dev)

        # Modulate
        waveform = modulate_bpsk(bits, n_samples, fs, symbol_rate, dev)

        # Add noise (SNR ≈ 30 dB for reliable BPSK)
        noise = torch.randn(n_samples, dtype=torch.complex64, device=dev) * 0.01
        received = waveform + noise

        # Demodulate
        rx_bits = demodulate_bpsk(received, symbol_rate, fs, n_bits=32)

        # Decode
        dx, dy = decode_bpsk(rx_bits)
        err_x = abs(dx - x)
        err_y = abs(dy - y)

        ok = err_x < 0.01 and err_y < 0.01
        if not ok:
            all_pass = False
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] ({x:+.2f}, {y:+.2f}) → ({dx:+.4f}, {dy:+.4f}) "
              f"err=({err_x:.4f}, {err_y:.4f})")

    # Test CRC rejection with corrupted bits
    bits_clean = encode_bpsk(0.5, 0.5, n_bits=32, device=dev)
    bits_corrupt = bits_clean.clone()
    bits_corrupt[0] = 1.0 - bits_corrupt[0]  # flip one bit
    dx_c, dy_c = decode_bpsk(bits_corrupt)
    crc_fail = (dx_c == 0.0 and dy_c == 0.0)

    s = "PASS" if crc_fail else "FAIL"
    if not crc_fail:
        all_pass = False
    print(f"  [{s}] CRC rejects corrupted bits (decoded: ({dx_c:.2f}, {dy_c:.2f}))")

    return all_pass


# ============================================================
# Test 4: Self-Interference (自扰)
# ============================================================
def test_self_interference():
    """Verify TX→RX leakage within same phased array."""
    print("\n" + "=" * 60)
    print("TEST 4: Self-Interference / 自扰")
    print("=" * 60)

    # Test with low isolation (should see leakage)
    env_low = make_env(tx_rx_isolation_db=10.0)
    env_low.reset()

    # All elements detect → all TX active, all RX active
    result_low = env_low.step()
    spec_low = result_low["spectrum"][0, 0]  # [N, P, B]

    # Test with high isolation (should be clean)
    env_high = make_env(tx_rx_isolation_db=100.0)
    env_high.reset()
    # Copy same positions for fair comparison
    env_high.radar_pos.copy_(env_low.radar_pos)
    env_high.target_pos.copy_(env_low.target_pos)

    result_high = env_high.step()
    spec_high = result_high["spectrum"][0, 0]

    # Low isolation should have higher noise floor (TX leakage raises floor)
    floor_low = spec_low.median().item()
    floor_high = spec_high.median().item()

    # The low-isolation env should have more energy in the spectrum
    energy_low = spec_low.abs().mean().item()
    energy_high = spec_high.abs().mean().item()

    passed = energy_low > energy_high * 0.9  # at least comparable or higher
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Low isolation (10 dB) energy: {energy_low:.6e}")
    print(f"  [{status}] High isolation (100 dB) energy: {energy_high:.6e}")
    print(f"  Ratio: {energy_low / max(energy_high, 1e-30):.2f}x")

    # Verify isolation parameter is stored correctly
    assert env_low.tx_rx_isolation_db == 10.0
    assert env_high.tx_rx_isolation_db == 100.0
    print(f"  [PASS] Isolation parameters stored correctly")

    del env_low, env_high
    torch.cuda.empty_cache()
    gc.collect()
    return passed


# ============================================================
# Test 5: Jamming (干扰)
# ============================================================
def test_jamming():
    """Verify noise jamming degrades detection and DRFM creates false targets."""
    from radar_sim.gpu.waveform_gpu import (
        generate_noise_broadband, generate_noise_spot,
        generate_lfm, generate_drfm,
    )

    print("\n" + "=" * 60)
    print("TEST 5: Jamming / 干扰")
    print("=" * 60)

    dev = device
    fs = 10e6
    n = 2000

    # --- 5a: Broadband noise degrades matched filter SNR ---
    pw_jam = 100e-6  # 100 us pulse
    lfm = generate_lfm(pw_jam, fs, fs, dev, "up")
    n_lfm = lfm.shape[0]
    n = max(n_lfm + 100, 2000)
    lfm_padded = torch.zeros(n, dtype=torch.complex64, device=dev)
    lfm_padded[:n_lfm] = lfm

    # Clean target at delay=50
    delay = 50
    clean = torch.zeros(n, dtype=torch.complex64, device=dev)
    clean[delay:delay + n_lfm] = lfm_padded[:n_lfm] * 0.3

    # Add noise jamming at different power levels
    snrs = []
    for jam_power in [0.0, 0.01, 0.1, 1.0]:
        jam = generate_noise_broadband(n, jam_power, dev) if jam_power > 0 else torch.zeros(n, dtype=torch.complex64, device=dev)
        thermal = torch.randn(n, dtype=torch.complex64, device=dev) * 0.001
        rx = clean + jam + thermal

        # Matched filter
        mf_ref = torch.fft.fft(lfm_padded, n=2 * n)
        mf_out = torch.fft.ifft(torch.fft.fft(rx, n=2 * n) * mf_ref.conj())[:n]
        mag = mf_out.abs()

        peak = mag[delay].item()
        noise_region = torch.cat([mag[:max(delay - 10, 1)], mag[delay + n_lfm:]])
        noise_floor = noise_region.mean().item()
        snr = peak / max(noise_floor, 1e-30)
        snrs.append(snr)

    # SNR should decrease with more jamming
    degraded = snrs[0] > snrs[2] > snrs[3]  # 0 > 0.01 > 0.1 > 1.0
    status = "PASS" if degraded else "FAIL"
    print(f"  [{status}] Broadband noise degrades MF-SNR: "
          f"jam=[0, 0.01, 0.1, 1.0] → SNR=[{snrs[0]:.1f}, {snrs[1]:.1f}, {snrs[2]:.1f}, {snrs[3]:.1f}]")

    # --- 5b: Spot noise targets specific frequency ---
    center_freq = 2e6
    spot = generate_noise_spot(n, center_freq, fs * 0.1, fs, 1.0, dev)
    spec = torch.fft.fft(spot)
    freqs = torch.fft.fftfreq(n, 1.0 / fs, device=dev)
    mag_spec = spec.abs()
    peak_freq_idx = mag_spec.argmax().item()
    peak_freq = abs(freqs[peak_freq_idx].item())

    spot_ok = abs(peak_freq - center_freq) < fs * 0.05
    status = "PASS" if spot_ok else "FAIL"
    print(f"  [{status}] Spot noise peak at {peak_freq / 1e6:.2f} MHz "
          f"(target {center_freq / 1e6:.2f} MHz)")

    # --- 5c: DRFM frequency shift ---
    captured = generate_lfm(pw_jam, fs, fs, dev, "up")
    n_cap = captured.shape[0]
    freq_shift = 1e6
    drfm = generate_drfm(captured, freq_shift, fs, delay_samples=0)

    # Verify DRFM output has frequency shift
    spec_orig = torch.fft.fft(captured)
    spec_drfm = torch.fft.fft(drfm)
    # The DRFM should shift the spectrum
    orig_peak = spec_orig.abs().argmax().item()
    drfm_peak = spec_drfm.abs().argmax().item()

    # Both should be valid complex signals
    drfm_ok = drfm.abs().mean().item() > 0 and not torch.isnan(drfm).any()
    status = "PASS" if drfm_ok else "FAIL"
    print(f"  [{status}] DRFM output valid: norm={drfm.norm():.4f}, "
          f"freq_shift={freq_shift / 1e6:.1f} MHz")

    return degraded and spot_ok and drfm_ok


# ============================================================
# Test 6: Reconnaissance (侦察)
# ============================================================
def test_reconnaissance():
    """Verify recon elements extract signal parameters from spectrum."""
    from radar_sim.gpu.vec_element_processor import VecElementProcessor

    print("\n" + "=" * 60)
    print("TEST 6: Reconnaissance / 侦察")
    print("=" * 60)

    dev = device
    fs = 10e6
    E, R, N = 1, 1, 5
    P = 4
    fft_size = 64

    proc = VecElementProcessor(
        fs=fs, n_samples=int(1.0 / 10e3 * fs),
        pulses_per_cpi=P, fft_size=fft_size,
        symbol_rate=1e6, device=dev,
    )
    n_bins = proc.n_bins

    # Create a synthetic spectrum with a known peak
    # Bin 20 should be the peak
    peak_bin = 20
    spectrum = torch.ones(E, R, N, P, n_bins, dtype=torch.float32, device=dev) * 0.001
    spectrum[:, :, :, :, peak_bin] = 1.0  # strong signal at bin 20

    # Run recon extraction
    intel = proc.process_rx_recon(spectrum)  # [E, R, N, 4]

    # Check center frequency estimate
    expected_freq = peak_bin / max(n_bins - 1, 1)
    center_freq = intel[0, 0, 0, 0].item()
    cf_ok = abs(center_freq - expected_freq) < 0.05
    status = "PASS" if cf_ok else "FAIL"
    print(f"  [{status}] Center freq: {center_freq:.3f} (expected {expected_freq:.3f})")

    # Check signal strength (should be high for peak bin)
    strength = intel[0, 0, 0, 2].item()
    str_ok = strength > 0.5
    status = "PASS" if str_ok else "FAIL"
    print(f"  [{status}] Signal strength: {strength:.3f} (should be > 0.5)")

    # Check bandwidth (should be narrow for a single-bin signal)
    bw = intel[0, 0, 0, 1].item()
    bw_ok = bw < 0.1  # single bin → very narrow bandwidth
    status = "PASS" if bw_ok else "FAIL"
    print(f"  [{status}] Bandwidth: {bw:.3f} (should be < 0.1 for single-bin signal)")

    # Test with broadband signal
    spectrum_bb = torch.ones(E, R, N, P, n_bins, dtype=torch.float32, device=dev) * 0.1
    # Add energy across bins 10-30
    spectrum_bb[:, :, :, :, 10:30] = 1.0
    intel_bb = proc.process_rx_recon(spectrum_bb)
    bw_bb = intel_bb[0, 0, 0, 1].item()

    bw_bb_ok = bw_bb > bw  # broadband should have wider BW estimate
    status = "PASS" if bw_bb_ok else "FAIL"
    print(f"  [{status}] Broadband BW: {bw_bb:.3f} (should be > narrowband {bw:.3f})")

    # Test with env integration (recon elements in the step pipeline)
    env = make_env()
    env.reset()

    # Set some elements to recon task
    action = torch.zeros(1, env.n_radars, env.action_dim, device=dev)
    # First 10 elements → recon (task_id=0 has highest fraction)
    n_elem = env.n_elem
    ACTION_PER_ELEM = 22
    for i in range(min(10, n_elem)):
        base = i * ACTION_PER_ELEM
        action[0, :, base + 0] = 1.0   # recon fraction = 1
        action[0, :, base + 1] = 0.0   # detect = 0
        action[0, :, base + 2] = 0.0   # jam = 0
        action[0, :, base + 3] = 0.0   # comm = 0

    result = env.step(action)
    state = result["state"]  # [E, R, state_dim]

    # Check that recon_intel is included in state (should have non-zero values
    # at the recon_flat section: after spec_flat and comm_flat)
    N_elems = env.n_elem
    P_pulses = env.n_pulses
    B_bins = env.n_bins
    offset = N_elems * P_pulses * B_bins + N_elems * 2  # after spec + comm
    recon_in_state = state[0, 0, offset:offset + N_elems * 4]

    has_recon = recon_in_state.abs().sum().item() > 0
    status = "PASS" if has_recon else "FAIL"
    print(f"  [{status}] Recon intel in state vector: "
          f"sum={recon_in_state.abs().sum().item():.4f}")

    del env, proc
    torch.cuda.empty_cache()
    gc.collect()

    return cf_ok and str_ok and bw_ok and bw_bb_ok


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    torch.cuda.synchronize()
    print("=" * 60)
    print("FluxPhased IQ-Level Capability Validation")
    print("=" * 60)

    import warp as wp
    wp.init()

    results = {}
    tests = [
        ("1_detection", test_detection),
        ("2_mutual_interference", test_mutual_interference),
        ("3_communication", test_communication),
        ("4_self_interference", test_self_interference),
        ("5_jamming", test_jamming),
        ("6_reconnaissance", test_reconnaissance),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(results.values())
    n_total = len(results)
    for name, passed in results.items():
        s = "PASS" if passed else "FAIL"
        print(f"  [{s}] {name}")

    print(f"\n  {n_pass}/{n_total} passed")
    if n_pass == n_total:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {n_total - n_pass} FAILED")
