"""Element self-consistent algorithm: per-element FFT magnitude spectrum + BPSK demod.

Unified RX pipeline for all 4 tasks (detect/recon/jam/comm).
Detection and reconnaissance share the same FFT path — the only difference is
whether matched filtering is applied (detection knows its TX waveform, recon doesn't).
Jamming is TX-only (RX output = zeros). Communication adds BPSK demod on top.
"""

import torch
import numpy as np

from .waveform_gpu import (
    encode_bpsk, decode_bpsk, modulate_bpsk, demodulate_bpsk,
    generate_lfm, generate_barker, generate_frank, generate_costas,
    generate_nlfm, generate_p4,
    generate_noise_broadband, generate_noise_spot, generate_drfm,
)

SPEED_OF_LIGHT = 299792458.0

# Task IDs
TASK_RECON = 0
TASK_DETECT = 1
TASK_JAM = 2
TASK_COMM = 3

# Detection waveform lookup table
_DETECT_WAVEFORMS = {
    0: "lfm_up", 1: "lfm_down", 2: "barker_13",
    3: "frank_16", 4: "costas_16", 5: "nlfm", 6: "p4_code",
}

# Jamming waveform types
_JAM_NOISE_BROADBAND = 0
_JAM_NOISE_SPOT = 1
_JAM_DRFM = 2


class VecElementProcessor:
    """Batched per-element RX processing (FFT magnitude spectrum) and TX assembly.

    All operations are batched across [E, R, N] (envs × radars × elements).
    """

    def __init__(
        self, fs: float, n_samples: int, pulses_per_cpi: int = 32,
        fft_size: int = 0, symbol_rate: float = 1e6,
        device: str = "cuda",
    ):
        """
        Args:
            fs: sampling rate (Hz)
            n_samples: samples per pulse
            pulses_per_cpi: number of pulses in a CPI (temporal dimension)
            fft_size: FFT size (0 = auto = next power of 2 >= n_samples)
            symbol_rate: BPSK symbol rate for communication
            device: torch device
        """
        self.fs = fs
        self.n_samples = n_samples
        self.pulses_per_cpi = pulses_per_cpi
        self.device = device
        self.symbol_rate = symbol_rate

        if fft_size > 0:
            self.fft_size = fft_size
        else:
            self.fft_size = 1
            while self.fft_size < n_samples:
                self.fft_size *= 2

        self.n_bins = self.fft_size

    # ------------------------------------------------------------------
    # Perception uplink (RX): IQ → FFT magnitude spectrum + BPSK data
    # ------------------------------------------------------------------

    def process_rx_spectrum(
        self, iq_signal: torch.Tensor, waveform_ref: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute FFT magnitude spectrum for one pulse.

        Args:
            iq_signal: [..., S] complex64 IQ samples (any leading batch dims)
            waveform_ref: [..., S] or [S] complex64 reference waveform for
                          matched filtering (detection/comm). None for recon.
        Returns:
            [..., n_bins] float32 power spectrum (|FFT|²)
        """
        spectrum = torch.fft.fft(iq_signal, n=self.fft_size, dim=-1)

        if waveform_ref is not None:
            ref_spectrum = torch.fft.fft(waveform_ref, n=self.fft_size, dim=-1)
            spectrum = spectrum * torch.conj(ref_spectrum)

        return torch.abs(spectrum) ** 2

    def process_rx_cpi(
        self, iq_pulses: torch.Tensor, waveform_ref: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process full CPI: [E, R, N, P, S] → [E, R, N, P, n_bins].

        Args:
            iq_pulses: [E, R, N, P, S] complex64 per-element per-pulse IQ
            waveform_ref: [E, R, N, S] or broadcastable complex64 reference
        Returns:
            [E, R, N, P, n_bins] float32 power spectra
        """
        # FFT on last dim, matched filter if reference provided
        spectrum = torch.fft.fft(iq_pulses, n=self.fft_size, dim=-1)

        if waveform_ref is not None:
            ref_spectrum = torch.fft.fft(waveform_ref, n=self.fft_size, dim=-1)
            # Broadcast: ref may be [E,R,N,S] or [E,R,1,S] etc.
            spectrum = spectrum * torch.conj(ref_spectrum.unsqueeze(-2))

        return torch.abs(spectrum) ** 2

    def process_rx_cpi_unified(
        self, iq_pulses: torch.Tensor,
        waveform_refs: dict,
        iq_is_fft: bool = False,
        rx_beam_weights: torch.Tensor = None,
    ) -> dict:
        """Single FFT pass with per-task matched filtering.

        Args:
            iq_pulses: [E, R, N, P, S_or_bins] complex64
                If iq_is_fft=False: time-domain IQ samples, S = n_samples.
                If iq_is_fft=True: pre-computed FFT, S_or_bins = n_bins.
            waveform_refs: {task_id: ref_waveform_or_None}
                ref_waveform: broadcastable to [E, R, N, S] complex64
            iq_is_fft: True if iq_pulses is already FFT'd (streaming mode).
            rx_beam_weights: [E, R, N] complex64 — if provided, coherently
                beamform elements before |·|^2, giving +10*log10(N) SNR gain.
        Returns:
            {task_id: [E, R, N, P, n_bins] float32 power spectrum}
            task_id 0 (recon) with ref=None returns raw |FFT|^2.
        """
        raw_fft = iq_pulses if iq_is_fft else torch.fft.fft(
            iq_pulses, n=self.fft_size, dim=-1,
        )

        results = {}
        for task_id, ref in waveform_refs.items():
            if ref is None:
                spec = torch.abs(raw_fft) ** 2
            else:
                ref_spectrum = torch.fft.fft(ref, n=self.fft_size, dim=-1)
                mf = raw_fft * torch.conj(ref_spectrum)
                # Coherent RX beamforming: weight and sum over elements
                if rx_beam_weights is not None:
                    w = torch.conj(rx_beam_weights).unsqueeze(-1).unsqueeze(-1)
                    mf = (w * mf).sum(dim=2, keepdim=True)  # [E,R,1,P,B]
                # IFFT back to time domain for pulse compression
                mf_time = torch.fft.ifft(mf, dim=-1).clone()
                spec = torch.abs(mf_time) ** 2
            # If beamformed, expand back to N elements for task mask compatibility
            if rx_beam_weights is not None and spec.shape[2] == 1:
                spec = spec.expand(-1, -1, iq_pulses.shape[2], -1, -1)
            results[task_id] = spec

        return results

    def process_rx_comm(
        self, iq_pulses: torch.Tensor, waveform_ref: torch.Tensor,
    ) -> torch.Tensor:
        """BPSK demodulation for communication elements.

        Args:
            iq_pulses: [E, R, N, P, S] complex64 (comm elements only)
            waveform_ref: [S] or broadcastable complex64 BPSK reference
        Returns:
            [E, R, N, 2] float32 decoded (X, Y) per element
        """
        # Matched filter first pulse (comm repeats same data each pulse)
        first_pulse = iq_pulses[..., 0, :]  # [E, R, N, S]
        n_fft = 1
        while n_fft < self.n_samples * 2:
            n_fft *= 2
        mf_ref = torch.fft.fft(waveform_ref.conj().flip(0), n=n_fft)
        sig_fft = torch.fft.fft(first_pulse, n=n_fft, dim=-1)
        filtered = torch.fft.ifft(sig_fft * mf_ref, dim=-1)[..., :self.n_samples]

        # Sample symbols and demodulate
        sps = max(1, int(self.fs / self.symbol_rate))
        n_bits = 32
        indices = torch.arange(n_bits, device=filtered.device) * sps + sps // 2
        indices = indices.clamp(max=self.n_samples - 1)

        # [E, R, N, 32]
        symbols = filtered[..., indices]
        bits = (symbols.real > 0).float()  # hard decision

        # Decode each element's bits to (X, Y)
        shape = bits.shape[:-1]  # [E, R, N]
        xy = torch.zeros(*shape, 2, dtype=torch.float32, device=self.device)

        # Vectorized CRC check and decode
        # bits: [E, R, N, 32] — decode to integer, check CRC, extract X, Y
        # For efficiency, do batched decode
        flat_bits = bits.reshape(-1, 32)  # [E*R*N, 32]
        n_elem_total = flat_bits.shape[0]

        # Convert bits to integer
        powers = (2.0 ** torch.arange(31, -1, -1, dtype=torch.float32, device=self.device))
        words = (flat_bits * powers).sum(dim=-1).long()  # [E*R*N]

        # Extract fields: [X:14 | Y:14 | CRC:4]
        x_int = ((words >> 18) & ((1 << 14) - 1))
        y_int = ((words >> 4) & ((1 << 14) - 1))
        crc_received = words & 0xF

        # CRC check: 7 nibbles of (x_int << 14 | y_int)
        data_28 = (x_int << 14) | y_int
        crc_computed = torch.zeros_like(data_28)
        for shift in range(7):
            crc_computed ^= ((data_28 >> (shift * 4)) & 0xF)
        crc_ok = (crc_computed & 0xF) == crc_received
        data_x = x_int / (2**14 - 1) * 2.0 - 1.0
        data_y = y_int / (2**14 - 1) * 2.0 - 1.0

        # Zero out failed CRC
        data_x = data_x * crc_ok.float()
        data_y = data_y * crc_ok.float()

        xy_flat = torch.stack([data_x, data_y], dim=-1)  # [E*R*N, 2]
        return xy_flat.reshape(*shape, 2)

    def process_rx_recon(
        self, spectrum: torch.Tensor,
    ) -> torch.Tensor:
        """Extract signal parameters from recon element spectrum.

        For each recon element, estimates:
          - Center frequency (normalized peak bin)
          - Bandwidth (3dB width around peak, normalized)
          - Signal strength (peak power in dB, normalized to [0,1])
          - DOA hint (0 for now, placeholder for beam-comparison AOA)

        Args:
            spectrum: [E, R, N, P, n_bins] float32 power spectrum (recon elements)
        Returns:
            [E, R, N, 4] float32 recon intelligence per element
        """
        # Average over pulses: [E, R, N, n_bins]
        spec_avg = spectrum.mean(dim=-2)
        n_bins = spec_avg.shape[-1]

        # Peak bin per element
        peak_bin = spec_avg.argmax(dim=-1)  # [E, R, N]
        peak_power = spec_avg.max(dim=-1).values  # [E, R, N]

        # Center frequency (normalized to [0, 1])
        center_freq = peak_bin.float() / max(n_bins - 1, 1)

        # 3dB bandwidth: count bins above half-power around peak
        half_power = peak_power * 0.5  # [E, R, N]
        above_3db = (spec_avg > half_power.unsqueeze(-1)).sum(dim=-1).float()
        bw_norm = above_3db / max(n_bins, 1)  # normalized to [0, 1]

        # Signal strength (peak dB, normalized to [0, 1])
        eps = 1e-30
        peak_db = 10.0 * torch.log10(peak_power + eps)
        # Normalize: clip to [-60, 0] dB range, map to [0, 1]
        strength = ((peak_db + 60.0) / 60.0).clamp(0, 1)

        # DOA hint: placeholder 0 (requires multi-element beam comparison)
        doa_hint = torch.zeros_like(center_freq)

        return torch.stack([center_freq, bw_norm, strength, doa_hint], dim=-1)

    # ------------------------------------------------------------------
    # Decision downlink (TX): action → waveform × weight
    # ------------------------------------------------------------------

    def generate_waveform(
        self, task_id: int, waveform_type: int, params: torch.Tensor,
        n_samples: int, captured_signal: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate waveform for one task type.

        Args:
            task_id: 0=recon, 1=detect, 2=jam, 3=comm
            waveform_type: index into waveform lookup table
            params: task-specific parameters tensor
            n_samples: output waveform length
            captured_signal: [S] complex64 captured RX signal for DRFM
        Returns:
            [n_samples] complex64 waveform, or None for recon
        """
        dev = self.device

        if task_id == TASK_RECON:
            return None

        elif task_id == TASK_DETECT:
            wf_name = _DETECT_WAVEFORMS.get(waveform_type % 7, "lfm_up")
            pw = params[0].item() if params.numel() > 0 else 0.5
            bw = params[1].item() if params.numel() > 1 else 0.5
            # Map normalized [0,1] params to physical values
            pw = max(pw, 0.01) * 100e-6  # scale to [1μs, 100μs]
            bw = max(bw, 0.01) * self.fs  # scale to [fs*0.01, fs]

            gens = {
                "lfm_up": lambda: generate_lfm(pw, bw, self.fs, dev, "up"),
                "lfm_down": lambda: generate_lfm(pw, bw, self.fs, dev, "down"),
                "barker_13": lambda: generate_barker(13, pw / 13, self.fs, dev),
                "frank_16": lambda: generate_frank(4, self.fs, pw, dev),
                "costas_16": lambda: generate_costas(16, pw, self.fs, dev),
                "nlfm": lambda: generate_nlfm(pw, bw, self.fs, dev),
                "p4_code": lambda: generate_p4(4, pw, self.fs, dev),
            }
            wf = gens[wf_name]()
            # Pad or trim to n_samples
            if wf.shape[0] < n_samples:
                pad = torch.zeros(
                    n_samples - wf.shape[0], dtype=torch.complex64, device=dev,
                )
                wf = torch.cat([wf, pad])
            return wf[:n_samples]

        elif task_id == TASK_JAM:
            jam_type = waveform_type % 3
            if jam_type == _JAM_NOISE_BROADBAND:
                power = params[0].item() if params.numel() > 0 else 1.0
                return generate_noise_broadband(n_samples, power, dev)
            elif jam_type == _JAM_NOISE_SPOT:
                center = params[0].item() if params.numel() > 0 else 0.0
                power = params[1].item() if params.numel() > 1 else 1.0
                bw = params[2].item() if params.numel() > 2 else self.fs * 0.1
                return generate_noise_spot(
                    n_samples, center * self.fs, bw * self.fs, self.fs, power, dev,
                )
            else:  # DRFM
                if captured_signal is not None and captured_signal.norm() > 1e-10:
                    freq_shift = params[2].item() if params.numel() > 2 else 0.0
                    freq_shift_hz = freq_shift * self.fs * 0.01
                    power = params[0].item() if params.numel() > 0 else 1.0
                    drfm = generate_drfm(
                        captured_signal[:n_samples], freq_shift_hz, self.fs,
                    )
                    return drfm * (power ** 0.5)
                else:
                    power = params[0].item() if params.numel() > 0 else 1.0
                    return generate_noise_broadband(n_samples, power, dev)

        elif task_id == TASK_COMM:
            data_x = params[0].item() if params.numel() > 0 else 0.0
            data_y = params[1].item() if params.numel() > 1 else 0.0
            sym_rate = params[2].item() if params.numel() > 2 else self.symbol_rate
            sym_rate = sym_rate * self.fs * 0.01  # normalize to physical rate

            bits = encode_bpsk(
                float(np.clip(data_x, -1, 1)),
                float(np.clip(data_y, -1, 1)),
                device=dev,
            )
            return modulate_bpsk(bits, n_samples, self.fs, sym_rate, dev)

        return None

    def assemble_tx_per_element(
        self,
        task_ids: torch.Tensor,    # [E, R, N] int task per element
        beam_az: torch.Tensor,     # [E, R, N] float32 azimuth per element
        beam_el: torch.Tensor,     # [E, R, N] float32 elevation per element
        wf_types: torch.Tensor,    # [E, R, N] int waveform type per element
        detect_params: torch.Tensor,  # [E, R, N, 3]
        jam_params: torch.Tensor,     # [E, R, N, 3]
        comm_params: torch.Tensor,    # [E, R, N, 3]
        elem_x: torch.Tensor,      # [N] float32 element x positions (m)
        elem_y: torch.Tensor,      # [N] float32 element y positions (m)
        wavelength: float,
        n_samples: int,
        captured_signal: torch.Tensor = None,  # [E, R, S] for DRFM
    ) -> torch.Tensor:
        """Assemble per-element TX signals from action parameters.

        Instead of looping per-element, we batch-generate one waveform per
        unique (task_id, wf_type) combination and broadcast.

        Args:
            task_ids, beam_az, beam_el, wf_types: [E, R, N]
            detect/jam/comm_params: [E, R, N, 3]
            elem_x, elem_y: [N] element positions in meters
            wavelength: carrier wavelength
            n_samples: samples per pulse
        Returns:
            [E, R, N, S] complex64 TX signal per element
        """
        E, R, N = task_ids.shape
        dev = torch.device(self.device)
        k = 2.0 * np.pi / wavelength
        DEG2RAD = np.pi / 180.0

        # Phase weights for beam steering
        az_rad = beam_az.clamp(-90, 90) * DEG2RAD  # [E, R, N]
        el_rad = beam_el.clamp(-90, 90) * DEG2RAD
        u = torch.sin(az_rad) * torch.cos(el_rad)
        v = torch.sin(el_rad)
        ex = elem_x.view(1, 1, N)
        ey = elem_y.view(1, 1, N)
        phase = -k * (ex * u + ey * v)
        weights = torch.exp(1j * phase)  # [E, R, N] complex64

        # Generate waveforms per unique task/type combination
        # Pre-build a waveform bank: { (task, type): waveform [S] }
        wf_bank = {}
        # Per-radar DRFM waveforms: { (task, type, e, r): waveform [S] }
        wf_bank_drfm = {}
        for t in range(4):
            mask_t = (task_ids == t)
            if not mask_t.any():
                continue
            types_in_task = wf_types[mask_t].unique()
            for wt in types_in_task:
                wt_int = int(wt.item())
                jam_type = wt_int % 3
                if t == TASK_JAM and jam_type == _JAM_DRFM and captured_signal is not None:
                    # DRFM: generate per-radar waveform (different captured signals)
                    for e in range(E):
                        for r in range(R):
                            params_map = {
                                TASK_DETECT: detect_params[e, r],
                                TASK_JAM: jam_params[e, r],
                                TASK_COMM: comm_params[e, r],
                            }
                            params = params_map.get(t, torch.zeros(3, device=dev))
                            cap = captured_signal[e, r]
                            wf = self.generate_waveform(
                                t, wt_int, params, n_samples,
                                captured_signal=cap,
                            )
                            if wf is not None:
                                wf_bank_drfm[(t, wt_int, e, r)] = wf
                else:
                    params_map = {
                        TASK_DETECT: detect_params[mask_t][0],
                        TASK_JAM: jam_params[mask_t][0],
                        TASK_COMM: comm_params[mask_t][0],
                    }
                    params = params_map.get(t, torch.zeros(3, device=dev))
                    wf = self.generate_waveform(t, wt_int, params, n_samples)
                    if wf is not None:
                        wf_bank[(t, wt_int)] = wf

        # Build TX signal tensor
        tx_out = torch.zeros(E, R, N, n_samples, dtype=torch.complex64, device=dev)

        for (t, wt), wf in wf_bank.items():
            mask = (task_ids == t) & (wf_types == wt)
            if not mask.any():
                continue
            mask_expanded = mask.unsqueeze(-1)  # [E,R,N,1]
            weighted = weights.unsqueeze(-1) * wf.view(1, 1, 1, -1)  # [E,R,N,S]
            tx_out = torch.where(mask_expanded, weighted, tx_out)

        # Per-radar DRFM waveforms
        for (t, wt, e, r), wf in wf_bank_drfm.items():
            mask = (task_ids[e, r] == t) & (wf_types[e, r] == wt)  # [N]
            if not mask.any():
                continue
            weighted = weights[e, r] * wf.view(1, -1)  # [N, S]
            tx_out[e, r] = torch.where(mask.unsqueeze(-1), weighted, tx_out[e, r])

        return tx_out
