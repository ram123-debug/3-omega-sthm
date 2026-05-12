"""
3omega-SThM Simulation App
==========================

Interactive simulation of a custom 3-omega Scanning Thermal Microscopy
measurement, built on top of a Bruker/Anasys VITA SThM platform.

The Bruker SThM software only outputs DC voltage. This app shows how an
external lock-in amplifier and AC drive (injected through the CAL box's
AC IN port) can be combined with the DC bias to extract the 3-omega
thermal signal at every pixel during a raster scan.

Run:
    streamlit run 3omega_sthm_sim.py

Author: simulation for advisor review
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.signal import butter, sosfiltfilt

# ============================================================
# Page config and global styling
# ============================================================
st.set_page_config(
    page_title="3ω-SThM Simulation",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .small-note { color: #666; font-size: 0.85em; }
    .equation-box {
        background-color: #f5f5f5;
        padding: 0.8em;
        border-left: 3px solid #1f77b4;
        margin: 0.5em 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("3ω-SThM Measurement Simulation")
st.markdown(
    "*Understanding how lock-in detection extracts thermal conductivity "
    "from a Bruker/Anasys SThM probe, even though the SThM software "
    "only controls DC.*"
)

# ============================================================
# Sidebar — instrumental parameters
# ============================================================
st.sidebar.header("Instrumental Parameters")

st.sidebar.subheader("Bridge & Probe")
R_ballast = st.sidebar.number_input(
    "Ballast resistor R_L (Ω)", 100, 10000, 1000, step=100,
    help="The two 1 kΩ series resistors in the bridge cable.",
)
R_probe_0 = st.sidebar.number_input(
    "Probe resistance R_p at room T (Ω)", 100, 2000, 400, step=10,
    help="Cold resistance of the GLA probe at 25 °C.",
)
alpha_R = st.sidebar.number_input(
    "Probe TCR α_R (Ω/°C)", 0.1, 5.0, 1.0, step=0.1,
    help="Temperature coefficient of resistance. Calibrated per probe.",
)
gain_index = st.sidebar.selectbox(
    "CAL box gain", options=[10, 100, 1000], index=2,
    help="Selectable Vs-Vr gain stage in the Anasys CAL box.",
)

st.sidebar.subheader("Drive Voltages")
V_DC = st.sidebar.slider(
    "DC bias V_DC (V) — set by SThM software", 0.0, 2.0, 1.0, step=0.05,
    help="DC component, controlled by the Bruker/Anasys software.",
)
V_AC = st.sidebar.slider(
    "AC drive V_AC (V peak) — injected at AC IN", 0.0, 1.0, 0.2, step=0.01,
    help="AC component, injected externally through the CAL box AC IN port.",
)
f_drive = st.sidebar.slider(
    "Drive frequency ω/2π (Hz)", 10, 5000, 500, step=10,
    help="Lock-in internal oscillator frequency.",
)

st.sidebar.subheader("Sample")
k_sample = st.sidebar.number_input(
    "Sample thermal conductivity k (W/m·K)", 0.1, 200.0, 1.4, step=0.1,
    help="Thermal conductivity of the material under the probe.",
)
diffusivity = st.sidebar.number_input(
    "Sample thermal diffusivity α (m²/s × 10⁻⁶)", 0.1, 100.0, 0.8, step=0.1,
    help="Thermal diffusivity. Si=88, SiO2=0.8, polymer~0.1 (×10⁻⁶).",
) * 1e-6
b_contact = st.sidebar.number_input(
    "Tip-sample contact radius b (nm)", 10, 500, 100, step=10,
    help="Effective heated contact radius. Bigger = less localization.",
) * 1e-9

st.sidebar.subheader("Measurement")
lockin_TC = st.sidebar.select_slider(
    "Lock-in time constant", options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    value=0.3, help="Longer = better SNR but slower per pixel.",
)
noise_nV_rtHz = st.sidebar.slider(
    "Noise floor (nV/√Hz)", 1.0, 100.0, 10.0, step=1.0,
    help="Input-referred noise. SR830 ~6 nV/√Hz at 1 kHz.",
)

# ============================================================
# Physics functions
# ============================================================

def simulate_probe_voltage(V_DC, V_AC, omega, t, R_L, R_p0, alpha_R,
                            k, alpha_diff, b):
    """
    Full time-domain simulation:
    1. Compute current I(t) through probe
    2. Compute power dissipated P(t) = I(t)² × R_p
       For V(t) = V_DC + V_AC cos(ωt), and I = V/(R_L+R_p):
       I² R_p has DC + 1ω + 2ω components
       The 2ω power amplitude is V_AC²/(2(R_L+R_p)²) × R_p
    3. Temperature oscillates at 2ω with amplitude ΔT_2ω ∝ P_2ω/k
    4. R_p(t) = R_p0 + α_R × ΔT(t) → oscillates at 2ω
    5. V_p(t) = I(t) × R_p(t):
       (1ω current) × (2ω resistance) → 1ω and 3ω sidebands.
    """
    V_drive = V_DC + V_AC * np.cos(omega * t)
    I_t = V_drive / (R_L + R_p0)

    # 2ω power amplitude (point-contact heater on half-space)
    P_2omega_amplitude = (V_AC**2 / (2 * (R_L + R_p0)**2)) * R_p0

    # Temperature amplitude — point source on semi-infinite half-space:
    # ΔT = P / (4π k b) in quasi-static limit (λ >> b)
    lambda_th = np.sqrt(alpha_diff / (2 * omega))
    if lambda_th > b:
        delta_T_2omega = P_2omega_amplitude / (4 * np.pi * k * b)
        # Weak log frequency dependence for slope method
        delta_T_2omega *= max(0.3, 1.0 - 0.1 * np.log(b / max(lambda_th, b * 1.01)))
    else:
        delta_T_2omega = P_2omega_amplitude / (4 * np.pi * k * lambda_th)

    delta_T_2omega = max(delta_T_2omega, 1e-9)

    tau_thermal = b**2 / alpha_diff
    phase_lag = -np.arctan(2 * omega * tau_thermal)

    # DC heating raises mean temperature, capped to prevent unphysical values
    P_DC = (V_DC / (R_L + R_p0))**2 * R_p0
    mean_T_rise = min(P_DC / (4 * np.pi * b * k), 200.0)

    R_p_t = R_p0 + alpha_R * (
        mean_T_rise + delta_T_2omega * np.cos(2 * omega * t + phase_lag)
    )

    V_probe = I_t * R_p_t
    return V_probe, I_t, R_p_t, delta_T_2omega


def lockin_demodulate(signal, t, omega_ref, harmonic, time_constant):
    """
    Simulate a lock-in amplifier: multiply by reference at n*omega,
    then low-pass filter with the time constant.
    Returns X (in-phase) and Y (quadrature) amplitudes.
    """
    dt = t[1] - t[0]
    fs = 1 / dt
    # Reference signals
    ref_x = 2 * np.cos(harmonic * omega_ref * t)
    ref_y = -2 * np.sin(harmonic * omega_ref * t)
    # Mix
    mixed_x = signal * ref_x
    mixed_y = signal * ref_y
    # Low-pass filter at f_cutoff = 1 / (2*pi*TC)
    f_cutoff = 1 / (2 * np.pi * time_constant)
    nyq = fs / 2
    normalized = np.clip(f_cutoff / nyq, 1e-6, 0.9999)
    if f_cutoff < nyq:
        sos = butter(4, normalized, btype='low', output='sos')
        X = sosfiltfilt(sos, mixed_x)
        Y = sosfiltfilt(sos, mixed_y)
    else:
        X = mixed_x
        Y = mixed_y
    # Take steady-state value (last quarter of signal, after settling)
    idx_settle = int(0.75 * len(t))
    return np.mean(X[idx_settle:]), np.mean(Y[idx_settle:])


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "📐 1. Circuit & Concept",
    "🌊 2. Time-Domain Signals",
    "📊 3. Frequency Sweep (Cahill slope)",
    "🗺️ 4. Raster Scan with 3ω at every pixel",
    "📚 5. How it integrates with Bruker software",
])

# ============================================================
# TAB 1 — Circuit overview
# ============================================================
with tabs[0]:
    st.header("Why 3ω appears in the probe voltage")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            """
            **The physics in four steps:**

            1. **Drive** the probe with V(t) = V_DC + V_AC cos(ωt).
            The SThM software sets V_DC; an external lock-in or
            function generator injects V_AC through the CAL box's
            AC IN port.

            2. **Current** through the probe: I(t) = V(t) / (R_L + R_p).
            This current has DC and 1ω components.

            3. **Joule heating** in the probe: P(t) = I(t)² × R_p.
            Squaring the drive produces DC, 1ω, **and 2ω** components.
            The temperature oscillates at 2ω.

            4. **Resistance modulation**: R_p(t) = R_p0 + α_R × ΔT(t).
            Since ΔT has a 2ω component, R_p also oscillates at 2ω.

            5. **Probe voltage** V_p(t) = I(t) × R_p(t).
            The product of a 1ω current with a 2ω resistance generates
            **1ω and 3ω components** — exactly what the lock-in
            detects.
            """
        )

        st.markdown('<div class="equation-box">', unsafe_allow_html=True)
        st.latex(r"V_{3\omega} \approx \frac{1}{2} \, I_{1\omega} \, \alpha_R \, \Delta T_{2\omega}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
            And the temperature amplitude ΔT depends on the **thermal
            conductivity of the sample beneath the probe** — that's
            what we measure.
            """
        )

    with col2:
        # Block diagram (matplotlib)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        boxes = [
            (1, 10, 3, 1, "Lock-in osc.\nω", "#FFEBE8"),
            (6, 10, 3, 1, "SThM software\nDC bias", "#E8F1FF"),
            (3.5, 7.5, 3, 1.2, "CAL box\nbridge + gain", "#FFF8DC"),
            (3.5, 5, 3, 1.2, "GLA probe\non sample", "#E8FFE8"),
            (3.5, 2.5, 3, 1, "Diff. preamp", "#F0E8FF"),
            (3.5, 0.3, 3, 1, "Lock-in\ndetect at 3ω", "#FFE8F0"),
        ]
        for x, y, w, h, label, color in boxes:
            box = FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='black', linewidth=1.2,
            )
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                    fontsize=10, fontweight='bold')

        # Arrows
        arrows = [
            ((2.5, 10), (4.5, 8.7), "ω AC"),
            ((7.5, 10), (5.5, 8.7), "V_DC"),
            ((5, 7.5), (5, 6.2), "drive"),
            ((5, 5), (5, 3.5), "V_s−V_r"),
            ((5, 2.5), (5, 1.3), "× ~100"),
        ]
        for (x1, y1), (x2, y2), lbl in arrows:
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', mutation_scale=18,
                color='#444', linewidth=1.5,
            )
            ax.add_patch(arrow)
            ax.text((x1+x2)/2 + 0.3, (y1+y2)/2, lbl,
                    fontsize=8, color='#444', style='italic')

        # Reference line from lock-in osc to lock-in detect
        ref_arrow = FancyArrowPatch(
            (1, 10.3), (3.5, 0.7),
            arrowstyle='->', mutation_scale=15,
            color='#cc6600', linewidth=1, linestyle='--',
            connectionstyle="arc3,rad=-0.3",
        )
        ax.add_patch(ref_arrow)
        ax.text(0.5, 5.5, "ω reference\n(phase lock)",
                fontsize=8, color='#cc6600',
                style='italic', rotation=90)

        ax.set_title("Signal flow: DC from SThM SW + AC from lock-in\n"
                     "→ probe → bridge → preamp → lock-in @ 3ω",
                     fontsize=11, pad=15)
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**Key insight for your advisor:** the Bruker SThM software is unmodified. "
        "It just sets V_DC and reads R_probe for safety. All 3ω action happens "
        "in parallel hardware (lock-in + preamp), tapped off the CAL box's "
        "existing V_s−V_r differential output."
    )

# ============================================================
# TAB 2 — Time domain signals
# ============================================================
with tabs[1]:
    st.header("Time-domain view of the signals")
    st.markdown(
        "Watch the spectral content of the probe voltage. The 1ω peak is "
        "huge (it's just the drive), but a small 3ω peak appears — "
        "that's our thermal signal."
    )

    # Simulate over several periods at the drive frequency
    omega = 2 * np.pi * f_drive
    n_periods = 30
    n_samples_per_period = 200
    t = np.linspace(0, n_periods / f_drive, n_periods * n_samples_per_period)

    V_p, I_t, R_t, dT = simulate_probe_voltage(
        V_DC, V_AC, omega, t, R_ballast, R_probe_0, alpha_R,
        k_sample, diffusivity, b_contact
    )

    # AC component only
    V_p_AC = V_p - np.mean(V_p)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Probe voltage (AC component)")
        fig, ax = plt.subplots(figsize=(7, 4))
        # Show only first few periods
        n_show = 5 * n_samples_per_period
        ax.plot(t[:n_show] * 1000, V_p_AC[:n_show] * 1000,
                color='#1f77b4', linewidth=1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("V_probe AC (mV)")
        ax.set_title(f"Drive at {f_drive} Hz — looks sinusoidal at 1ω, "
                     "but 3ω is hiding inside")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            f"<span class='small-note'>Mean ΔT (DC heating): "
            f"{(V_DC/(R_ballast+R_probe_0))**2 * R_probe_0 / (4*np.pi*b_contact*k_sample):.1f} °C • "
            f"ΔT amplitude at 2ω: {dT*1000:.2f} m°C</span>",
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("FFT of probe voltage (AC)")
        fig, ax = plt.subplots(figsize=(7, 4))
        dt = t[1] - t[0]
        fft_signal = np.fft.rfft(V_p_AC)
        freqs = np.fft.rfftfreq(len(t), dt)
        ax.semilogy(freqs, np.abs(fft_signal) / len(t) * 2, color='#d62728')
        ax.axvline(f_drive, color='gray', linestyle='--', alpha=0.5,
                   label=f"1ω = {f_drive} Hz")
        ax.axvline(2 * f_drive, color='orange', linestyle='--', alpha=0.5,
                   label=f"2ω = {2*f_drive} Hz")
        ax.axvline(3 * f_drive, color='red', linestyle='--', alpha=0.5,
                   label=f"3ω = {3*f_drive} Hz")
        ax.set_xlim(0, 5 * f_drive)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (V)")
        ax.set_title("Spectrum — 1ω dominates, 3ω is the thermal signal")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Lock-in demodulation
    st.subheader("What the lock-in sees")
    col3, col4, col5 = st.columns(3)

    X1, Y1 = lockin_demodulate(V_p_AC, t, omega, 1, lockin_TC)
    X3, Y3 = lockin_demodulate(V_p_AC, t, omega, 3, lockin_TC)
    R1 = np.sqrt(X1**2 + Y1**2) * gain_index
    R3 = np.sqrt(X3**2 + Y3**2) * gain_index

    with col3:
        st.metric("V_1ω (after gain)", f"{R1*1000:.3f} mV",
                  help="The drive itself — huge, contains no thermal info.")
    with col4:
        st.metric("V_3ω (after gain)", f"{R3*1e6:.3f} µV",
                  help="The thermal signal we care about.")
    with col5:
        ratio = R1 / max(R3, 1e-12)
        st.metric("V_1ω / V_3ω ratio", f"{ratio:.0f}×",
                  help="Why you need a differential preamp and a "
                       "harmonic-detection lock-in.")

    # Noise comparison
    bandwidth = 1 / (4 * lockin_TC)  # ENBW for 1st order LPF approx
    noise_rms = noise_nV_rtHz * 1e-9 * np.sqrt(bandwidth) * gain_index
    snr = R3 / noise_rms
    st.markdown(
        f"<div class='equation-box'>"
        f"<b>SNR estimate:</b> V_3ω = {R3*1e6:.2f} µV, "
        f"noise floor (TC = {lockin_TC} s, BW ≈ {bandwidth:.1f} Hz) "
        f"= {noise_rms*1e6:.3f} µV → <b>SNR = {snr:.1f}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if snr < 3:
        st.warning("⚠️ SNR < 3: increase V_AC, increase lock-in time "
                   "constant, or use higher gain.")
    elif snr > 100:
        st.success("✅ Excellent SNR — you can scan faster (shorter TC).")

# ============================================================
# TAB 3 — Frequency sweep
# ============================================================
with tabs[2]:
    st.header("Frequency sweep — extracting thermal conductivity")
    st.markdown(
        "**Cahill's slope method:** sweep ω over 2+ decades. Plot ΔT vs ln(ω). "
        "The slope is inversely proportional to k. This is how a single "
        "measurement gives you a quantitative k."
    )

    f_min = st.slider("Min frequency (Hz)", 1, 500, 30)
    f_max = st.slider("Max frequency (Hz)", 100, 5000, 3000)
    n_points = st.slider("Number of points", 5, 50, 20)

    if f_max <= f_min:
        st.error("f_max must be greater than f_min")
    else:
        freqs_sweep = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        deltaT_measured = []
        V3w_measured = []

        for f in freqs_sweep:
            om = 2 * np.pi * f
            n_per = max(20, int(2000 / f))  # adaptive samples
            tt = np.linspace(0, n_per / f, n_per * 200)
            V_p, I_t, R_t, dT_local = simulate_probe_voltage(
                V_DC, V_AC, om, tt, R_ballast, R_probe_0, alpha_R,
                k_sample, diffusivity, b_contact,
            )
            V_p_AC_local = V_p - np.mean(V_p)
            X3_l, Y3_l = lockin_demodulate(V_p_AC_local, tt, om, 3, lockin_TC)
            R3_l = np.sqrt(X3_l**2 + Y3_l**2)
            # Convert V_3ω back to ΔT: ΔT = 2 V_3ω / (α_R × I_1ω)
            I_1omega = V_AC / (R_ballast + R_probe_0)
            dT_extracted = 2 * R3_l / (alpha_R * I_1omega) if I_1omega > 0 else 0
            deltaT_measured.append(dT_extracted)
            V3w_measured.append(R3_l * gain_index)

        deltaT_measured = np.array(deltaT_measured)
        V3w_measured = np.array(V3w_measured)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.semilogx(freqs_sweep, deltaT_measured * 1000,
                        'o-', color='#1f77b4', markersize=6)
            ax.set_xlabel("Frequency (Hz, log scale)")
            ax.set_ylabel("ΔT amplitude (m°C)")
            ax.set_title(f"Slope plot — current sample k = {k_sample} W/m·K")
            ax.grid(which='both', alpha=0.3)
            # Linear fit on ln(f)
            mask = deltaT_measured > 0
            if mask.sum() > 3:
                p = np.polyfit(np.log(freqs_sweep[mask]),
                               deltaT_measured[mask], 1)
                slope = p[0]
                ax.plot(freqs_sweep[mask],
                        p[0] * np.log(freqs_sweep[mask]) + p[1],
                        'r--', alpha=0.7, label=f"Slope = {slope*1000:.2f} m°C/decade")
                ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.semilogx(freqs_sweep, V3w_measured * 1e6,
                        's-', color='#d62728', markersize=6)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("V_3ω at lock-in input (µV)")
            ax.set_title("Lock-in V_3ω vs frequency")
            ax.grid(which='both', alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            f"<div class='equation-box'>"
            f"For a line heater on a semi-infinite substrate:<br>"
            f"k = −P / (2πL) × d(ln ω)/d(ΔT)<br>"
            f"For tip-contact geometry, use <b>differential method</b>: "
            f"measure on bare substrate, then on film, subtract."
            f"</div>",
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 4 — Raster scan with 3ω at each pixel
# ============================================================
with tabs[3]:
    st.header("3ω at every pixel during a raster scan")
    st.markdown(
        "**Your key question:** the Bruker software only outputs DC, "
        "so how do we get a 3ω value at each pixel? The answer is: "
        "we run the lock-in in parallel and stream its X/Y outputs "
        "into the AFM controller's auxiliary input (or log them "
        "separately and align by timestamp)."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Simulated sample map")
        scan_size = st.select_slider("Pixels per line", [16, 32, 64, 128], value=64)
        pixel_dwell = st.slider("Pixel dwell time (ms)", 1, 100, 20)

        # Make a synthetic sample with two materials
        x = np.linspace(-1, 1, scan_size)
        y = np.linspace(-1, 1, scan_size)
        X, Y = np.meshgrid(x, y)
        # A circular inclusion of low-k material in a high-k matrix
        radius = 0.4
        mask_film = (X**2 + Y**2) < radius**2
        k_map = np.where(mask_film, 0.3, k_sample)  # low-k inclusion
        # Add some texture
        k_map += 0.05 * k_map * np.random.RandomState(42).randn(*k_map.shape)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(k_map, cmap='viridis', extent=[-1, 1, -1, 1])
        ax.set_title(f"Sample k map (W/m·K) — {scan_size}×{scan_size} px")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        plt.colorbar(im, ax=ax, label="k (W/m·K)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Simulated 3ω image")
        # For each pixel, compute V_3ω given local k using same physics
        omega_scan = 2 * np.pi * f_drive
        # 2ω power amplitude (point-contact heater)
        P_2omega = (V_AC**2 / (2 * (R_ballast + R_probe_0)**2)) * R_probe_0
        I_1omega = V_AC / (R_ballast + R_probe_0)
        # ΔT amplitude per pixel (point source on half-space)
        delta_T_map = P_2omega / (4 * np.pi * b_contact * k_map)
        # Frequency factor
        lambda_th = np.sqrt(diffusivity / (2 * omega_scan))
        if lambda_th > b_contact:
            freq_factor = max(0.3, 1.0 - 0.1 * np.log(b_contact / max(lambda_th, b_contact*1.01)))
            delta_T_map = delta_T_map * freq_factor
        # V_3ω map: V_3ω ≈ ½ × I_1ω × α_R × ΔT_2ω
        V3w_map = 0.5 * I_1omega * alpha_R * delta_T_map * gain_index
        # Add noise
        bandwidth = 1 / (4 * lockin_TC)
        noise_per_px = noise_nV_rtHz * 1e-9 * np.sqrt(bandwidth) * gain_index
        rng = np.random.RandomState(7)
        V3w_map_noisy = V3w_map + noise_per_px * rng.randn(*V3w_map.shape)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(V3w_map_noisy * 1e6, cmap='hot',
                       extent=[-1, 1, -1, 1])
        ax.set_title(f"Measured V_3ω map (µV) — drive at {f_drive} Hz")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        plt.colorbar(im, ax=ax, label="V_3ω (µV)")
        st.pyplot(fig)
        plt.close(fig)

    # Acquisition timing
    n_pixels = scan_size ** 2
    total_time = n_pixels * pixel_dwell / 1000  # seconds
    settling_time = 5 * lockin_TC
    if pixel_dwell / 1000 < settling_time:
        st.error(
            f"⚠️ Pixel dwell ({pixel_dwell} ms) is shorter than 5× the lock-in "
            f"time constant ({settling_time*1000:.0f} ms). The lock-in won't "
            f"settle before the probe moves on. Either increase dwell time "
            f"or decrease TC."
        )
    else:
        st.success(
            f"✅ Pixel dwell ({pixel_dwell} ms) > 5×TC ({settling_time*1000:.0f} ms). "
            f"Total scan time: **{total_time/60:.1f} minutes** for "
            f"{scan_size}×{scan_size} pixels."
        )

    st.markdown(
        """
        **Practical acquisition workflow:**

        1. **Bruker side (unmodified):** Start a normal contact-mode raster
           scan in Nanoscope. Set V_DC in VITA_Studio. The SThM software
           records its usual V_s−V_r DC signal.

        2. **Lock-in side (parallel):** Lock-in continuously drives AC and
           measures V_3ω(X), V_3ω(Y) at its time constant. Its analog
           outputs (rear panel BNCs labeled CH1, CH2 on SR830) go to two
           **auxiliary input BNCs** on the NS-5 controller (Aux Input 1
           and Aux Input 2 on Bruker controllers).

        3. **Nanoscope captures these as extra image channels** alongside
           topography and SThM-DC, perfectly registered to the scan
           pixels — no software modification needed.

        4. Alternatively, **log lock-in data via GPIB/USB** to a CSV with
           timestamps, then align to the AFM scan post-hoc using the
           scan trigger output as a synchronization signal.
        """
    )

# ============================================================
# TAB 5 — Integration with Bruker software
# ============================================================
with tabs[4]:
    st.header("How this all integrates with the Bruker software stack")

    st.markdown(
        """
        ### What stays the same
        - **Nanoscope** controls all AFM functions (XY scan, Z feedback,
          contact-mode engage, deflection monitoring). Unchanged.
        - **VITA_Studio** sets the DC bias on the probe and reads R_probe
          for safety abort. Unchanged.
        - Existing AC IN port on the CAL box accepts the lock-in AC drive
          (mentioned in Section 4.3 of the SThM manual).

        ### What you add externally
        - **Lock-in amplifier** (SR830 or MFLI) — provides AC drive,
          demodulates V_3ω.
        - **Differential preamp** (SR560 or DIY INA188) between CAL box
          V_s−V_r output and lock-in signal input.
        - **A few BNC cables and a PC running a Python data-acquisition
          script** that talks to the lock-in via GPIB/USB.

        ### Signal routing summary

        | Signal | From | To |
        |---|---|---|
        | DC bias | SThM Controller | CAL box drive |
        | AC ω | Lock-in osc. output | CAL box AC IN |
        | Bridge unbalance | CAL box V_s−V_r BNC | Diff. preamp input |
        | Amplified signal | Diff. preamp output | Lock-in signal input |
        | V_3ω X | Lock-in CH1 BNC | NS-5 Aux Input 1 |
        | V_3ω Y | Lock-in CH2 BNC | NS-5 Aux Input 2 |
        | ω ref. (internal) | Lock-in (internal) | Lock-in detector |

        ### Why this works for the raster scan
        The lock-in is **continuously locked** to its own internal
        oscillator and demodulating in real time. Its CH1/CH2 outputs
        are slowly-varying voltages (bandwidth ≈ 1/TC ≈ a few Hz)
        proportional to V_3ω.X and V_3ω.Y. These look to the Bruker
        controller exactly like a thermocouple or photodetector
        voltage — it samples them once per pixel as it would any
        auxiliary input. The 3ω modulation is *already gone* by the
        time the signal reaches the AFM controller; what remains is a
        slow voltage that encodes the local thermal response.

        ### Caveat: the 500 Hz internal filter
        The CAL box has a low-pass filter on V_s−V_r at ~500 Hz. For
        3ω at f_drive = 200 Hz, the signal at 600 Hz is still partly
        attenuated. Two solutions:
        1. Drive at lower frequency (f_drive < 150 Hz so 3ω < 450 Hz).
        2. Tap V_s and V_r before the filter (Bruker can provide a
           modified bridge cable, per Section 4.3 of the manual).
        """
    )

    st.info(
        "**Bottom line for your advisor:** This is a non-invasive add-on. "
        "The Bruker system runs in its normal mode. The 3ω capability "
        "lives entirely in external hardware that taps the bridge output. "
        "No firmware changes, no opening of the CAL box, no warranty risk."
    )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<div class='small-note'>"
    "Simulation uses simplified analytical models (Cahill line-source + "
    "point-contact corrections, ideal lock-in demodulation, additive "
    "Gaussian noise). Real systems have additional non-idealities: "
    "tip-sample contact resistance, parasitic capacitance, ground loops, "
    "1/f noise, cantilever thermal mass, and non-uniform heating of the "
    "probe leg. Use this for intuition and design space exploration, not "
    "for predicting absolute V_3ω values."
    "</div>",
    unsafe_allow_html=True,
)
