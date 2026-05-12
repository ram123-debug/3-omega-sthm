# 3ω-SThM Simulation

An interactive Streamlit app that simulates a custom 3-omega Scanning Thermal Microscopy (SThM) measurement built on top of a Bruker/Anasys VITA SThM platform.

## Background

The Bruker SThM software only outputs DC voltage. This simulation shows how an external lock-in amplifier and AC drive — injected through the CAL box's AC IN port — can be combined with the DC bias to extract the 3ω thermal signal at every pixel during a raster scan, without any firmware changes or modification to the Bruker system.

## Features

- **Circuit & Concept** — Block diagram and step-by-step explanation of why the 3ω signal appears in the probe voltage
- **Time-Domain Signals** — Probe voltage waveform, FFT, and lock-in demodulation with live SNR estimate
- **Frequency Sweep (Cahill slope)** — Sweep drive frequency over 2+ decades to extract thermal conductivity using the slope method
- **Raster Scan** — Simulated 3ω image at every pixel for a two-material sample, with pixel dwell time and timing validation
- **Bruker Integration** — Signal routing table and practical acquisition workflow for running the lock-in in parallel with Nanoscope

## Requirements

```
numpy
streamlit
matplotlib
scipy
```

Install with:

```bash
pip install numpy streamlit matplotlib scipy
```

## Usage

```bash
streamlit run 3omega_sthm_sim.py
```

Then adjust the sidebar parameters (probe resistance, TCR, drive voltages, sample thermal conductivity/diffusivity, lock-in time constant, etc.) to explore the design space interactively.

## Physics

The simulation models:
- Joule heating in the probe: `P(t) = I(t)² × R_p`
- 2ω temperature oscillation via point-contact heat source on a semi-infinite half-space
- Resistance modulation: `R_p(t) = R_p0 + α_R × ΔT(t)`
- 3ω probe voltage: `V_3ω ≈ ½ × I_1ω × α_R × ΔT_2ω`
- Lock-in demodulation using a 4th-order Butterworth low-pass filter

> Models are analytical approximations for design-space exploration. Real systems have additional non-idealities (tip-sample contact resistance, parasitic capacitance, 1/f noise, cantilever thermal mass).
