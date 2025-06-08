DESCRIPTIONS = {
    "velocity_rms": "√(1/N × Σ(v_z²))",
    "crest_factor": "Peak / RMS",
    "kurtosis_opt": "μ₄ / μ₂²",
    "peak_value_opt": "max(|x|)",
    "rms_10_25khz": "√(1/N × Σ(x²)) in 10–25.6 kHz band",
    "rms_1_10khz": "√(1/N × Σ(x²)) in 1–10 kHz band",
    "rms_0.1_10hz": "√(1/N × Σ(x²)) in 0.1–10 Hz band",
    "rms_10_100hz": "√(1/N × Σ(x²)) in 10–100 Hz band",
    "peak_10_1000hz": "max(|x|) in 10–1000 Hz band",
    "alignment_status": "f(axial_vibration_pattern)",
    "bearing_lubrication": "f(bearing_freq_energy)",
    "electromagnetic_status": "f(motor_current_harmonics)",
    "fit_condition": "f(impulse_signal_features)",
    "rotor_balance_status": "f(phase_diff, amplitude)",
    "rubbing_condition": "f(high_freq_noise, friction)"
}

# techdebt: hardcoding col names - should vary by what is in df
CHART_1_COLS = [
    "alignment_status", "bearing_lubrication", "electromagnetic_status",
    "fit_condition", "rotor_balance_status", "rubbing_condition"
]

# techdebt: hardcoding col names - should include all remaining, and narrow down without breaking
CHART_2_COLS = [
    "velocity_rms", "crest_factor", "kurtosis_opt",
    "peak_value_opt", "rms_10_25khz", "rms_1_10khz"
]