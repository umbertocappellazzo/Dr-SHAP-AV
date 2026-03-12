#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modality Contribution Analysis During Token Generation
- Computes windowed audio/video contributions from SHAP values
- Plots Clean vs Noisy conditions for two methods
- 10 samples, 10 windows (10% intervals)

Usage:
    python Plot_output_trend.py \
        --method_A_clean  path/to/WF_clean.npz \
        --method_A_noisy  path/to/WF_noisy.npz \
        --method_B_clean  path/to/Omni_clean.npz \
        --method_B_noisy  path/to/Omni_noisy.npz \
        --method_C_clean  path/to/AVHuBERT_clean.npz \
        --method_C_noisy  path/to/AVHuBERT_noisy.npz
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
from matplotlib.lines import Line2D

sns.set_context('paper')
rc('font', **{'family': 'cursive', 'cursive': ['Comic Sans MS']})

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description='Modality Contribution Analysis During Token Generation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--Whisper-Flamingo-clean-path', required=True, metavar='PATH',
                    help='Path to Whisper-Flamingo clean .npz file.')
parser.add_argument('--Whisper-Flamingo-noisy-path', required=True, metavar='PATH',
                    help='Path to Whisper-Flamingo noisy .npz file.')
parser.add_argument('--Omni-AVSR-clean-path', required=True, metavar='PATH',
                    help='Path to Omni-AVSR clean .npz file.')
parser.add_argument('--Omni-AVSR-noisy-path', required=True, metavar='PATH',
                    help='Path to Omni-AVSR noisy .npz file.')
parser.add_argument('--AVHuBERT-clean-path', required=True, metavar='PATH',
                    help='Path to AV-HuBERT clean .npz file.')
parser.add_argument('--AVHuBERT-noisy-path', required=True, metavar='PATH',
                    help='Path to AV-HuBERT noisy .npz file.')
parser.add_argument('--num-samples', required=True, type=int, default= 20,
                    help='The number samples to use to compute generative SHAP. By default, we use the 20 longest ones.')
parser.add_argument('--num-windows', required=True, type=int, default= 5,
                    help='Number of windows.')

args = parser.parse_args()

# =============================================================================
# PARAMETERS
# =============================================================================
num_samples = args.num_samples
num_windows = args.num_windows



# Plot settings
color_clean = 'xkcd:teal'
color_noisy = 'xkcd:coral'
marker_method_A = 'o'
marker_method_B = 'X'
marker_method_C = "D"
method_A_name = 'Whisper-Flamingo'
method_B_name = 'Omni-AVSR'
method_C_name = "AV-HuBERT"

# =============================================================================
# FUNCTION TO COMPUTE WINDOWED CONTRIBUTIONS
# =============================================================================
def compute_windowed_contributions(npz_file, method, num_samples=10, num_windows=10):
    """
    Load SHAP values and compute windowed audio/video contributions.
    
    Returns:
    --------
    audio_mean, audio_std, video_mean, video_std : np.arrays of shape (num_windows,)
    """
    data = np.load(npz_file, allow_pickle=True)
    
    shap_values_all = data['shap_values']
    num_audio_tokens = data['num_audio_tokens']

    if method in ["Omni-AVSR", "avhubert"]:
        sorted_indices_desc = sorted(range(len(num_audio_tokens)), key=lambda i: num_audio_tokens[i], reverse=True)
    
    all_audio = []
    all_video = []
    
    for sample_idx in range(num_samples):
        if method in ["Omni-AVSR", "avhubert"]:
            shap_values = shap_values_all[sorted_indices_desc[sample_idx]]
        else:
            shap_values = shap_values_all[sample_idx]
        T_out = shap_values.shape[1]
        if method == "Omni-AVSR":
            N_a = num_audio_tokens[sorted_indices_desc[sample_idx]]//4
        elif method == "whisper-flamingo":
            N_a = num_audio_tokens[sample_idx] // 2 
        elif method == "avhubert":
            N_a = num_audio_tokens[sorted_indices_desc[sample_idx]]

        # Percentage-based window boundaries
        window_boundaries = np.linspace(0, T_out, num_windows + 1).astype(int)
        
        audio_windowed = []
        video_windowed = []
        
        for w in range(num_windows):
            start_idx = window_boundaries[w]
            end_idx = window_boundaries[w + 1]
            
            if end_idx <= start_idx:
                audio_windowed.append(audio_windowed[-1] if audio_windowed else 0.5)
                video_windowed.append(video_windowed[-1] if video_windowed else 0.5)
                continue
            
            audio_win = np.abs(shap_values[:N_a, start_idx:end_idx]).sum()
            video_win = np.abs(shap_values[N_a:, start_idx:end_idx]).sum()
            
            total_win = audio_win + video_win
            audio_windowed.append(audio_win / total_win)
            video_windowed.append(video_win / total_win)
        
        all_audio.append(audio_windowed)
        all_video.append(video_windowed)
        
        print(f"  Sample {sample_idx}: T_out={T_out}, N_audio={N_a}, N_video={shap_values.shape[0]-N_a}")
    
    all_audio = np.array(all_audio)
    all_video = np.array(all_video)
    
    return (np.mean(all_audio, axis=0), np.std(all_audio, axis=0),
            np.mean(all_video, axis=0), np.std(all_video, axis=0))

# =============================================================================
# COMPUTE CONTRIBUTIONS FOR ALL CONDITIONS
# =============================================================================
print("="*60)
print("Computing windowed contributions...")
print("="*60)

print(f"\nWhisper-Flamingo - Clean:")
audio_A_clean_mean, audio_A_clean_std, _, _ = compute_windowed_contributions(
    args.Whisper_Flamingo_clean_path, "whisper-flamingo", num_samples, num_windows)

print(f"\nWhisper-Flamingo - Noisy:")
audio_A_noisy_mean, audio_A_noisy_std, _, _ = compute_windowed_contributions(
    args.Whisper_Flamingo_noisy_path, "whisper-flamingo", num_samples, num_windows)

print(f"\nOmni-AVSR - Clean:")
audio_B_clean_mean, audio_B_clean_std, _, _ = compute_windowed_contributions(
    args.Omni_AVSR_clean_path, "Omni-AVSR", num_samples, num_windows)

print(f"\nOmni-AVSR - Noisy:")
audio_B_noisy_mean, audio_B_noisy_std, _, _ = compute_windowed_contributions(
   args.Omni_AVSR_noisy_path, "Omni-AVSR", num_samples, num_windows)

print(f"\nAV-HuBERT - Clean:")
audio_C_clean_mean, audio_C_clean_std, _, _ = compute_windowed_contributions(
    args.AVHuBERT_clean_path, "avhubert", num_samples, num_windows)

print(f"\nAV-HuBERT - Noisy:")
audio_C_noisy_mean, audio_C_noisy_std, _, _ = compute_windowed_contributions(
   args.AVHuBERT_noisy_path, "avhubert", num_samples, num_windows)

# =============================================================================
# PLOT
# =============================================================================
print("\n" + "="*60)
print("Creating plot...")
print("="*60)

# X-axis: center of each 10% window
x_positions = np.linspace(5, 95, num_windows)

fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# Confidence bands
ax.fill_between(x_positions, 
                (audio_A_clean_mean - audio_A_clean_std) * 100, 
                (audio_A_clean_mean + audio_A_clean_std) * 100,
                color=color_clean, alpha=0.15)
ax.fill_between(x_positions, 
               (audio_B_clean_mean - audio_B_clean_std) * 100, 
               (audio_B_clean_mean + audio_B_clean_std) * 100,
               color=color_clean, alpha=0.15)
ax.fill_between(x_positions, 
               (audio_C_clean_mean - audio_C_clean_std) * 100, 
               (audio_C_clean_mean + audio_C_clean_std) * 100,
               color=color_clean, alpha=0.15)
ax.fill_between(x_positions, 
                (audio_A_noisy_mean - audio_A_noisy_std) * 100, 
                (audio_A_noisy_mean + audio_A_noisy_std) * 100,
                color=color_noisy, alpha=0.15)
ax.fill_between(x_positions, 
               (audio_B_noisy_mean - audio_B_noisy_std) * 100, 
               (audio_B_noisy_mean + audio_B_noisy_std) * 100,
               color=color_noisy, alpha=0.15)
ax.fill_between(x_positions, 
               (audio_C_noisy_mean - audio_C_noisy_std) * 100, 
               (audio_C_noisy_mean + audio_C_noisy_std) * 100,
               color=color_noisy, alpha=0.15)

# Mean lines
ax.plot(x_positions, audio_A_clean_mean * 100, color=color_clean, 
        linewidth=3, marker=marker_method_A, markersize=20, 
        markeredgecolor='xkcd:charcoal grey', markeredgewidth=2., linestyle='-')
ax.plot(x_positions, audio_B_clean_mean * 100, color=color_clean, 
       linewidth=3, marker=marker_method_B, markersize=20, 
       markeredgecolor='xkcd:charcoal grey', markeredgewidth=2., linestyle='-')
ax.plot(x_positions, audio_C_clean_mean * 100, color=color_clean, 
       linewidth=3, marker=marker_method_C, markersize=16, 
       markeredgecolor='xkcd:charcoal grey', markeredgewidth=2., linestyle='-')
ax.plot(x_positions, audio_A_noisy_mean * 100, color=color_noisy, 
        linewidth=3, marker=marker_method_A, markersize=20, 
        markeredgecolor='xkcd:charcoal grey', markeredgewidth=2., linestyle='-')
ax.plot(x_positions, audio_B_noisy_mean * 100, color=color_noisy, 
       linewidth=3, marker=marker_method_B, markersize=20, 
       markeredgecolor='xkcd:charcoal grey', markeredgewidth=2, linestyle='-')
ax.plot(x_positions, audio_C_noisy_mean * 100, color=color_noisy, 
       linewidth=3, marker=marker_method_C, markersize=16, 
       markeredgecolor='xkcd:charcoal grey', markeredgewidth=2, linestyle='-')

# Axis styling
ax.set_xlabel('Token Generation Progress (%)', fontsize=20)
ax.set_ylabel('Audio Contribution (%)', fontsize=20)
ax.set_xlim(0, 100)
ax.set_ylim(25, 75)
ax.tick_params(axis='both', labelsize=14)

# Right y-axis (Video, inverted)
ax2 = ax.twinx()
ax2.set_ylim(75, 25)
ax2.set_ylabel('Video Contribution (%)', fontsize=20)
ax2.tick_params(axis='y', labelsize=14)

# Grid
ax.grid(color='#95a5a6', linestyle='--', linewidth=0.5, alpha=0.4)
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#cccccc', linewidth=0.5)

# Legend 1: Condition
legend_condition = [
    Line2D([0], [0], color=color_clean, linewidth=3, linestyle='-', label='Clean (∞ SNR)'),
    Line2D([0], [0], color=color_noisy, linewidth=3, linestyle='-', label='Noisy (-10 dB)')
]
leg1 = ax.legend(handles=legend_condition, title='Condition', fontsize=17, title_fontsize=18,
                loc='best', bbox_to_anchor=(0.65, 0.725), fancybox=True, shadow=True, framealpha=0.95)

ax.add_artist(leg1)

# Legend 2: Method
legend_method = [
    Line2D([0], [0], color='gray', marker=marker_method_A, markersize=15, 
           markeredgecolor='xkcd:charcoal grey', markeredgewidth=2, 
           linewidth=0, linestyle='', label=method_A_name),
    Line2D([0], [0], color='gray', marker=marker_method_B, markersize=15, 
           markeredgecolor='xkcd:charcoal grey', markeredgewidth=2, 
           linewidth=0, linestyle='', label=method_B_name),
    Line2D([0], [0], color='gray', marker=marker_method_C, markersize=13, 
           markeredgecolor='xkcd:charcoal grey', markeredgewidth=2, 
           linewidth=0, linestyle='', label=method_C_name)
]
leg2 = ax.legend(handles=legend_method, title='Method', fontsize=17, title_fontsize=18,
                 loc='best', bbox_to_anchor=(0.1, 0.75), fancybox=True, shadow=True, framealpha=0.95)

plt.tight_layout()

plt.savefig('modality_contribution_output_tokens.pdf', dpi=400, bbox_inches='tight')

print("\nPlot saved to 'modality_contribution_clean_noisy.png' and '.pdf'")
plt.show()