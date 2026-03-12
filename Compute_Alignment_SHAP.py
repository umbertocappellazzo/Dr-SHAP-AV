#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Alignment SHAP

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
from scipy import stats
import argparse

sns.set_context('paper')
rc('font', **{'family': 'cursive', 'cursive': ['Comic Sans MS']})


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description='Modality Contribution Analysis During Token Generation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--path-to-data', required=True, metavar='PATH',
                    help='Path to .npz file.')
parser.add_argument('--num-samples', required=True, type=int, default= 20,
                    help='The number of samples to use to compute generative SHAP. By default, we use the 20 longest ones.')
parser.add_argument('--num-bins', required=True, type=int, default= 10,
                    help='Number of bins for temporal analysis')

args = parser.parse_args()

# =============================================================================
# LOAD DATA
# =============================================================================

experiment_path = args.path_to_data
num_samples = args.num_samples
n_bins = args.num_bins

data = np.load(experiment_path, allow_pickle=True)

shap_values_all = data['shap_values']
num_audio_tokens = data['num_audio_tokens']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_binned_heatmap(shap_matrix, n_feature_bins=10, n_token_bins=10):
    """
    Bin features and tokens, compute mean absolute contribution in each bin.
    
    Parameters:
    -----------
    shap_matrix : np.array, shape (N_features, T_out)
    
    Returns:
    --------
    binned_matrix : np.array, shape (n_feature_bins, n_token_bins)
    """
    N_features, T_out = shap_matrix.shape
    
    feature_bins = np.linspace(0, N_features, n_feature_bins + 1).astype(int)
    token_bins = np.linspace(0, T_out, n_token_bins + 1).astype(int)
    
    binned_matrix = np.zeros((n_feature_bins, n_token_bins))
    
    for i in range(n_feature_bins):
        for j in range(n_token_bins):
            f_start, f_end = feature_bins[i], feature_bins[i+1]
            t_start, t_end = token_bins[j], token_bins[j+1]
            
            if f_end > f_start and t_end > t_start:
                binned_matrix[i, j] = np.abs(shap_matrix[f_start:f_end, t_start:t_end]).mean()
    
    return binned_matrix


def compute_alignment_score(binned_matrix):
    """
    Compute temporal alignment score: diagonal / off-diagonal ratio.
    Score > 1 indicates temporal alignment (late features to late tokens).
    """
    n = binned_matrix.shape[0]
    diagonal_mean = np.diag(binned_matrix).mean()
    off_diagonal_mask = ~np.eye(n, dtype=bool)
    off_diagonal_mean = binned_matrix[off_diagonal_mask].mean()
    
    return diagonal_mean / off_diagonal_mean if off_diagonal_mean > 0 else np.inf



def get_temporal_contribution(shap_matrix, n_feature_bins=3, n_token_bins=10):
    """
    Get contribution per token bin for early/middle/late features.
    """
    N_features, T_out = shap_matrix.shape
    feature_bins = np.linspace(0, N_features, n_feature_bins + 1).astype(int)
    token_bins = np.linspace(0, T_out, n_token_bins + 1).astype(int)
    
    results = []
    for i in range(n_feature_bins):
        f_start, f_end = feature_bins[i], feature_bins[i+1]
        contributions = []
        for j in range(n_token_bins):
            t_start, t_end = token_bins[j], token_bins[j+1]
            contrib = np.abs(shap_matrix[f_start:f_end, t_start:t_end]).sum()
            contributions.append(contrib)
        contributions = np.array(contributions) / np.sum(contributions)
        results.append(contributions)
    return np.array(results)


# =============================================================================
# AGGREGATE ANALYSIS OVER MULTIPLE SAMPLES
# =============================================================================
print("="*60)
print("TEMPORAL ALIGNMENT SHAP ANALYSIS")
print("="*60)

# Storage for aggregated results
all_audio_binned = []
all_video_binned = []
all_audio_temporal = []
all_video_temporal = []
audio_correlations = []
video_correlations = []

is_OmniAVSR = True if ("Omni" in experiment_path or "Llama-AVSR" in experiment_path) else False
is_avhubert = True if "av_hubert" in experiment_path else False
if is_OmniAVSR or is_avhubert:
        sorted_indices_desc = sorted(range(len(num_audio_tokens)), key=lambda i: num_audio_tokens[i], reverse=True)

for sample_idx in range(num_samples):
    shap_values = shap_values_all[sorted_indices_desc[sample_idx]] if (is_OmniAVSR or is_avhubert) else shap_values_all[sample_idx]
    if is_OmniAVSR:
        N_a = num_audio_tokens[sorted_indices_desc[sample_idx]]//4
    elif is_avhubert:
        N_a = num_audio_tokens[sorted_indices_desc[sample_idx]]
    else:
        N_a = num_audio_tokens[sample_idx] // 2
    N_total, T_out = shap_values.shape

    N_v = N_total - N_a
    
    # Split into audio and video
    shap_audio = shap_values[:N_a, :]
    shap_video = shap_values[N_a:, :]
    
    # Binned heatmaps
    audio_binned = create_binned_heatmap(shap_audio, n_bins, n_bins)
    video_binned = create_binned_heatmap(shap_video, n_bins, n_bins)
    
    # Normalize
    audio_binned_norm = audio_binned / audio_binned.sum()
    video_binned_norm = video_binned / video_binned.sum()
    
    all_audio_binned.append(audio_binned_norm)
    all_video_binned.append(video_binned_norm)
    
    # Temporal contribution (early/middle/late features)
    audio_temporal = get_temporal_contribution(shap_audio, n_feature_bins=3, n_token_bins=n_bins)
    video_temporal = get_temporal_contribution(shap_video, n_feature_bins=3, n_token_bins=n_bins)
    
    all_audio_temporal.append(audio_temporal)
    all_video_temporal.append(video_temporal)
    
    
    print(f"Sample {sample_idx}: N_audio={N_a}, N_video={N_v}, T_out={T_out},")

# Average across samples
mean_audio_binned = np.mean(all_audio_binned, axis=0)
mean_video_binned = np.mean(all_video_binned, axis=0)
mean_audio_temporal = np.mean(all_audio_temporal, axis=0)
mean_video_temporal = np.mean(all_video_temporal, axis=0)
std_audio_temporal = np.std(all_audio_temporal, axis=0)
std_video_temporal = np.std(all_video_temporal, axis=0)

# Compute alignment scores on averaged heatmaps
audio_alignment = compute_alignment_score(mean_audio_binned)
video_alignment = compute_alignment_score(mean_video_binned)

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Audio alignment score: {audio_alignment:.2f} (>1 = temporal alignment)")
print(f"Video alignment score: {video_alignment:.2f} (>1 = temporal alignment)")

# =============================================================================
# FIGURE 1: BINNED TEMPORAL HEATMAPS
# =============================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Audio heatmap
im1 = axes1[0].imshow(mean_audio_binned * 100, aspect='auto', cmap='Oranges', origin='lower')
axes1[0].set_xlabel('Output Token Position (%)', fontsize=20, fontweight='bold')
axes1[0].set_ylabel('Audio Feature Position (%)', fontsize=20, fontweight='bold')
axes1[0].set_title('Audio', 
                   fontsize=20, fontweight='bold')
axes1[0].set_xticks(np.arange(n_bins))
axes1[0].set_xticklabels([f'{i*10}' for i in range(n_bins)], fontsize=15)
axes1[0].set_yticks(np.arange(n_bins))
axes1[0].set_yticklabels([f'{i*10}' for i in range(n_bins)], fontsize=15)
cbar_audio = plt.colorbar(im1, ax=axes1[0])
cbar_audio.set_label('Contribution (%)', fontsize=15, fontweight='bold')
cbar_audio.ax.tick_params(labelsize=15)
axes1[0].plot([-0.5, n_bins-0.5], [-0.5, n_bins-0.5], 'w--', linewidth=2, alpha=0.7)

# Video heatmap
im2 = axes1[1].imshow(mean_video_binned * 100, aspect='auto', cmap='GnBu', origin='lower')
axes1[1].set_xlabel('Output Token Position (%)', fontsize=20, fontweight='bold')
axes1[1].set_ylabel('Video Feature Position (%)', fontsize=20, fontweight='bold')
axes1[1].set_title(f'Video', 
                   fontsize=20, fontweight='bold')
axes1[1].set_xticks(np.arange(n_bins))
axes1[1].set_xticklabels([f'{i*10}' for i in range(n_bins)], fontsize=15)
axes1[1].set_yticks(np.arange(n_bins))
axes1[1].set_yticklabels([f'{i*10}' for i in range(n_bins)], fontsize=15)
cbar_video = plt.colorbar(im2, ax=axes1[1])
cbar_video.set_label('Contribution (%)', fontsize=15, fontweight='bold')
cbar_video.ax.tick_params(labelsize=15)
axes1[1].plot([-0.5, n_bins-0.5], [-0.5, n_bins-0.5], 'w--', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('temporal_alignment_heatmap.pdf', dpi=400, bbox_inches='tight')
print("\nSaved: temporal_alignment_heatmap.png")

# =============================================================================
# FIGURE 2: EARLY vs MIDDLE vs LATE FEATURES
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

x_tokens = np.linspace(5, 95, n_bins)
labels = ['Early (0-33%)', 'Middle (33-66%)', 'Late (66-100%)']
colors_audio = ['xkcd:pastel pink', 'xkcd:coral', 'xkcd:dark red']
colors_video = ['xkcd:seafoam', 'xkcd:teal', 'xkcd:dark teal']
markers = ['o', 's', '^']

# Audio
for i in range(3):
    axes2[0].errorbar(x_tokens, mean_audio_temporal[i] * 100, 
                      yerr=std_audio_temporal[i] * 100,
                      fmt=f'{markers[i]}-', color=colors_audio[i],
                      linewidth=2.5, markersize=12, capsize=3,
                      label=f'{labels[i]}')
axes2[0].set_xlabel('Output Token Position (%)', fontsize=20, fontweight='bold')
axes2[0].set_ylabel('Relative Contribution (%)', fontsize=20, fontweight='bold')
axes2[0].set_title('Audio', fontsize=20, fontweight='bold')
axes2[0].legend(fontsize=12, title='Features', title_fontsize=13, loc='best', ncol=3, columnspacing=0.5)
axes2[0].grid(alpha=0.3)
axes2[0].tick_params(labelsize=15)
axes2[0].set_xlim(0, 100)

# Video
for i in range(3):
    axes2[1].errorbar(x_tokens, mean_video_temporal[i] * 100, 
                      yerr=std_video_temporal[i] * 100,
                      fmt=f'{markers[i]}-', color=colors_video[i],
                      linewidth=2.5, markersize=12, capsize=3,
                      label=f'{labels[i]}')
axes2[1].set_xlabel('Output Token Position (%)', fontsize=20, fontweight='bold')
axes2[1].set_ylabel('Relative Contribution (%)', fontsize=20, fontweight='bold')
axes2[1].set_title('Video', fontsize=20, fontweight='bold')
axes2[1].legend(fontsize=12, title='Features', title_fontsize=13, loc='best', ncol=3, columnspacing=0.5)
axes2[1].grid(alpha=0.3)
axes2[1].tick_params(labelsize=15)
axes2[1].set_xlim(0, 100)

plt.tight_layout()
plt.savefig('early_vs_late_features.pdf', dpi=400, bbox_inches='tight')
print("Saved: early_vs_late_features.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nInterpretation guide:")
print("  1. Heatmaps: Strong diagonal = temporal alignment")
print("  2. Alignment score > 1.0: Late features contribute more to late tokens")
print("  3. Early/Late plot: If lines cross, there's temporal structure")

plt.show()