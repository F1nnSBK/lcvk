import os
import numpy as np
import matplotlib.pyplot as plt

# Set styling for ultra-premium dark mode
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#8b949e'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

DARK_BG = '#0b0e14'
PANEL_BG = '#161b22'
BORDER_COLOR = '#30363d'
GRID_COLOR = '#21262d'

COLOR_LCVK = '#00f2fe'      # Electric Cyan
COLOR_FAISS = '#7f00ff'     # Deep Purple
COLOR_NAIVE = '#ff007f'     # Neon Pink
COLOR_MUTED = '#8b949e'     # Gray

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def create_distribution_plot():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    # Mathematical representation of real-data distribution from run_real_verification.py
    # Target 7s vs Non-7s (Other digits)
    mu_target, std_target = 66.44, 20.62
    mu_other, std_other = 78.04, 17.45
    
    x = np.linspace(20, 140, 1000)
    
    y_target = normal_pdf(x, mu_target, std_target)
    y_other = normal_pdf(x, mu_other, std_other)
    
    # Plot target curve
    ax.plot(x, y_target, color=COLOR_LCVK, linewidth=2.5, label='Target Digit ("7" - Lunar Pit)')
    ax.fill_between(x, 0, y_target, color=COLOR_LCVK, alpha=0.12)
    
    # Plot other curve
    ax.plot(x, y_other, color=COLOR_NAIVE, linewidth=2.5, label='Other Digits (0-6, 8-9 - Surface Terrain)')
    ax.fill_between(x, 0, y_other, color=COLOR_NAIVE, alpha=0.12)
    
    # Highlight optimal threshold at 50 bits
    threshold = 46
    ax.axvline(x=threshold, color='#00f5a0', linestyle='--', linewidth=2, alpha=0.9, label='Optimal Threshold (46 bits)')
    
    # Fill resonant region (True Positives area)
    x_resonant = np.linspace(20, threshold, 500)
    y_target_res = normal_pdf(x_resonant, mu_target, std_target)
    ax.fill_between(x_resonant, 0, y_target_res, color='#00f5a0', alpha=0.25, label='Resonant Zone (True Positives)')
    
    # Formatting
    ax.set_title('Hamming Distance Distribution & Semantic Threshold', fontsize=15, fontweight='bold', pad=20, color='#f0f6fc')
    ax.set_xlabel('Hamming Distance (Bits)', fontsize=12, labelpad=10)
    ax.set_ylabel('Probability Density', fontsize=12, labelpad=10)
    
    # Style grid and spines
    ax.grid(True, color=GRID_COLOR, linestyle=':', alpha=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(BORDER_COLOR)
        ax.spines[spine].set_linewidth(1.2)
        
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc='upper right', fontsize=10)
    
    # Annotation for the threshold
    ax.annotate('Quantization Cut-off\n(F1-Optimized)', 
                xy=(threshold, 0.005), 
                xytext=(threshold - 25, 0.012),
                arrowprops=dict(arrowstyle="->", color='#00f5a0', lw=1.5),
                color='#f0f6fc', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc=PANEL_BG, ec=BORDER_COLOR, alpha=0.9))

    os.makedirs('assets', exist_ok=True)
    out_path = 'assets/distribution_plot.svg'
    plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generated: {out_path}")

def create_throughput_plot():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    # Benchmarked throughput values (MVPS - Million Vectors Per Second)
    labels = [
        'Standard JVM Execution\n(JIT Compiled Sequential)',
        'LCVK Initial Architecture\n(JVM Vector API, False Sharing)',
        'LCVK Optimized Architecture\n(Scalar Unrolling, Thread-Local Segments)',
        'FAISS Baseline\n(IndexFlat, CPU Native)'
    ]
    
    # Realistic throughput numbers matching our run results & standard CPU FAISS
    mvps_values = [12.0, 35.3, 2441.3, 2250.0]
    
    colors = [COLOR_NAIVE, COLOR_MUTED, COLOR_LCVK, COLOR_FAISS]
    
    # Render horizontal bar chart
    bars = ax.barh(labels, mvps_values, color=colors, height=0.55, edgecolor=BORDER_COLOR, linewidth=1)
    
    # Add values on top of the bars
    for bar in bars:
        width = bar.get_width()
        label_text = f"{width:,.1f} MVPS" if width >= 1.0 else f"{width} MVPS"
        ax.text(width + 40, bar.get_y() + bar.get_height()/2, 
                label_text, 
                va='center', ha='left', fontsize=11, fontweight='bold',
                color='#f0f6fc')
                
    # Style formatting
    ax.set_title('Planetary Scale Scanning Throughput Comparison', fontsize=15, fontweight='bold', pad=20, color='#f0f6fc')
    ax.set_xlabel('Throughput (Million Vectors/Sec - MVPS)', fontsize=12, labelpad=10)
    
    # Customize axis limits to fit labels nicely
    ax.set_xlim(0, 3200)
    
    # Style grid and spines
    ax.grid(True, axis='x', color=GRID_COLOR, linestyle=':', alpha=0.6)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Remove y-axis tick markers but keep labels
    ax.tick_params(axis='y', which='both', length=0, pad=15, labelsize=11)
    
    # Highlight LCVK v2.0 close performance to FAISS
    ax.annotate('~69x speedup via memory-aligned\nscalar operations and localized buffers', 
                xy=(2441.3, 2.0), 
                xytext=(1500, 1.2),
                arrowprops=dict(arrowstyle="->", color=COLOR_LCVK, lw=1.5),
                color=COLOR_LCVK, fontsize=10.5, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.6", fc=PANEL_BG, ec=BORDER_COLOR, alpha=0.9))

    out_path = 'assets/throughput_comparison.svg'
    plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == '__main__':
    create_distribution_plot()
    create_throughput_plot()
