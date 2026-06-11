import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
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

COLOR_PITHOS = '#00f2fe'     # Electric Cyan
COLOR_FAISS = '#7f00ff'      # Deep Purple
COLOR_NAIVE = '#ff007f'      # Neon Pink
COLOR_MUTED = '#8b949e'      # Gray

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def load_metrics():
    defaults = {
        "mu_target": 67.25,
        "std_target": 19.35,
        "mu_other": 86.29,
        "std_other": 15.96,
        "threshold": 51,
        "best_mvps": 2441.3
    }
    metrics_path = "pithos_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    defaults[k] = v
            print(f"Loaded dynamic metrics from {metrics_path}")
        except Exception as e:
            print(f"Warning: Could not read {metrics_path} ({e}). Using default values.")
    else:
        print(f"No dynamic metrics file found at {metrics_path}. Using default values.")
    return defaults

def load_baselines():
    defaults = {
        "faiss_mvps": 75.38,
        "sequential_mvps": 4.52
    }
    baselines_path = "baselines_metrics.json"
    if os.path.exists(baselines_path):
        try:
            with open(baselines_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    defaults[k] = v
            print(f"Loaded dynamic baselines from {baselines_path}")
        except Exception as e:
            print(f"Warning: Could not read {baselines_path} ({e}). Using default values.")
    return defaults

def create_distribution_plot():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    metrics = load_metrics()
    mu_target, std_target = metrics["mu_target"], metrics["std_target"]
    mu_other, std_other = metrics["mu_other"], metrics["std_other"]
    threshold = metrics["threshold"]
    
    x = np.linspace(0, 384, 1000)
    
    y_target = normal_pdf(x, mu_target, std_target)
    y_other = normal_pdf(x, mu_other, std_other)
    
    # Plot target curve
    ax.plot(x, y_target, color=COLOR_PITHOS, linewidth=2.5, label='Target Class (Lunar Pit/Cave Entrance)')
    ax.fill_between(x, 0, y_target, color=COLOR_PITHOS, alpha=0.12)
    
    # Plot other curve
    ax.plot(x, y_other, color=COLOR_NAIVE, linewidth=2.5, label='Background Class (Flat Mondgelände/Terrain)')
    ax.fill_between(x, 0, y_other, color=COLOR_NAIVE, alpha=0.12)
    
    # Highlight optimal threshold
    ax.axvline(x=threshold, color='#00f5a0', linestyle='--', linewidth=2, alpha=0.9, label=f'Optimal Threshold ({threshold} bits, F1-optimized)')
    
    # Fill resonant region (True Positives area)
    x_resonant = np.linspace(0, threshold, 500)
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

    os.makedirs('assets', exist_ok=True)
    out_svg = 'assets/distribution_plot.svg'
    out_png = 'assets/distribution_plot.png'
    plt.savefig(out_svg, format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generated: {out_svg}")
    print(f"Generated: {out_png}")

def create_throughput_plot():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    metrics = load_metrics()
    best_mvps = metrics["best_mvps"]
    
    baselines = load_baselines()
    faiss_mvps = baselines["faiss_mvps"]
    seq_mvps = baselines["sequential_mvps"]
    
    # Benchmarked throughput values (MVPS - Million Vectors Per Second)
    labels = [
        'Standard JVM Sequential\n(JIT Compiled Baseline)',
        'Pithos Initial Architecture\n(JVM Vector API, False Sharing)',
        'FAISS Baseline\n(IndexFlatL2, CPU Native)',
        'Pithos Optimized Host-Native\n(Panama FFM & 3-Gate Cascade)'
    ]
    
    mvps_values = [seq_mvps, 35.3, faiss_mvps, best_mvps]
    colors = [COLOR_NAIVE, COLOR_MUTED, COLOR_FAISS, COLOR_PITHOS]
    
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
    ax.set_xlim(0, best_mvps * 1.2)
    
    # Style grid and spines
    ax.grid(True, axis='x', color=GRID_COLOR, linestyle=':', alpha=0.6)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Remove y-axis tick markers but keep labels
    ax.tick_params(axis='y', which='both', length=0, pad=15, labelsize=11)
    
    # Highlight Pithos speedup dynamically compared to FAISS Flat L2
    speedup = best_mvps / faiss_mvps
    ax.annotate(f'~{speedup:.0f}x speedup over FAISS Flat L2\nvia binary transforms & 3-Gate Cascade', 
                xy=(best_mvps, 3.0), 
                xytext=(min(1500, best_mvps - 1500), 2.0),
                arrowprops=dict(arrowstyle="->", color=COLOR_PITHOS, lw=1.5),
                color=COLOR_PITHOS, fontsize=10.5, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.6", fc=PANEL_BG, ec=BORDER_COLOR, alpha=0.9))

    os.makedirs('assets', exist_ok=True)
    out_svg = 'assets/throughput_comparison.svg'
    out_png = 'assets/throughput_comparison.png'
    plt.savefig(out_svg, format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generated: {out_svg}")
    print(f"Generated: {out_png}")

if __name__ == '__main__':
    create_distribution_plot()
    create_throughput_plot()
