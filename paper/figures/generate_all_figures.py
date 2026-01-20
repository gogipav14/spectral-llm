"""
Generate all figures for the Boolean Fourier Logic paper.

Figures:
1. Architecture diagram (spectral composition pipeline)
2. Sparsity comparison across phases
3. Phase 3 ternary mask heatmap
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))


def figure1_architecture():
    """
    Figure 1: Hierarchical Spectral Composition Architecture
    Shows the pipeline: Input -> Fourier Basis -> Spectral Selection -> Sign Modulation -> Output
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#E8F4FD',
        'fourier': '#FFF3E0',
        'sinkhorn': '#E8F5E9',
        'sign': '#FCE4EC',
        'output': '#F3E5F5'
    }

    # Box positions: (x, y, width, height)
    boxes = [
        (0.3, 1.5, 1.5, 1.2, 'Input\n$(a,b) \\in \\{-1,+1\\}^2$', colors['input']),
        (2.3, 1.5, 1.8, 1.2, 'Fourier Basis\n$\\phi = [1,a,b,ab]$', colors['fourier']),
        (4.6, 1.5, 2.0, 1.2, 'Sinkhorn\nProjection\n$P \\in \\mathcal{B}_n$', colors['sinkhorn']),
        (7.1, 1.5, 1.8, 1.2, 'Sign Mod.\n$s \\in \\{-1,+1\\}^n$', colors['sign']),
        (9.4, 1.5, 1.5, 1.2, 'Output\n$\\hat{y} \\in \\{-1,+1\\}$', colors['output']),
    ]

    # Draw boxes
    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9,
                fontweight='bold', linespacing=1.3)

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5,
                       connectionstyle='arc3,rad=0')
    arrows = [(1.8, 2.1), (4.1, 2.1), (6.6, 2.1), (8.9, 2.1)]
    for i, (x, y) in enumerate(arrows):
        ax.annotate('', xy=(x+0.5, y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # Add equation at bottom
    eq = r'$\hat{y} = \text{sign}\left(s \cdot (P \cdot W_{\text{logic}})^\top \phi(x)\right)$'
    ax.text(5.5, 0.5, eq, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # Title
    ax.text(5.5, 3.5, 'Hierarchical Spectral Composition', ha='center', va='center',
            fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'architecture.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'architecture.png'), bbox_inches='tight', dpi=300)
    print("Saved: architecture.pdf/png")
    plt.close()


def figure2_sparsity():
    """
    Figure 2: Sparsity Comparison Across Phases
    Bar chart showing increasing sparsity with n
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    phases = ['Phase 2\n($n=2$)', 'Phase 3\n($n=3$)', 'Phase 4\n($n=4$)']
    sparsity = [37.5, 41.25, 56.25]
    basis_dim = [4, 8, 16]

    colors = ['#4CAF50', '#2196F3', '#9C27B0']
    bars = ax.bar(phases, sparsity, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val, dim in zip(bars, sparsity, basis_dim):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'dim={dim}', ha='center', va='center', fontsize=9, color='white',
                fontweight='bold')

    ax.set_ylabel('Zero Coefficients (%)', fontsize=11)
    ax.set_title('Spectral Sparsity Increases with $n$', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add trend annotation
    ax.annotate('', xy=(2.3, 58), xytext=(0.7, 40),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
    ax.text(1.5, 52, 'Increasing\nconcentration', ha='center', va='center',
            fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sparsity.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'sparsity.png'), bbox_inches='tight', dpi=300)
    print("Saved: sparsity.pdf/png")
    plt.close()


def figure3_phase3_heatmap():
    """
    Figure 3: Phase 3 Ternary Mask Heatmap
    Shows all 10 operations with their 8-dim masks
    """
    # Load Phase 3 results
    json_path = './boolean_fourier/checkpoints/phase3_final/phase3_final_results.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract masks in order
    operations = [
        'parity_3', 'majority_3', 'and_3', 'or_3',
        'xor_ab_xor_c', 'and_ab_or_c', 'or_ab_and_c',
        'implies_ab_c', 'xor_and_ab_c', 'and_xor_ab_c'
    ]

    basis = ['$1$', '$a$', '$b$', '$c$', '$ab$', '$ac$', '$bc$', '$abc$']

    # Create mask matrix
    mask_matrix = np.array([data['optimal_masks'][op] for op in operations])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Custom colormap: red (-1), white (0), blue (+1)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E53935', '#FFFFFF', '#1E88E5'])

    im = ax.imshow(mask_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add cell values
    for i in range(len(operations)):
        for j in range(8):
            val = mask_matrix[i, j]
            color = 'white' if val != 0 else 'black'
            text = '+1' if val == 1 else ('-1' if val == -1 else '0')
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                   color=color, fontweight='bold')

    # Labels
    ax.set_xticks(range(8))
    ax.set_xticklabels(basis, fontsize=10)
    ax.set_yticks(range(len(operations)))

    # Clean operation names for display
    op_labels = [op.replace('_', '\\_') for op in operations]
    ax.set_yticklabels(op_labels, fontsize=9, family='monospace')

    ax.set_xlabel('Boolean Fourier Basis Characters', fontsize=11)
    ax.set_ylabel('Operation', fontsize=11)
    ax.set_title('Phase 3 Ternary Masks ($n=3$, 100% Accuracy)', fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.8)
    cbar.ax.set_yticklabels(['-1', '0', '+1'])
    cbar.set_label('Coefficient Value', fontsize=10)

    # Add sparsity annotation
    sparsity = data['mean_sparsity'] * 100
    ax.text(0.98, 0.02, f'Sparsity: {sparsity:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase3_heatmap.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase3_heatmap.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase3_heatmap.pdf/png")
    plt.close()


def figure4_phase1_masks():
    """
    Figure 4: Phase 1 Ternary Masks (all 16 operations)
    Compact heatmap showing 2-variable operations
    """
    # All 16 operations with their masks
    operations = {
        'FALSE': [-1, 0, 0, 0],
        'TRUE': [1, 0, 0, 0],
        'AND': [-1, 1, 1, 1],
        'NAND': [1, -1, -1, -1],
        'OR': [1, 1, 1, -1],
        'NOR': [-1, -1, -1, 1],
        'XOR': [0, 0, 0, -1],
        'XNOR': [0, 0, 0, 1],
        'A': [0, 1, 0, 0],
        'NOT_A': [0, -1, 0, 0],
        'B': [0, 0, 1, 0],
        'NOT_B': [0, 0, -1, 0],
        'A_AND_NOT_B': [-1, 1, -1, -1],
        'A_OR_NOT_B': [1, 1, -1, 1],
        'NOT_A_AND_B': [-1, -1, 1, -1],
        'NOT_A_OR_B': [1, -1, 1, 1],
    }

    basis = ['$1$', '$a$', '$b$', '$ab$']
    op_names = list(operations.keys())
    mask_matrix = np.array(list(operations.values()))

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E53935', '#FFFFFF', '#1E88E5'])

    im = ax.imshow(mask_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add cell values
    for i in range(len(op_names)):
        for j in range(4):
            val = mask_matrix[i, j]
            color = 'white' if val != 0 else 'black'
            text = '+1' if val == 1 else ('-1' if val == -1 else '0')
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                   color=color, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(basis, fontsize=11)
    ax.set_yticks(range(len(op_names)))
    ax.set_yticklabels(op_names, fontsize=9, family='monospace')

    ax.set_xlabel('Boolean Fourier Basis', fontsize=11)
    ax.set_title('Phase 1 Ternary Masks ($n=2$, All 16 Operations)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.6)
    cbar.ax.set_yticklabels(['-1', '0', '+1'])

    # Calculate sparsity
    sparsity = np.mean(mask_matrix == 0) * 100
    ax.text(0.98, 0.02, f'Sparsity: {sparsity:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_heatmap.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_heatmap.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase1_heatmap.pdf/png")
    plt.close()


def figure5_benchmark():
    """
    Figure 5: Benchmark Performance
    Bar chart comparing backends
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    backends = ['JAX/GPU\n(RTX 3080)', 'NumPy/CPU\n(INT8)']
    throughput = [10959.40, 28.84]

    colors = ['#4CAF50', '#FF9800']
    bars = ax.bar(backends, throughput, color=colors, edgecolor='black', linewidth=1.2)

    # Log scale
    ax.set_yscale('log')

    # Add value labels
    for bar, val in zip(bars, throughput):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Throughput (MOps/s)', fontsize=11)
    ax.set_title('Phase 3 Inference Throughput\n(batch=100k, bits=64)', fontsize=12, fontweight='bold')
    ax.set_ylim(10, 50000)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add speedup annotation
    speedup = throughput[0] / throughput[1]
    ax.annotate(f'{speedup:.0f}x', xy=(0.5, 1000), fontsize=14, fontweight='bold',
                ha='center', color='#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'benchmark.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'benchmark.png'), bbox_inches='tight', dpi=300)
    print("Saved: benchmark.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("Generating figures for Boolean Fourier paper...")
    print("=" * 50)

    figure1_architecture()
    figure2_sparsity()
    figure3_phase3_heatmap()
    figure4_phase1_masks()
    figure5_benchmark()

    print("=" * 50)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
