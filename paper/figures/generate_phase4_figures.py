"""
Generate Phase 4 Figures for Paper
===================================

Creates figures showing:
1. Phase 4 optimal masks heatmap (10 operations × 16 basis elements)
2. Sparsity analysis
3. Spectral synthesis pipeline diagram
4. Validation summary
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# Phase 4 optimal masks from spectral synthesis (100% accuracy)
# Basis: ['1', 'd', 'c', 'cd', 'b', 'bd', 'bc', 'bcd', 'a', 'ad', 'ac', 'acd', 'ab', 'abd', 'abc', 'abcd']
PHASE4_OPTIMAL_MASKS = {
    'xor_4':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'and_4':          [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'or_4':           [1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    'majority_4':     [1, 1, 1, -1, 1, 0, 0, -1, 1, -1, 0, 0, -1, -1, -1, 1],
    'threshold_3of4': [-1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1],
    'exactly_2of4':   [-1, 0, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1, 0, 0, 1],
    'xor_ab_and_cd':  [-1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'or_ab_xor_cd':   [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1],
    'nested_xor':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'implies_chain':  [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
}

BASIS_NAMES_4VAR = ['1', 'd', 'c', 'cd', 'b', 'bd', 'bc', 'bcd', 'a', 'ad', 'ac', 'acd', 'ab', 'abd', 'abc', 'abcd']
OP_NAMES = list(PHASE4_OPTIMAL_MASKS.keys())


def figure_phase4_masks_heatmap():
    """
    Figure: Heatmap of all 10 Phase 4 optimal masks.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Build mask matrix
    mask_matrix = np.array([PHASE4_OPTIMAL_MASKS[name] for name in OP_NAMES])

    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E53935', '#FFFFFF', '#1E88E5'])

    im = ax.imshow(mask_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add cell values
    for i in range(10):
        for j in range(16):
            val = mask_matrix[i, j]
            color = 'white' if val != 0 else 'black'
            text = '+' if val == 1 else ('-' if val == -1 else '0')
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color=color, fontweight='bold')

    # Add horizontal line between pure and cascade
    ax.axhline(y=5.5, color='black', linewidth=2, linestyle='--')
    ax.text(16.2, 2.5, 'Pure\n4-var', fontsize=10, va='center', style='italic')
    ax.text(16.2, 7.5, 'Cascade\nCompositions', fontsize=10, va='center', style='italic')

    ax.set_xticks(range(16))
    ax.set_xticklabels(BASIS_NAMES_4VAR, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'{i}: {name}' for i, name in enumerate(OP_NAMES)], fontsize=9, family='monospace')

    ax.set_xlabel('Boolean Fourier Basis (16-dim)', fontsize=11)
    ax.set_ylabel('Operation', fontsize=11)
    ax.set_title('Phase 4: Optimal Ternary Masks for 4-Variable Operations\n(All achieve 100% accuracy via Spectral Synthesis)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.8)
    cbar.ax.set_yticklabels(['-1', '0', '+1'])
    cbar.set_label('Coefficient', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_masks.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_masks.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase4_masks.pdf/png")
    plt.close()


def figure_phase4_sparsity():
    """
    Figure: Sparsity analysis for Phase 4 masks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate sparsity per operation
    sparsities = []
    supports = []
    for name in OP_NAMES:
        mask = np.array(PHASE4_OPTIMAL_MASKS[name])
        support = np.sum(mask != 0)
        supports.append(support)
        sparsities.append((16 - support) / 16 * 100)

    # Bar chart of sparsity
    ax = axes[0]
    colors = ['#1E88E5'] * 6 + ['#E53935'] * 4  # Pure vs Cascade
    bars = ax.bar(range(10), sparsities, color=colors, edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_xticklabels([name.replace('_', '\n') for name in OP_NAMES], rotation=0, ha='center', fontsize=7)
    ax.set_ylabel('Sparsity (%)')
    ax.set_xlabel('Operation')
    ax.set_title('Per-Operation Sparsity')
    ax.set_ylim(0, 100)
    ax.axhline(y=36, color='green', linestyle='--', alpha=0.7, label=f'Mean: 36%')
    ax.legend()

    # Support size visualization
    ax = axes[1]
    ax.bar(range(10), supports, color=colors, edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_xticklabels([name.replace('_', '\n') for name in OP_NAMES], rotation=0, ha='center', fontsize=7)
    ax.set_ylabel('Support Size (non-zero coefficients)')
    ax.set_xlabel('Operation')
    ax.set_title('Spectral Support Size')
    ax.set_ylim(0, 18)
    ax.axhline(y=10.3, color='green', linestyle='--', alpha=0.7, label=f'Mean: 10.3/16')
    ax.legend()

    # Add legend for categories
    pure_patch = mpatches.Patch(color='#1E88E5', label='Pure 4-var (0-5)')
    cascade_patch = mpatches.Patch(color='#E53935', label='Cascade (6-9)')
    fig.legend(handles=[pure_patch, cascade_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Phase 4: Spectral Sparsity Analysis\n(16-dim Boolean Fourier Basis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_sparsity.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_sparsity.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase4_sparsity.pdf/png")
    plt.close()


def figure_phase4_validation_summary():
    """
    Figure: Validation summary heatmap (5 seeds × 10 operations).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # All 5 seeds achieved 100% on all operations
    results = np.ones((5, 10)) * 100.0

    im = ax.imshow(results, cmap='Greens', vmin=0, vmax=100, aspect='auto')

    # Add cell values
    for i in range(5):
        for j in range(10):
            ax.text(j, i, '100%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xticks(range(10))
    ax.set_xticklabels([name.replace('_', '\n') for name in OP_NAMES], rotation=0, ha='center', fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'Seed {i}' for i in range(5)])

    ax.set_xlabel('Operation')
    ax.set_ylabel('Test Seed')
    ax.set_title('Phase 4 Final Validation: 100% Accuracy on All Operations\n(5 seeds × 10 operations with synthesized ternary masks)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Accuracy (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_validation.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_validation.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase4_validation.pdf/png")
    plt.close()


def figure_spectral_synthesis_pipeline():
    """
    Figure: Spectral synthesis pipeline diagram.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Pipeline boxes
    boxes = [
        (1, 2, 2.5, 1.5, 'Input\nf(x)', '#E3F2FD'),
        (4, 2, 2.5, 1.5, 'Monte Carlo\nFourier Est.', '#FFF3E0'),
        (7, 2, 2.5, 1.5, 'Ternary\nQuantization', '#E8F5E9'),
        (10, 2, 2.5, 1.5, 'MCMC\nRefinement', '#FCE4EC'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    arrow_style = dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=2)
    ax.annotate('', xy=(4, 2.75), xytext=(3.5, 2.75), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 2.75), xytext=(6.5, 2.75), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 2.75), xytext=(9.5, 2.75), arrowprops=arrow_style)

    # Labels under arrows
    ax.text(3.75, 2.2, 'f̂(S) = E[f·χ_S]', ha='center', fontsize=8, style='italic')
    ax.text(6.75, 2.2, 'sign(f̂)', ha='center', fontsize=8, style='italic')
    ax.text(9.75, 2.2, 'Gibbs + PT', ha='center', fontsize=8, style='italic')

    # Output
    ax.text(13, 2.75, 'Optimal\nTernary\nMask', ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#C8E6C9', edgecolor='black', linewidth=2))

    # Final arrow
    ax.annotate('', xy=(12.2, 2.75), xytext=(12.5, 2.75), arrowprops=arrow_style)

    # Details text
    ax.text(5.25, 0.8, '50K samples', ha='center', fontsize=8, color='gray')
    ax.text(8.25, 0.8, 'threshold=0.1', ha='center', fontsize=8, color='gray')
    ax.text(11.25, 0.8, '500 steps\n4 chains', ha='center', fontsize=8, color='gray')

    ax.set_title('Phase 4: Spectral Synthesis Pipeline (n=4, 16-dim basis)', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_pipeline.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_pipeline.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase4_pipeline.pdf/png")
    plt.close()


def figure_scaling_analysis():
    """
    Figure: Scaling analysis across Phases 2-4.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    phases = ['Phase 2\n(n=2)', 'Phase 3\n(n=3)', 'Phase 4\n(n=4)']
    basis_dims = [4, 8, 16]
    sparsities = [31.2, 39, 36]  # Updated values
    n_ops = [16, 10, 10]

    # Basis dimension vs sparsity
    ax = axes[0]
    ax.plot(basis_dims, sparsities, 'o-', markersize=12, linewidth=2, color='#1E88E5')
    for i, (d, s, p) in enumerate(zip(basis_dims, sparsities, phases)):
        ax.annotate(f'{s:.1f}%', (d, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    ax.set_xlabel('Basis Dimension')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Sparsity vs. Basis Dimension')
    ax.set_xticks(basis_dims)
    ax.set_xticklabels([f'{d}\n(n={i+2})' for i, d in enumerate(basis_dims)])
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3)

    # Bar chart: operations tested
    ax = axes[1]
    colors = ['#1E88E5', '#FF9800', '#4CAF50']
    bars = ax.bar(phases, n_ops, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Operations')
    ax.set_title('Operations per Phase')
    ax.set_ylim(0, 20)
    for bar, n in zip(bars, n_ops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{n}', ha='center', fontsize=12, fontweight='bold')

    # Add accuracy annotation
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
               '100%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.suptitle('Scaling Analysis: Boolean Fourier Logic Across Phases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'scaling_analysis.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'scaling_analysis.png'), bbox_inches='tight', dpi=300)
    print("Saved: scaling_analysis.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("Generating Phase 4 figures...")
    print("=" * 50)

    figure_phase4_masks_heatmap()
    figure_phase4_sparsity()
    figure_phase4_validation_summary()
    figure_spectral_synthesis_pipeline()
    figure_scaling_analysis()

    print("=" * 50)
    print("All Phase 4 figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
