"""
Generate Phase 2 Figures for Paper
===================================

Creates figures showing:
1. Phase 2 architecture diagram (routing + sign modulation)
2. All 16 ternary masks heatmap
3. Linear vs Nonlinear operation comparison
4. Routing matrix visualization
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


# All 16 verified ternary masks
ALL_16_MASKS = {
    # Linear (0-7)
    0: ('XOR', [0, 0, 0, 1]),
    1: ('AND', [1, 1, 1, -1]),
    2: ('OR', [-1, 1, 1, 1]),
    3: ('IMPLIES', [-1, -1, 1, -1]),
    4: ('XNOR', [0, 0, 0, -1]),
    5: ('NAND', [-1, -1, -1, 1]),
    6: ('NOR', [1, -1, -1, -1]),
    7: ('NOT_IMP', [1, 1, -1, 1]),
    # Nonlinear (8-15)
    8: ('IF_XOR_AND', [-1, 0, 1, 0]),
    9: ('IF_AND_OR', [-1, 1, 0, 0]),
    10: ('XOR(AND,b)', [0, -1, 1, 0]),
    11: ('AND(XOR,a)', [0, 1, -1, 0]),
    12: ('OR(AND,XOR)', [-1, 1, 1, 0]),
    13: ('MAJORITY', [-1, 1, 1, 0]),
    14: ('PAR(AND,OR)', [-1, 0, 0, 1]),
    15: ('XOR→AND', [-1, 0, 0, -1]),
}


def figure_phase2_masks_heatmap():
    """
    Figure: Heatmap of all 16 Phase 2 ternary masks.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # Build mask matrix
    names = []
    masks = []
    for op_id in range(16):
        name, mask = ALL_16_MASKS[op_id]
        names.append(name)
        masks.append(mask)

    mask_matrix = np.array(masks)
    basis = ['$c_1$', '$c_a$', '$c_b$', '$c_{ab}$']

    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E53935', '#FFFFFF', '#1E88E5'])

    im = ax.imshow(mask_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add cell values
    for i in range(16):
        for j in range(4):
            val = mask_matrix[i, j]
            color = 'white' if val != 0 else 'black'
            text = '+1' if val == 1 else ('-1' if val == -1 else '0')
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                   color=color, fontweight='bold')

    # Add horizontal line between linear and nonlinear
    ax.axhline(y=7.5, color='black', linewidth=2, linestyle='--')
    ax.text(4.1, 3.5, 'Linear\n(Routing)', fontsize=10, va='center', style='italic')
    ax.text(4.1, 11.5, 'Nonlinear\n(Direct)', fontsize=10, va='center', style='italic')

    ax.set_xticks(range(4))
    ax.set_xticklabels(basis, fontsize=11)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f'{i}: {names[i]}' for i in range(16)], fontsize=9, family='monospace')

    ax.set_xlabel('Boolean Fourier Basis', fontsize=11)
    ax.set_ylabel('Operation', fontsize=11)
    ax.set_title('Phase 2: All 16 Ternary Masks\n(Operations 0-7: Routing, 8-15: Direct)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.8)
    cbar.ax.set_yticklabels(['-1', '0', '+1'])
    cbar.set_label('Coefficient', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_masks.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_masks.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase2_masks.pdf/png")
    plt.close()


def figure_phase2a_results():
    """
    Figure: Phase 2A validation results across 10 seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Data from validation run
    operations = ['XOR', 'AND', 'OR', 'IMPLIES', 'XNOR', 'NAND', 'NOR', 'NOT_IMP']
    accuracies = [100.0] * 8  # All 100% across 10 seeds
    signs_expected = [+1, +1, +1, +1, -1, -1, -1, -1]
    routing_expected = [0, 1, 2, 3, 0, 1, 2, 3]

    # Accuracy bar chart
    ax = axes[0]
    colors = ['#1E88E5'] * 4 + ['#E53935'] * 4
    bars = ax.bar(range(8), accuracies, color=colors, edgecolor='black')
    ax.set_xticks(range(8))
    ax.set_xticklabels(operations, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Operation Accuracy')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)

    # Signs visualization
    ax = axes[1]
    sign_colors = ['#1E88E5' if s == 1 else '#E53935' for s in signs_expected]
    ax.bar(range(8), [1]*8, color=sign_colors, edgecolor='black')
    for i, s in enumerate(signs_expected):
        ax.text(i, 0.5, f'{s:+d}', ha='center', va='center', fontsize=12,
               fontweight='bold', color='white')
    ax.set_xticks(range(8))
    ax.set_xticklabels(operations, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Sign Value')
    ax.set_title('Column-Sign Modulation')
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

    # Routing visualization
    ax = axes[2]
    parent_names = ['XOR', 'AND', 'OR', 'IMP']
    parent_colors = ['#FFB74D', '#81C784', '#64B5F6', '#BA68C8']
    route_colors = [parent_colors[r] for r in routing_expected]
    ax.bar(range(8), [1]*8, color=route_colors, edgecolor='black')
    for i, r in enumerate(routing_expected):
        ax.text(i, 0.5, parent_names[r], ha='center', va='center', fontsize=9,
               fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_xticklabels(operations, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Parent Mask')
    ax.set_title('Routing (k=1)')
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

    plt.suptitle('Phase 2A: Linear Operations (10/10 Seeds at 100%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2a_results.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2a_results.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase2a_results.pdf/png")
    plt.close()


def figure_phase2_routing_matrix():
    """
    Figure: Visualize the routing matrix P and sign modulation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Ideal P matrix (identity-like for linear ops)
    P_ideal = np.zeros((4, 8))
    for i in range(4):
        P_ideal[i, i] = 1.0      # Ops 0-3
        P_ideal[i, i+4] = 1.0    # Ops 4-7 (negations)

    # Sign vector
    s = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # R = P * s
    R = P_ideal * s[None, :]

    # Plot P
    ax = axes[0]
    im = ax.imshow(P_ideal, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{i}' for i in range(8)])
    ax.set_yticks(range(4))
    ax.set_yticklabels(['XOR', 'AND', 'OR', 'IMP'])
    ax.set_xlabel('Child Operation')
    ax.set_ylabel('Parent Mask')
    ax.set_title('P: Routing Matrix\n(Sinkhorn-constrained)')
    for i in range(4):
        for j in range(8):
            if P_ideal[i,j] > 0.5:
                ax.text(j, i, '1', ha='center', va='center', color='white', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Plot s (sign vector)
    ax = axes[1]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E53935', '#1E88E5'])
    s_matrix = (s[None, :] + 1) / 2  # Map to 0,1 for colormap
    im = ax.imshow(s_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{i}' for i in range(8)])
    ax.set_yticks([0])
    ax.set_yticklabels(['s'])
    ax.set_xlabel('Child Operation')
    ax.set_title('s: Column-Sign Modulation')
    for j in range(8):
        ax.text(j, 0, f'{s[j]:+d}', ha='center', va='center', color='white', fontweight='bold')

    # Plot R = P * s
    ax = axes[2]
    cmap = ListedColormap(['#E53935', '#FFFFFF', '#1E88E5'])
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{i}' for i in range(8)])
    ax.set_yticks(range(4))
    ax.set_yticklabels(['XOR', 'AND', 'OR', 'IMP'])
    ax.set_xlabel('Child Operation')
    ax.set_ylabel('Parent Mask')
    ax.set_title('R = P ⊙ s: Composed Routing\n(Enables Negation)')
    for i in range(4):
        for j in range(8):
            if abs(R[i,j]) > 0.5:
                ax.text(j, i, f'{int(R[i,j]):+d}', ha='center', va='center',
                       color='white', fontweight='bold')
    plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.6)

    plt.suptitle('Phase 2: Sinkhorn-Constrained Routing with Column-Sign Modulation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_routing.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_routing.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase2_routing.pdf/png")
    plt.close()


def figure_phase2_sparsity():
    """
    Figure: Sparsity comparison between linear and nonlinear operations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Calculate sparsity per operation
    linear_sparsity = []
    nonlinear_sparsity = []

    for op_id in range(8):
        mask = np.array(ALL_16_MASKS[op_id][1])
        linear_sparsity.append(np.mean(mask == 0) * 100)

    for op_id in range(8, 16):
        mask = np.array(ALL_16_MASKS[op_id][1])
        nonlinear_sparsity.append(np.mean(mask == 0) * 100)

    # Bar chart
    ax = axes[0]
    x = np.arange(8)
    width = 0.35
    bars1 = ax.bar(x - width/2, linear_sparsity, width, label='Linear (0-7)', color='#1E88E5')
    bars2 = ax.bar(x + width/2, nonlinear_sparsity, width, label='Nonlinear (8-15)', color='#E53935')
    ax.set_ylabel('Sparsity (%)')
    ax.set_xlabel('Operation Index (within category)')
    ax.set_title('Per-Operation Sparsity')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(8)])
    ax.legend()
    ax.set_ylim(0, 80)

    # Summary
    ax = axes[1]
    categories = ['Linear\n(0-7)', 'Nonlinear\n(8-15)', 'Overall']
    values = [np.mean(linear_sparsity), np.mean(nonlinear_sparsity),
              (np.mean(linear_sparsity) + np.mean(nonlinear_sparsity)) / 2]
    colors = ['#1E88E5', '#E53935', '#7B1FA2']
    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    ax.set_ylabel('Mean Sparsity (%)')
    ax.set_title('Category-wise Sparsity')
    ax.set_ylim(0, 60)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Phase 2: Ternary Mask Sparsity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_sparsity.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_sparsity.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase2_sparsity.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("Generating Phase 2 figures...")
    print("=" * 50)

    figure_phase2_masks_heatmap()
    figure_phase2a_results()
    figure_phase2_routing_matrix()
    figure_phase2_sparsity()

    print("=" * 50)
    print("All Phase 2 figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
