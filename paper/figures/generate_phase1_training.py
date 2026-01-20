"""
Generate Phase 1 Training Figures for Paper
============================================

Creates figures showing:
1. Training dynamics (loss & accuracy over steps)
2. XOR mask evolution (parity character emergence)
3. Final ternary masks for all 4 operations
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


def figure_phase1_training_dynamics():
    """
    Figure: Training dynamics showing sequential operation learning.
    Based on train_phase1_fixed.py results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Training data from the run (extracted from logs)
    # Each operation: 5000 steps, logged every 500 steps
    steps = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

    # XOR training data
    xor_acc = np.array([49.75, 24.99, 49.75, 50.15, 50.15, 50.15, 100.0, 100.0, 100.0, 100.0, 100.0])
    xor_loss = np.array([1.18, 0.88, 0.85, 0.88, 0.91, 0.95, 0.47, 0.0, 0.0, 0.0, 0.0])
    xor_purity = np.array([0.19, 0.42, 0.44, 0.44, 0.42, 0.41, 0.50, 1.0, 1.0, 1.0, 1.0])

    # AND training (converges very fast)
    and_acc = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    and_loss = np.array([2.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # OR training
    or_acc = np.array([24.99, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    or_loss = np.array([0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # IMPLIES training
    implies_acc = np.array([49.85, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    implies_loss = np.array([1.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Plot XOR (most interesting)
    ax = axes[0, 0]
    ax.plot(steps, xor_acc, 'b-', linewidth=2, marker='o', markersize=4, label='Accuracy')
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Target (100%)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('XOR Training (Parity Operation)')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.annotate('Parity emerges', xy=(3000, 100), xytext=(2000, 70),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # XOR purity evolution
    ax = axes[0, 1]
    ax.plot(steps, xor_purity, 'r-', linewidth=2, marker='s', markersize=4)
    ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Target (95%)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Parity Concentration')
    ax.set_title('XOR Spectral Purity (ab coefficient)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.fill_between(steps, 0, xor_purity, alpha=0.3, color='red')

    # AND/OR comparison
    ax = axes[1, 0]
    ax.plot(steps, and_acc, 'g-', linewidth=2, marker='^', markersize=4, label='AND')
    ax.plot(steps, or_acc, 'm-', linewidth=2, marker='v', markersize=4, label='OR')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('AND & OR Training')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.annotate('Fast convergence\n(<500 steps)', xy=(500, 100), xytext=(1500, 60),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # IMPLIES and loss overview
    ax = axes[1, 1]
    ax.plot(steps, implies_acc, 'c-', linewidth=2, marker='d', markersize=4, label='IMPLIES Acc')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('IMPLIES Training')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_training.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_training.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase1_training.pdf/png")
    plt.close()


def figure_phase1_masks():
    """
    Figure: Final ternary masks for all 4 Phase 1 operations.
    Shows the learned spectral coefficients.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    # Learned masks (from training run, code encoding: -1=TRUE, +1=FALSE)
    operations = ['XOR', 'AND', 'OR', 'IMPLIES']
    basis = ['$1$', '$a$', '$b$', '$ab$']

    # Masks in code encoding (-1=TRUE, +1=FALSE)
    mask_matrix = np.array([
        [0, 0, 0, 1],      # XOR: parity only
        [1, 1, 1, -1],     # AND
        [-1, 1, 1, 1],     # OR
        [-1, -1, 1, -1],   # IMPLIES
    ])

    # Custom colormap: red (-1), white (0), blue (+1)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E53935', '#FFFFFF', '#1E88E5'])

    im = ax.imshow(mask_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add cell values
    for i in range(len(operations)):
        for j in range(4):
            val = mask_matrix[i, j]
            color = 'white' if val != 0 else 'black'
            text = '+1' if val == 1 else ('-1' if val == -1 else '0')
            ax.text(j, i, text, ha='center', va='center', fontsize=11,
                   color=color, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(basis, fontsize=11)
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations, fontsize=11, family='monospace')

    ax.set_xlabel('Boolean Fourier Basis', fontsize=11)
    ax.set_title('Phase 1 Learned Ternary Masks ($n=2$, 100% Accuracy)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.8)
    cbar.ax.set_yticklabels(['-1', '0', '+1'])
    cbar.set_label('Coefficient Value', fontsize=10)

    # Add sparsity annotation
    sparsity = np.mean(mask_matrix == 0) * 100
    ax.text(0.98, 0.02, f'Sparsity: {sparsity:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add encoding note
    ax.text(0.02, -0.18, 'Encoding: $-1$ = TRUE, $+1$ = FALSE',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_masks.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_masks.png'), bbox_inches='tight', dpi=300)
    print("Saved: phase1_masks.pdf/png")
    plt.close()


def figure_xor_parity_emergence():
    """
    Figure: XOR parity character emergence during training.
    Shows how the ab coefficient grows while others shrink.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    steps = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

    # Approximate mask evolution (from training logs)
    # [c_1, c_a, c_b, c_ab] over time
    c_1 = np.array([-0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    c_a = np.array([0.38, 0.50, 0.73, 0.88, 0.95, 0.97, 1.0, 0.0, 0.0, 0.0, 0.0])
    c_b = np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    c_ab = np.array([-0.11, -0.36, -0.57, -0.69, -0.69, -0.68, 1.0, 1.0, 1.0, 1.0, 1.0])

    ax.plot(steps, np.abs(c_1), 'g-', linewidth=2, marker='o', markersize=4, label='$|c_1|$ (constant)')
    ax.plot(steps, np.abs(c_a), 'b-', linewidth=2, marker='s', markersize=4, label='$|c_a|$')
    ax.plot(steps, np.abs(c_b), 'c-', linewidth=2, marker='^', markersize=4, label='$|c_b|$')
    ax.plot(steps, np.abs(c_ab), 'r-', linewidth=2, marker='d', markersize=6, label='$|c_{ab}|$ (parity)')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Coefficient Magnitude')
    ax.set_title('XOR Mask Evolution: Parity Character Emergence', fontsize=12, fontweight='bold')
    ax.legend(loc='center right')
    ax.set_ylim(-0.05, 1.15)

    # Add annotations
    ax.annotate('Parity dominates', xy=(3500, 1.0), xytext=(4200, 0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    ax.annotate('Other coefficients\n→ 0', xy=(3500, 0.0), xytext=(4200, 0.25),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    ax.fill_between(steps, 0, np.abs(c_ab), alpha=0.2, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'xor_parity_emergence.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'xor_parity_emergence.png'), bbox_inches='tight', dpi=300)
    print("Saved: xor_parity_emergence.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("Generating Phase 1 training figures...")
    print("=" * 50)

    figure_phase1_training_dynamics()
    figure_phase1_masks()
    figure_xor_parity_emergence()

    print("=" * 50)
    print("All Phase 1 figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
