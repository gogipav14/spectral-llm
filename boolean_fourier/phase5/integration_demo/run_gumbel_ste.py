"""Run Gumbel-STE router experiment only."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from demo_router_learning import run_integration_demo
import json

if __name__ == '__main__':
    print("="*60)
    print("GUMBEL-STE ROUTER EXPERIMENT")
    print("="*60)

    results = run_integration_demo(
        n_train=10000,
        n_test=5000,
        n_steps=5000,
        batch_size=128,
        lr=1e-2,
        seed=0,
        router_type='gumbel-ste'
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / 'integration_demo_gumbel-ste.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved: {output_path}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
