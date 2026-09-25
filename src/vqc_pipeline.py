"""
VQC Pipeline — Real-Life Usage of Saved .npz NN Models
=======================================================
Loads trained_nn_model_depth_4.npz and trained_nn_model_depth_10.npz
and provides 4 ready-to-use pipelines:

  1. VQE         — NN warm-start → Adam fine-tune → ground state energy
  2. Benchmark   — test saved models across different J/h Hamiltonians
  3. ParamGen    — on-demand parameter generation + ranking (no training)
  4. Export      — save parameters as CSV / JSON / PyTorch-ready dict

Usage:
  python vqc_pipeline.py --pipeline vqe      --depth 4
  python vqc_pipeline.py --pipeline vqe      --depth 10
  python vqc_pipeline.py --pipeline benchmark
  python vqc_pipeline.py --pipeline paramgen --depth 4  --n_candidates 50
  python vqc_pipeline.py --pipeline export   --depth 10 --format json
  python vqc_pipeline.py --pipeline all                         (runs everything)
"""

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# GLOBAL CONFIG  (must match training run)
# ─────────────────────────────────────────
N_QUBITS   = 6
J_DEFAULT  = 1.0
H_DEFAULT  = 0.5
SEED       = 42
MODEL_FILES = {
    4:  'trained_nn_model_depth_4.npz',
    10: 'trained_nn_model_depth_10.npz',
}

np.random.seed(SEED)


# ══════════════════════════════════════════
# CORE HELPERS  (shared by all pipelines)
# ══════════════════════════════════════════

def make_hamiltonian(n, J=1.0, h=0.5):
    terms = []
    for i in range(n):
        j = (i + 1) % n
        p = ['I'] * n; p[i] = 'Z'; p[j] = 'Z'
        terms.append((''.join(reversed(p)), -J))
    for i in range(n):
        p = ['I'] * n; p[i] = 'X'
        terms.append((''.join(reversed(p)), -h))
    return SparsePauliOp.from_list(terms)


def exact_ground_energy(n, J, h):
    Hmat = make_hamiltonian(n, J, h).to_matrix()
    return float(np.linalg.eigvalsh(Hmat)[0])


def energy(params, n_layers, Hmat):
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(N_QUBITS)
    idx = 0
    for _ in range(n_layers):
        for q in range(N_QUBITS):
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
    sv = Statevector(qc).data
    return float(np.real(np.conj(sv) @ Hmat @ sv))


def gradient(params, n_layers, Hmat):
    g = np.zeros_like(params)
    for i in range(len(params)):
        pp = params.copy(); pp[i] += np.pi / 2
        pm = params.copy(); pm[i] -= np.pi / 2
        g[i] = (energy(pp, n_layers, Hmat) - energy(pm, n_layers, Hmat)) / 2
    return g


def adam_step(params, g, m, v, t, lr=0.03):
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    t += 1
    params = params - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return params, m, v, t


def n_params(n_layers):
    return 2 * N_QUBITS * n_layers


# ══════════════════════════════════════════
# MODEL LOADER & INFERENCE
# ══════════════════════════════════════════

def load_model(depth):
    path = MODEL_FILES[depth]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found.\n"
            f"Run the training script (1.py) first to generate it."
        )
    m = np.load(path)
    print(f"  [✓] Loaded model: {path}  "
          f"(W1={m['W1'].shape}, W2={m['W2'].shape})")
    return m['W1'], m['b1'], m['W2'], m['b2']


def nn_forward(z, W1, b1, W2, b2):
    """2-layer ReLU + Tanh*pi network — identical to training."""
    h = np.maximum(0, z @ W1 + b1)
    return np.tanh(h @ W2 + b2) * np.pi


def generate_params(W1, b1, W2, b2, n=1, seed=None):
    """Sample n parameter vectors from the trained NN."""
    rng = np.random.default_rng(seed)
    thetas = []
    for _ in range(n):
        z = rng.standard_normal(W1.shape[0])
        thetas.append(nn_forward(z, W1, b1, W2, b2))
    return thetas  # list of np.arrays


# ══════════════════════════════════════════
# PIPELINE 1 — VQE (NN warm-start + Adam)
# ══════════════════════════════════════════

def pipeline_vqe(depth, J=J_DEFAULT, h=H_DEFAULT,
                 n_candidates=20, adam_iters=150, lr=0.03, plot=True):
    """
    Real-life VQE pipeline:
      1. Load NN model for given depth
      2. Sample n_candidates parameter sets
      3. Evaluate energies, pick best starting point
      4. Fine-tune with Adam optimizer
      5. Compare vs exact ground state
    """
    print(f"\n{'━'*55}")
    print(f"  PIPELINE 1 — VQE  (depth={depth}, J={J}, h={h})")
    print(f"{'━'*55}")

    Hmat    = make_hamiltonian(N_QUBITS, J, h).to_matrix()
    E_exact = exact_ground_energy(N_QUBITS, J, h)
    print(f"  Exact ground energy : {E_exact:.6f}")

    # ── Step 1: Load model ──
    W1, b1, W2, b2 = load_model(depth)

    # ── Step 2: Candidate generation ──
    print(f"  Generating {n_candidates} candidate parameter sets …")
    thetas = generate_params(W1, b1, W2, b2, n=n_candidates, seed=SEED)
    cand_energies = [energy(t, depth, Hmat) for t in thetas]

    best_idx   = int(np.argmin(cand_energies))
    best_theta = thetas[best_idx].copy()
    print(f"  Best candidate energy : {cand_energies[best_idx]:.6f}  "
          f"(err={abs(cand_energies[best_idx]-E_exact):.5f})")

    # ── Step 3: Adam fine-tune ──
    print(f"  Fine-tuning with Adam ({adam_iters} iters) …")
    npt = n_params(depth)
    p   = best_theta
    m   = np.zeros(npt); v = np.zeros(npt); t = 0
    history = []

    t0 = time.time()
    for it in range(adam_iters):
        e = energy(p, depth, Hmat)
        history.append(e)
        g = gradient(p, depth, Hmat)
        p, m, v, t = adam_step(p, g, m, v, t, lr=lr)
        if (it + 1) % 30 == 0:
            print(f"    iter {it+1:4d}  E={e:.6f}  err={abs(e-E_exact):.5f}")

    elapsed = time.time() - t0
    final_e = history[-1]
    print(f"\n  ── VQE Result ──")
    print(f"  Final energy   : {final_e:.6f}")
    print(f"  Exact energy   : {E_exact:.6f}")
    print(f"  Error          : {abs(final_e - E_exact):.6f}")
    print(f"  Time           : {elapsed:.1f}s")
    print(f"  Optimal params saved → vqe_optimal_params_depth{depth}.npy")
    np.save(f'vqe_optimal_params_depth{depth}.npy', p)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history, lw=2, label='NN warm-start + Adam')
        ax.axhline(E_exact, ls=':', lw=2, c='k', label=f'Exact E₀={E_exact:.3f}')
        ax.set_xlabel('Adam Iteration'); ax.set_ylabel('Energy ⟨H⟩')
        ax.set_title(f'VQE Convergence — depth={depth}, J={J}, h={h}')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f'vqe_convergence_depth{depth}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved → {fname}")

    return {'depth': depth, 'final_energy': final_e,
            'exact_energy': E_exact, 'error': abs(final_e - E_exact),
            'optimal_params': p, 'history': history}


# ══════════════════════════════════════════
# PIPELINE 2 — BENCHMARK across Hamiltonians
# ══════════════════════════════════════════

def pipeline_benchmark(n_candidates=15):
    """
    Tests both saved models across a grid of (J, h) values.
    Shows how well the NN generalises to unseen Hamiltonians.
    """
    print(f"\n{'━'*55}")
    print(f"  PIPELINE 2 — BENCHMARK  (transfer across Hamiltonians)")
    print(f"{'━'*55}")

    J_vals = [0.5, 1.0, 1.5, 2.0]
    h_vals = [0.25, 0.5, 1.0]

    results = {d: [] for d in MODEL_FILES}

    for depth in MODEL_FILES:
        print(f"\n  Loading model for depth={depth} …")
        W1, b1, W2, b2 = load_model(depth)

        for J in J_vals:
            for h in h_vals:
                Hmat    = make_hamiltonian(N_QUBITS, J, h).to_matrix()
                E_exact = exact_ground_energy(N_QUBITS, J, h)

                thetas  = generate_params(W1, b1, W2, b2, n=n_candidates, seed=SEED)
                energies = [energy(t, depth, Hmat) for t in thetas]
                best_e   = min(energies)
                err      = abs(best_e - E_exact)

                results[depth].append({
                    'J': J, 'h': h,
                    'best_energy': best_e,
                    'exact_energy': E_exact,
                    'error': err,
                })
                print(f"    depth={depth} J={J:.1f} h={h:.2f} | "
                      f"Best E={best_e:.4f}  Exact={E_exact:.4f}  Err={err:.4f}")

    # ── Plot heatmap of errors ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, depth in zip(axes, MODEL_FILES):
        grid = np.zeros((len(J_vals), len(h_vals)))
        for r in results[depth]:
            i = J_vals.index(r['J'])
            j = h_vals.index(r['h'])
            grid[i, j] = r['error']
        im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=grid.max())
        ax.set_xticks(range(len(h_vals))); ax.set_xticklabels(h_vals)
        ax.set_yticks(range(len(J_vals))); ax.set_yticklabels(J_vals)
        ax.set_xlabel('h (transverse field)'); ax.set_ylabel('J (coupling)')
        ax.set_title(f'Energy Error — depth={depth}  (greener = better)')
        plt.colorbar(im, ax=ax, label='|E_NN - E_exact|')
        for i in range(len(J_vals)):
            for j in range(len(h_vals)):
                ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center',
                        fontsize=8, color='black')
    fig.suptitle('Benchmark: NN Model Transfer Across Hamiltonians', fontweight='bold')
    plt.tight_layout()
    plt.savefig('benchmark_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Benchmark heatmap saved → benchmark_heatmap.png")

    # Save raw results
    with open('benchmark_results.json', 'w') as f:
        json.dump({str(d): results[d] for d in results}, f, indent=2)
    print(f"  Raw results saved → benchmark_results.json")
    return results


# ══════════════════════════════════════════
# PIPELINE 3 — Parameter Generation & Ranking
# ══════════════════════════════════════════

def pipeline_paramgen(depth, n_candidates=50,
                      J=J_DEFAULT, h=H_DEFAULT, top_k=5):
    """
    On-demand parameter generation — no extra training needed.
    Generates n_candidates sets, ranks by energy, returns top_k.
    Useful for seeding other optimisers (SPSA, COBYLA, etc.)
    """
    print(f"\n{'━'*55}")
    print(f"  PIPELINE 3 — PARAM GEN  (depth={depth}, n={n_candidates})")
    print(f"{'━'*55}")

    Hmat    = make_hamiltonian(N_QUBITS, J, h).to_matrix()
    E_exact = exact_ground_energy(N_QUBITS, J, h)
    W1, b1, W2, b2 = load_model(depth)

    print(f"  Sampling {n_candidates} parameter sets …")
    thetas   = generate_params(W1, b1, W2, b2, n=n_candidates)
    energies = np.array([energy(t, depth, Hmat) for t in thetas])

    ranked   = np.argsort(energies)
    top_t    = [thetas[i] for i in ranked[:top_k]]
    top_e    = energies[ranked[:top_k]]

    print(f"\n  Top-{top_k} parameter sets by energy:")
    print(f"  {'Rank':>4} | {'Energy':>10} | {'Error':>10} | Params shape")
    print(f"  {'-'*50}")
    for rank, (e, t) in enumerate(zip(top_e, top_t), 1):
        print(f"  {rank:>4} | {e:>10.5f} | {abs(e-E_exact):>10.5f} | {t.shape}")

    # Save top-k as .npy files + CSV
    out = {'rank': [], 'energy': [], 'error': []}
    for rank, (e, t) in enumerate(zip(top_e, top_t), 1):
        fname = f'paramgen_depth{depth}_rank{rank}.npy'
        np.save(fname, t)
        out['rank'].append(rank)
        out['energy'].append(float(e))
        out['error'].append(float(abs(e - E_exact)))

    import csv
    with open(f'paramgen_depth{depth}_ranking.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank', 'energy', 'error'])
        w.writeheader(); w.writerows(
            [{'rank': r, 'energy': e, 'error': err}
             for r, e, err in zip(out['rank'], out['energy'], out['error'])]
        )
    print(f"\n  Saved top-{top_k} param files: paramgen_depth{depth}_rank1.npy … rank{top_k}.npy")
    print(f"  Ranking CSV saved → paramgen_depth{depth}_ranking.csv")

    # Distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(energies, bins=20, color='steelblue', alpha=0.7, label='All candidates')
    ax.axvline(top_e[0], c='green', lw=2, ls='--', label=f'Best NN (rank 1)')
    ax.axvline(E_exact,  c='red',   lw=2, ls=':',  label=f'Exact E₀')
    ax.set_xlabel('Energy ⟨H⟩'); ax.set_ylabel('Count')
    ax.set_title(f'Energy Distribution of NN-Generated Params — depth={depth}')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'paramgen_dist_depth{depth}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plot saved → paramgen_dist_depth{depth}.png")
    return top_t, top_e


# ══════════════════════════════════════════
# PIPELINE 4 — Export to other frameworks
# ══════════════════════════════════════════

def pipeline_export(depth, fmt='json'):
    """
    Export the saved NN model weights into formats usable by:
      - json   : human-readable, load anywhere
      - csv    : easy to inspect in Excel/pandas
      - pytorch: dict ready for torch.nn.Module.load_state_dict()
      - params : just the generated VQC parameters (for Qiskit, PennyLane, etc.)
    """
    print(f"\n{'━'*55}")
    print(f"  PIPELINE 4 — EXPORT  (depth={depth}, format={fmt})")
    print(f"{'━'*55}")

    W1, b1, W2, b2 = load_model(depth)

    if fmt == 'json':
        out = {
            'W1': W1.tolist(), 'b1': b1.tolist(),
            'W2': W2.tolist(), 'b2': b2.tolist(),
            'architecture': {'input': int(W1.shape[0]),
                             'hidden': int(W1.shape[1]),
                             'output': int(W2.shape[1])},
            'activation': 'relu_then_tanh_times_pi',
            'depth': depth, 'n_qubits': N_QUBITS,
        }
        fname = f'model_depth{depth}.json'
        with open(fname, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  JSON export → {fname}")

    elif fmt == 'csv':
        import csv
        for name, arr in [('W1', W1), ('b1', b1), ('W2', W2), ('b2', b2)]:
            fname = f'model_depth{depth}_{name}.csv'
            np.savetxt(fname, arr.reshape(1, -1) if arr.ndim == 1 else arr,
                       delimiter=',')
            print(f"  CSV export → {fname}")

    elif fmt == 'pytorch':
        # Saves a dict that maps directly to a torch.nn.Linear layer
        try:
            import torch
            state = {
                'fc1.weight': torch.tensor(W1.T, dtype=torch.float32),
                'fc1.bias':   torch.tensor(b1,   dtype=torch.float32),
                'fc2.weight': torch.tensor(W2.T, dtype=torch.float32),
                'fc2.bias':   torch.tensor(b2,   dtype=torch.float32),
            }
            fname = f'model_depth{depth}.pt'
            torch.save(state, fname)
            print(f"  PyTorch state_dict saved → {fname}")
            print(f"\n  Usage in PyTorch:")
            print(f"    import torch, torch.nn as nn")
            print(f"    model = nn.Sequential(")
            print(f"        nn.Linear({W1.shape[0]}, {W1.shape[1]}), nn.ReLU(),")
            print(f"        nn.Linear({W2.shape[0]}, {W2.shape[1]}),")
            print(f"    )")
            print(f"    state = torch.load('{fname}')")
            print(f"    model.load_state_dict(state)")
        except ImportError:
            print("  PyTorch not installed — saving as .npz instead.")
            np.savez(f'model_depth{depth}_pt_compat.npz',
                     W1=W1, b1=b1, W2=W2, b2=b2)

    elif fmt == 'params':
        # Export raw VQC parameters for use in Qiskit / PennyLane / Cirq
        thetas = generate_params(W1, b1, W2, b2, n=10, seed=SEED)
        out = {
            'n_qubits': N_QUBITS, 'depth': depth,
            'n_params': int(n_params(depth)),
            'parameter_sets': [t.tolist() for t in thetas],
            'usage_note': (
                'Each entry in parameter_sets is a flat list of VQC rotation angles. '
                'Unpack as: [ry_q0, rz_q0, ry_q1, rz_q1, ...] per layer.'
            )
        }
        fname = f'vqc_params_depth{depth}.json'
        with open(fname, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  VQC parameter export (10 sets) → {fname}")
        print(f"\n  Qiskit usage:")
        print(f"    import json")
        print(f"    data = json.load(open('{fname}'))")
        print(f"    params = data['parameter_sets'][0]  # first set")
        print(f"    # Assign to QuantumCircuit parameters")

    else:
        print(f"  Unknown format '{fmt}'. Choose: json | csv | pytorch | params")


# ══════════════════════════════════════════
# MAIN — CLI Entry Point
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='VQC Pipeline using saved .npz NN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--pipeline', default='all',
                        choices=['vqe', 'benchmark', 'paramgen', 'export', 'all'],
                        help='Which pipeline to run')
    parser.add_argument('--depth',         type=int,   default=4,
                        choices=[4, 10],   help='Circuit depth (model to load)')
    parser.add_argument('--n_candidates',  type=int,   default=20,
                        help='Number of NN-generated parameter candidates')
    parser.add_argument('--adam_iters',    type=int,   default=150,
                        help='Adam optimizer iterations (VQE pipeline)')
    parser.add_argument('--format',        default='json',
                        choices=['json', 'csv', 'pytorch', 'params'],
                        help='Export format (export pipeline)')
    parser.add_argument('--J',   type=float, default=J_DEFAULT, help='ZZ coupling')
    parser.add_argument('--h',   type=float, default=H_DEFAULT, help='Transverse field')

    args = parser.parse_args()

    print("\n" + "═"*55)
    print("  VQC PIPELINE — NN-Assisted Quantum Optimisation")
    print("═"*55)
    print(f"  Qubits : {N_QUBITS}")
    print(f"  Models : {list(MODEL_FILES.values())}")
    print(f"  Pipeline selected : {args.pipeline}")

    if args.pipeline in ('vqe', 'all'):
        for d in ([args.depth] if args.pipeline == 'vqe' else MODEL_FILES):
            pipeline_vqe(d, J=args.J, h=args.h,
                         n_candidates=args.n_candidates,
                         adam_iters=args.adam_iters)

    if args.pipeline in ('benchmark', 'all'):
        pipeline_benchmark(n_candidates=10)

    if args.pipeline in ('paramgen', 'all'):
        for d in ([args.depth] if args.pipeline == 'paramgen' else MODEL_FILES):
            pipeline_paramgen(d, n_candidates=args.n_candidates,
                              J=args.J, h=args.h)

    if args.pipeline in ('export', 'all'):
        fmts = ['json', 'csv', 'params'] if args.pipeline == 'all' else [args.format]
        for d in ([args.depth] if args.pipeline == 'export' else MODEL_FILES):
            for fmt in fmts:
                pipeline_export(d, fmt=fmt)

    print(f"\n{'═'*55}")
    print("  All selected pipelines complete.")
    print("═"*55)


if __name__ == '__main__':
    main()
