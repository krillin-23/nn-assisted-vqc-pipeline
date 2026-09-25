import argparse
from datetime import datetime
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

N_QUBITS   = 6
J_DEFAULT  = 1.0
H_DEFAULT  = 0.5
SEED       = 42
MODEL_FILES = {
    4:  'trained_nn_model_depth_4.npz',
    10: 'trained_nn_model_depth_10.npz',
}

np.random.seed(SEED)


def make_run_dir(pipeline_name: str) -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', f'{pipeline_name}_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"  [📁] Output folder → {run_dir}/")
    return run_dir


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
    h = np.maximum(0, z @ W1 + b1)
    return np.tanh(h @ W2 + b2) * np.pi


def generate_params(W1, b1, W2, b2, n=1, seed=None):
    rng = np.random.default_rng(seed)
    thetas = []
    for _ in range(n):
        z = rng.standard_normal(W1.shape[0])
        thetas.append(nn_forward(z, W1, b1, W2, b2))
    return thetas


def pipeline_vqe(depth, J=J_DEFAULT, h=H_DEFAULT,
                 n_candidates=20, adam_iters=150, lr=0.03, plot=True):

    print(f"\n{'━'*55}")
    print(f"  PIPELINE 1 — VQE  (depth={depth}, J={J}, h={h})")
    print(f"{'━'*55}")
    run_dir = make_run_dir(f'vqe_depth{depth}')

    Hmat    = make_hamiltonian(N_QUBITS, J, h).to_matrix()
    E_exact = exact_ground_energy(N_QUBITS, J, h)
    print(f"  Exact ground energy : {E_exact:.6f}")

    W1, b1, W2, b2 = load_model(depth)

    print(f"  Generating {n_candidates} candidate parameter sets …")
    thetas = generate_params(W1, b1, W2, b2, n=n_candidates, seed=SEED)
    cand_energies = [energy(t, depth, Hmat) for t in thetas]

    best_idx   = int(np.argmin(cand_energies))
    best_theta = thetas[best_idx].copy()
    print(f"  Best candidate energy : {cand_energies[best_idx]:.6f}  "
          f"(err={abs(cand_energies[best_idx]-E_exact):.5f})")

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
    params_fname = os.path.join(run_dir, f'vqe_optimal_params_depth{depth}.npy')
    print(f"  Optimal params saved → {params_fname}")
    np.save(params_fname, p)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history, lw=2)
        ax.axhline(E_exact, ls=':', lw=2, c='k')
        ax.set_xlabel('Adam Iteration'); ax.set_ylabel('Energy ⟨H⟩')
        ax.set_title(f'VQE Convergence — depth={depth}, J={J}, h={h}')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(run_dir, f'vqe_convergence_depth{depth}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved → {fname}")

    return {'depth': depth, 'final_energy': final_e,
            'exact_energy': E_exact, 'error': abs(final_e - E_exact),
            'optimal_params': p, 'history': history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', default='all',
                        choices=['vqe', 'benchmark', 'paramgen', 'export', 'all'])
    parser.add_argument('--depth', type=int, default=4, choices=[4, 10])
    parser.add_argument('--n_candidates', type=int, default=20)
    parser.add_argument('--adam_iters', type=int, default=150)
    parser.add_argument('--format', default='json',
                        choices=['json', 'csv', 'pytorch', 'params'])
    parser.add_argument('--J', type=float, default=J_DEFAULT)
    parser.add_argument('--h', type=float, default=H_DEFAULT)

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

    print(f"\n{'═'*55}")
    print("  All selected pipelines complete.")
    print("═"*55)


if __name__ == '__main__':
    main()