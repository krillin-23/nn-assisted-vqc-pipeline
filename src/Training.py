import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

N_QUBITS   = 6       
J          = 1.0     
H_FIELD    = 0.5     
MAX_DEPTH  = 12      
N_SAMPLES  = 150     
NN_TRAIN   = 80      
CONV_ITERS = 200     
CONV_DEPTHS = [4, 10] 
SEED       = 42

np.random.seed(SEED)

def make_hamiltonian(n, J=1.0, h=0.5):
    terms = []
    for i in range(n):
        j = (i+1) % n
        p = ['I']*n; p[i]='Z'; p[j]='Z'
        terms.append((''.join(reversed(p)), -J))
    for i in range(n):
        p = ['I']*n; p[i]='X'
        terms.append((''.join(reversed(p)), -h))
    return SparsePauliOp.from_list(terms)

H    = make_hamiltonian(N_QUBITS, J, H_FIELD)
Hmat = H.to_matrix()
E_exact = float(np.linalg.eigvalsh(Hmat)[0])
print(f"Hamiltonian: {N_QUBITS}-qubit TFIM  J={J}  h={H_FIELD}")
print(f"Exact ground state energy: {E_exact:.6f}\n")

def energy(params, n_layers):
    qc = QuantumCircuit(N_QUBITS)
    idx = 0
    for _ in range(n_layers):
        for q in range(N_QUBITS):
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q+1)
    sv = Statevector(qc).data
    return float(np.real(np.conj(sv) @ Hmat @ sv))

def n_params(n_layers):
    return 2 * N_QUBITS * n_layers

def gradient(params, n_layers):
    g = np.zeros_like(params)
    for i in range(len(params)):
        pp = params.copy(); pp[i] += np.pi/2
        pm = params.copy(); pm[i] -= np.pi/2
        g[i] = (energy(pp, n_layers) - energy(pm, n_layers)) / 2
    return g

def adam_step(params, g, m, v, t, lr=0.05):
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g**2
    t += 1
    params = params - lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
    return params, m, v, t

def nn_forward(z, W1, b1, W2, b2):
    h = np.maximum(0, z @ W1 + b1)
    return np.tanh(h @ W2 + b2) * np.pi

def nn_backward(z, W1, b1, W2, b2, gq, lr=2e-3):
    h    = np.maximum(0, z @ W1 + b1)
    pre2 = h @ W2 + b2
    d2   = gq * (1 - np.tanh(pre2)**2) * np.pi
    dW2  = np.outer(h, d2); db2 = d2
    d1   = (d2 @ W2.T) * (h > 0)
    dW1  = np.outer(z, d1); db1 = d1
    W1  -= lr * dW1; b1 -= lr * db1
    W2  -= lr * dW2; b2 -= lr * db2
    return W1, b1, W2, b2

def make_nn(input_dim, output_dim, hidden=32):
    W1 = np.random.randn(input_dim, hidden) * np.sqrt(2/input_dim)
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, output_dim) * np.sqrt(2/hidden)
    b2 = np.zeros(output_dim)
    return W1, b1, W2, b2

print("="*50)
print("EXPERIMENT 1: Gradient Variance vs Depth")
print("="*50)

depths  = list(range(1, MAX_DEPTH+1))
var_rand = []
var_nn   = []

for L in depths:
    npt = n_params(L)

    gs_r = []
    for _ in range(N_SAMPLES):
        t = np.random.uniform(-np.pi, np.pi, npt)
        gs_r.append(gradient(t, L)[0])
    var_rand.append(np.var(gs_r))

    W1, b1, W2, b2 = make_nn(8, npt, hidden=32)
    for _ in range(NN_TRAIN):
        z  = np.random.randn(8)
        th = nn_forward(z, W1, b1, W2, b2)
        gq = gradient(th, L)
        W1, b1, W2, b2 = nn_backward(z, W1, b1, W2, b2, gq)

    gs_n = []
    for _ in range(N_SAMPLES):
        z  = np.random.randn(8)
        th = nn_forward(z, W1, b1, W2, b2)
        gs_n.append(gradient(th, L)[0])
    var_nn.append(np.var(gs_n))

    print(f"  L={L:2d} | Var(rand)={var_rand[-1]:.3e} | Var(NN)={var_nn[-1]:.3e}"
          f" | Ratio={var_rand[-1]/max(var_nn[-1],1e-20):.1f}x")

print(f"\n{'='*50}")
print("EXPERIMENT 2: Energy Convergence Comparison")
print("="*50)

conv = {}
for L in CONV_DEPTHS:
    npt = n_params(L)
    np.random.seed(SEED + L)

    p = np.random.uniform(-np.pi, np.pi, npt)
    m = np.zeros(npt); v = np.zeros(npt); t = 0; hr = []
    for _ in range(CONV_ITERS):
        hr.append(energy(p, L))
        g = gradient(p, L)
        p, m, v, t = adam_step(p, g, m, v, t)
    print(f"  L={L} | Random | Final E={hr[-1]:.5f} | Err={abs(hr[-1]-E_exact):.5f}")

    np.random.seed(SEED + L + 100)
    W1, b1, W2, b2 = make_nn(8, npt, hidden=32)
    nn_h = []
    for _ in range(CONV_ITERS // 2):
        z  = np.random.randn(8)
        th = nn_forward(z, W1, b1, W2, b2)
        nn_h.append(energy(th, L))
        gq = gradient(th, L)
        W1, b1, W2, b2 = nn_backward(z, W1, b1, W2, b2, gq, lr=3e-3)

    best_t = None; best_E = np.inf
    for _ in range(30):
        z  = np.random.randn(8)
        t_ = nn_forward(z, W1, b1, W2, b2)
        e_ = energy(t_, L)
        if e_ < best_E: best_E = e_; best_t = t_.copy()

    model_filename = f'trained_nn_model_depth_{L}.npz'
    np.savez(model_filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"  [+] Model saved successfully as: {model_filename}")

    p = best_t; m = np.zeros(npt); v = np.zeros(npt); tc = 0; hft = []
    for _ in range(CONV_ITERS // 2):
        hft.append(energy(p, L))
        g = gradient(p, L)
        p, m, v, tc = adam_step(p, g, m, v, tc, lr=0.03)

    hn = nn_h + hft
    print(f"  L={L} | NN     | Final E={hn[-1]:.5f} | Err={abs(hn[-1]-E_exact):.5f}")
    conv[L] = {'rand': hr, 'nn': hn, 'exact': E_exact}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Mitigating Barren Plateaus via Neural Network Parameter Generation\n'
    r'Friedrich \& Maziero, Phys. Rev. A $\mathbf{106}$, 042433 (2022)',
    fontsize=13, fontweight='bold')

CR = '#d62728'
CN = '#1f77b4'

ax = axes[0, 0]
ax.plot(depths, var_rand, 'o-', c=CR, lw=2, ms=8, label='Random Init')
ax.plot(depths, var_nn,   's--', c=CN, lw=2, ms=8, label='NN-Generated Init')
ax.set_xlabel('Circuit Depth $L$')
ax.set_ylabel(r'Var$[\partial C/\partial\theta_k]$')
ax.set_title(f'Gradient Variance vs Depth  ($n={N_QUBITS}$ qubits)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.semilogy(depths, var_rand, 'o-',  c=CR, lw=2, ms=8, label='Random Init')
ax.semilogy(depths, var_nn,   's--', c=CN, lw=2, ms=8, label='NN-Generated Init')
ax.set_xlabel('Circuit Depth $L$')
ax.set_ylabel(r'Var$[\partial C/\partial\theta_k]$ (log scale)')
ax.set_title(f'Barren Plateau — Log Scale  ($n={N_QUBITS}$ qubits)')
ax.legend(); ax.grid(alpha=0.3, which='both')

for L, ax in zip(CONV_DEPTHS, [axes[1, 0], axes[1, 1]]):
    r = conv[L]
    ax.plot(r['rand'], c=CR, lw=2, label='Random Init')
    ax.plot(r['nn'],   c=CN, lw=2, ls='--', label='NN Init + Fine-tune')
    ax.axhline(r['exact'], c='k', ls=':', lw=2,
               label=f"Exact $E_0$={r['exact']:.3f}")
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel(r'Energy $\langle H \rangle$')
    ax.set_title(f'Energy Convergence — $L={L}$  ($n={N_QUBITS}$ qubits)')
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('barren_plateau_results.png', dpi=180, bbox_inches='tight')
print("\nFigure saved: barren_plateau_results.png")

print(f"\n{'='*55}")
print(f"RESULTS SUMMARY  ({N_QUBITS} qubits, TFIM J={J} h={H_FIELD})")
print(f"{'='*55}")

print(f"\nGradient Variance:")
print(f"{'L':>3} | {'Var(Random)':>13} | {'Var(NN)':>12} | {'Ratio':>7}")
print('-'*44)
for i, L in enumerate(depths):
    ratio = var_rand[i] / max(var_nn[i], 1e-20)
    print(f"{L:>3} | {var_rand[i]:>13.3e} | {var_nn[i]:>12.3e} | {ratio:>6.1f}x")

print(f"\nFinal Energies  (Exact = {E_exact:.6f}):")
print(f"{'L':>3} | {'Rand Final E':>14} | {'Rand Err':>10} | {'NN Final E':>12} | {'NN Err':>8} | {'Improv':>7}")
print('-'*70)
for L in CONV_DEPTHS:
    r   = conv[L]
    er  = abs(r['rand'][-1] - E_exact)
    en  = abs(r['nn'][-1]   - E_exact)
    imp = er / max(en, 1e-9)
    print(f"{L:>3} | {r['rand'][-1]:>14.5f} | {er:>10.5f} | {r['nn'][-1]:>12.5f} | {en:>8.5f} | {imp:>6.1f}x")