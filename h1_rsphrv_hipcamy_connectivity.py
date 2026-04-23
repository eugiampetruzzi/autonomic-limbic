"""
Hypothesis 1: RespHRV and hippocampal-amygdala connectivity
- Regress each of 32 HIP-AMY connections on each TSST phase (baseline, reactivity, recovery)
- 96 tests total (3 phases × 32 connections)
- Benjamini-Hochberg FDR correction within each phase
- Covariates: age, mean framewise displacement (mFD)
- All continuous predictors z-scored
"""
import pandas as pd
import numpy as np
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
OD      = '/Users/eu/Library/CloudStorage/OneDrive-Stanford/ELS Actigraphy/'
ELS_D   = '/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ELS/'

cardiac = pd.read_csv(OD + 'analytic_dataset.csv')
amy     = pd.read_csv(ELS_D + '0 - curated/amygda-hippo/amygdala_connectivity.csv')

# ── Column definitions ─────────────────────────────────────────────────────────
HIP_AMY  = [c for c in amy.columns if 'HIP' in c and 'AMY' in c]   # 32 connections
RSA_COLS = ['baseline_RSA', 'reactivity_RSA', 'recovery_RSA']

# ── Merge cardiac and connectivity at matched timepoints ───────────────────────
amy_t1 = amy[amy['timepoint'] == 'T1'][['subject_id', 'mean_fd'] + HIP_AMY]\
             .rename(columns={'subject_id': 'ID'})
amy_t2 = amy[amy['timepoint'] == 'T2'][['subject_id', 'mean_fd'] + HIP_AMY]\
             .rename(columns={'subject_id': 'ID'})
m1 = cardiac[cardiac['tsst_timepoint'] == 1].merge(amy_t1, on='ID', how='inner')
m2 = cardiac[cardiac['tsst_timepoint'] == 2].merge(amy_t2, on='ID', how='inner')
df = pd.concat([m1, m2], ignore_index=True)

all_cols = RSA_COLS + HIP_AMY + ['age_at_tsst', 'mean_fd']
d = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()
N = len(d)
print(f'Analytic N = {N}')

# ── Z-score all continuous predictors ─────────────────────────────────────────
def zscore(x):
    return (x - x.mean()) / x.std()

for col in RSA_COLS + HIP_AMY + ['age_at_tsst']:
    d = d.copy()
    d[col] = zscore(d[col])

# mean_fd is a covariate but not a predictor of interest; z-score for consistency
d['mean_fd'] = zscore(d['mean_fd'])

# ── OLS helper ────────────────────────────────────────────────────────────────
def ols_ttest(X, y):
    """Return (b, se, t, p) for all coefficients via OLS."""
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    mse = np.sum((y - X @ b) ** 2) / dof
    cov_m = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_m))
    t  = b / se
    p  = np.array([float(2 * stats.t.sf(abs(ti), dof)) for ti in t])
    return b, se, t, p, dof

# ── BH-FDR correction ─────────────────────────────────────────────────────────
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n = len(pvals)
    idx = np.argsort(pvals)
    q = np.empty(n)
    q[idx] = np.array(pvals)[idx] * n / (np.arange(1, n + 1))
    # Enforce monotonicity from right
    for i in range(n - 2, -1, -1):
        q[idx[i]] = min(q[idx[i]], q[idx[i + 1]])
    return np.clip(q, 0, 1)

# ── Main loop: 3 phases × 32 connections ──────────────────────────────────────
results = []
age_v = d['age_at_tsst'].values
mfd_v = d['mean_fd'].values

for phase in RSA_COLS:
    rsa_v = d[phase].values
    p_vals = []
    phase_rows = []

    for conn in HIP_AMY:
        y  = d[conn].values
        X  = np.column_stack([np.ones(N), rsa_v, age_v, mfd_v])
        b, se, t, p, dof = ols_ttest(X, y)
        # index 1 = RSA predictor
        p_vals.append(p[1])
        phase_rows.append({
            'phase': phase,
            'connection': conn,
            'b': b[1], 'se': se[1], 't': t[1], 'p': p[1], 'dof': dof
        })

    # FDR within phase
    q_vals = bh_fdr(p_vals)
    for row, q in zip(phase_rows, q_vals):
        row['q'] = q
    results.extend(phase_rows)

res = pd.DataFrame(results)

# ── Print results ──────────────────────────────────────────────────────────────
for phase in RSA_COLS:
    phase_res = res[res['phase'] == phase].sort_values('p')
    sig = phase_res[phase_res['q'] < .05]
    print(f'\n{"="*70}')
    print(f'Phase: {phase}  ({len(sig)}/{len(HIP_AMY)} connections survive FDR q<.05)')
    print(f'{"="*70}')
    print(f'{"Connection":<42} {"β":>7} {"t":>7} {"p":>8} {"q":>8}')
    print('-' * 70)
    for _, row in phase_res.iterrows():
        sig_marker = ' **' if row['q'] < .01 else (' *' if row['q'] < .05 else
                     (' +' if row['p'] < .10 else ''))
        print(f'{row["connection"]:<42} {row["b"]:>+7.3f} {row["t"]:>+7.2f} '
              f'{row["p"]:>8.4f} {row["q"]:>8.4f}{sig_marker}')

# ── Save full results table ────────────────────────────────────────────────────
OUT = '/Users/eu/Library/CloudStorage/OneDrive-Stanford/ELS Actigraphy/Analysis_Amygdala/paper_analyses/'
res.to_csv(OUT + 'h1_rsphrv_hipcamy_results.csv', index=False)
print(f'\nResults saved to {OUT}h1_rsphrv_hipcamy_results.csv')
