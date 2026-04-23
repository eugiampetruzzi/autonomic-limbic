"""
Hypothesis 2: RespHRV recovery × HIP-AMY connectivity × ELS → CDI-S
- Two-way model: recovery × connectivity (no adversity)
- Three-way models: recovery × connectivity × threat (separate)
                    recovery × connectivity × deprivation (separate)
                    recovery × connectivity × unpredictability (separate)
- Joint model: all three adversity dimensions simultaneously
- Simple-slopes decomposition at ±1 SD for significant three-way interactions
- Outcome: CDI-S total score (concurrent, matched to TSST timepoint)
- Covariates: age, mean framewise displacement (mFD)
- All continuous predictors z-scored
"""
import pandas as pd
import numpy as np
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
OD    = '/Users/eu/Library/CloudStorage/OneDrive-Stanford/ELS Actigraphy/'
BOX   = '/Users/eu/Library/CloudStorage/Box-Box/mooddata_nophi/ELS_RDoC/2. Curated Sheets/'
ELS_D = '/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ELS/'

cardiac   = pd.read_csv(OD + 'analytic_dataset.csv')
amy       = pd.read_csv(ELS_D + '0 - curated/amygda-hippo/amygdala_connectivity.csv')
t1        = pd.read_csv(BOX + 'ELS_T1_Child_Curated.csv').rename(columns={'ELS_ID': 'ID'})
t2        = pd.read_csv(BOX + 'ELS_T2_Curated.csv').rename(columns={'ELS_ID': 'ID'})
threat_df = pd.read_csv(ELS_D + '0 - curated/els/els_threat_score.csv').rename(columns={'ELS_ID': 'ID'})
depri_df  = pd.read_csv(ELS_D + '0 - curated/els/els_deprivation.csv').rename(columns={'ELS_ID': 'ID'})
unpred_df = pd.read_csv(ELS_D + '0 - curated/els/els_unpredict_score.csv').rename(columns={'ELS_ID': 'ID'})

# ── Merge at matched timepoints ────────────────────────────────────────────────
brain_col = 'tian_HIP_tail_rh_x_lAMY_lh'
amy_t1 = amy[amy['timepoint'] == 'T1'][['subject_id', 'mean_fd', brain_col]]\
             .rename(columns={'subject_id': 'ID'})
amy_t2 = amy[amy['timepoint'] == 'T2'][['subject_id', 'mean_fd', brain_col]]\
             .rename(columns={'subject_id': 'ID'})
m1 = cardiac[cardiac['tsst_timepoint'] == 1].merge(amy_t1, on='ID', how='inner')
m2 = cardiac[cardiac['tsst_timepoint'] == 2].merge(amy_t2, on='ID', how='inner')
df = pd.concat([m1, m2], ignore_index=True)

# Concurrent CDI-S
cdi1 = df['ID'].map(t1.drop_duplicates('ID').set_index('ID')['CDI_total.T1'])
cdi2 = df['ID'].map(t2.drop_duplicates('ID').set_index('ID')['CDI_total.T2'])
df['CDI_c'] = np.where(df['tsst_timepoint'] == 1, cdi1, cdi2).astype(float)

# Adversity scores
df['threat'] = df['ID'].map(threat_df.drop_duplicates('ID').set_index('ID')['threat_sumsev'])
df['depri']  = df['ID'].map(depri_df.drop_duplicates('ID').set_index('ID')['deprivation_sumsev'])
df['unpred'] = df['ID'].map(unpred_df.drop_duplicates('ID').set_index('ID')['unpred_sumsev'])

# ── Analytic sample ────────────────────────────────────────────────────────────
cols = ['recovery_RSA', brain_col, 'threat', 'depri', 'unpred',
        'CDI_c', 'age_at_tsst', 'mean_fd']
d = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
N = len(d)
print(f'Analytic N = {N}')

# ── Z-score all continuous predictors ─────────────────────────────────────────
def zscore(x):
    return (x - x.mean()) / x.std()

rv = zscore(d['recovery_RSA'].values.astype(float));  rsa_sd    = d['recovery_RSA'].std()
bv = zscore(d[brain_col].values.astype(float));       brain_sd  = 1.0  # already z-scored → SD=1
tv = zscore(d['threat'].values.astype(float));        threat_sd = 1.0
dv = zscore(d['depri'].values.astype(float));         depri_sd  = 1.0
uv = zscore(d['unpred'].values.astype(float));        unpred_sd = 1.0
av = zscore(d['age_at_tsst'].values.astype(float))
fv = zscore(d['mean_fd'].values.astype(float))
Y  = d['CDI_c'].values.astype(float)

# ── OLS helper ────────────────────────────────────────────────────────────────
def ols(C, y):
    b, _, _, _ = np.linalg.lstsq(C, y, rcond=None)
    dof   = len(y) - C.shape[1]
    mse   = np.sum((y - C @ b) ** 2) / dof
    cov_m = mse * np.linalg.inv(C.T @ C)
    se    = np.sqrt(np.diag(cov_m))
    t     = b / se
    p     = np.array([float(2 * stats.t.sf(abs(ti), dof)) for ti in t])
    return b, se, t, p, dof, cov_m

def print_model(labels, b, se, t, p, dof, title=''):
    if title:
        print(f'\n{"="*65}')
        print(title)
        print(f'{"="*65}')
    print(f'{"Term":<26} {"β":>8} {"SE":>7} {"t":>7} {"p":>8}')
    print('-' * 60)
    for lbl, bi, si, ti, pi in zip(labels, b, se, t, p):
        sig = ' **' if pi < .01 else (' *' if pi < .05 else (' +' if pi < .10 else ''))
        print(f'{lbl:<26} {bi:>+8.3f} {si:>7.3f} {ti:>+7.2f} {pi:>8.4f}{sig}')

def simple_slopes(b, cov_m, dof, adv_sd=1.0):
    """
    Simple slopes of recovery at brain ±1 SD × adversity ±1 SD.
    Model index: 0=int, 1=recov, 2=brain, 3=adv, 4=r×b, 5=r×adv, 6=b×adv, 7=r×b×adv, 8=age, 9=mfd
    """
    print(f'\n  Simple slopes of recovery at brain ±1 SD × adversity ±1 SD:')
    print(f'  {"Condition":<30} {"slope":>8} {"SE":>7} {"t":>7} {"p":>8}')
    print('  ' + '-' * 58)
    for b_lbl, bval in [('High connectivity (+1 SD)', 1.0), ('Low connectivity (−1 SD)', -1.0)]:
        for a_lbl, aval in [('High adversity (+1 SD)', adv_sd), ('Low adversity (−1 SD)', -adv_sd)]:
            ss  = float(b[1] + b[4]*bval + b[5]*aval + b[7]*bval*aval)
            g   = np.array([1.0, bval, aval, bval*aval])
            idx = [1, 4, 5, 7]
            var = float(g @ cov_m[np.ix_(idx, idx)] @ g)
            tv  = ss / np.sqrt(var)
            pv  = float(2 * stats.t.sf(abs(tv), dof))
            sig = ' **' if pv < .01 else (' *' if pv < .05 else (' +' if pv < .10 else ''))
            cond = f'{b_lbl}, {a_lbl}'
            print(f'  {cond:<30} {ss:>+8.3f} {np.sqrt(var):>7.3f} {tv:>+7.2f} {pv:>8.4f}{sig}')

# ══════════════════════════════════════════════════════════════════════════════
# Model 0: Two-way (recovery × connectivity, no adversity)
# ══════════════════════════════════════════════════════════════════════════════
C0 = np.column_stack([np.ones(N), rv, bv, rv*bv, av, fv])
lb0 = ['intercept', 'recovery', 'connectivity', 'recovery×connectivity', 'age', 'mFD']
b0, se0, t0, p0, dof0, _ = ols(C0, Y)
print_model(lb0, b0, se0, t0, p0, dof0,
            title='Model 0: Recovery × Connectivity → CDI-S (two-way, no adversity)')

# ══════════════════════════════════════════════════════════════════════════════
# Models 1–3: Separate three-way models
# ══════════════════════════════════════════════════════════════════════════════
adversity_vars = [
    ('Threat',          tv, 'threat'),
    ('Deprivation',     dv, 'depri'),
    ('Unpredictability', uv, 'unpred'),
]

for name, av_vec, _ in adversity_vars:
    C = np.column_stack([np.ones(N), rv, bv, av_vec,
                         rv*bv, rv*av_vec, bv*av_vec, rv*bv*av_vec,
                         av, fv])
    lb = ['intercept', 'recovery', 'connectivity', name,
          'recov×conn', f'recov×{name}', f'conn×{name}',
          f'recov×conn×{name}', 'age', 'mFD']
    b_, se_, t_, p_, dof_, cov_ = ols(C, Y)
    print_model(lb, b_, se_, t_, p_, dof_,
                title=f'Model: Recovery × Connectivity × {name} → CDI-S')
    # Decompose if three-way is significant
    if p_[7] < .10:
        simple_slopes(b_, cov_, dof_)

# ══════════════════════════════════════════════════════════════════════════════
# Model 4: Joint model — all three adversity dimensions simultaneously
# ══════════════════════════════════════════════════════════════════════════════
# Predictors: recovery, brain, threat, depri, unpred,
#             r×b, r×t, r×d, r×u, b×t, b×d, b×u,
#             r×b×t, r×b×d, r×b×u,
#             age, mfd
C_joint = np.column_stack([
    np.ones(N),
    rv, bv, tv, dv, uv,
    rv*bv, rv*tv, rv*dv, rv*uv,
    bv*tv, bv*dv, bv*uv,
    rv*bv*tv, rv*bv*dv, rv*bv*uv,
    av, fv
])
lb_joint = [
    'intercept', 'recovery', 'connectivity', 'threat', 'deprivation', 'unpredictability',
    'recov×conn', 'recov×threat', 'recov×depri', 'recov×unpred',
    'conn×threat', 'conn×depri', 'conn×unpred',
    'recov×conn×threat', 'recov×conn×depri', 'recov×conn×unpred',
    'age', 'mFD'
]
b_j, se_j, t_j, p_j, dof_j, cov_j = ols(C_joint, Y)
print_model(lb_joint, b_j, se_j, t_j, p_j, dof_j,
            title='Joint Model: Recovery × Connectivity × [Threat + Deprivation + Unpredictability] → CDI-S')
