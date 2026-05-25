"""
Seed-Level Statistical Analysis: ISR beta=0.1 vs Random beta=0.1 (5-seed)
Cluster-aware statistics addressing pooled Fisher test limitations.
"""
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportion_confint

# =============================================================================
# DATA (C013 taxonomy: genuine = proved - H7 - H1 - H8)
# =============================================================================

isr_data = {
    # seed: (proved, H7, H1, H8, genuine)
    42: (81, 13, 42, 12, 14),
    43: (61, 59,  0,  0,  2),
    44: (89, 89,  0,  0,  0),
    45: (42, 16,  2,  1, 23),
    46: (77, 11, 20,  4, 42),
}

random_data = {
    # seed: (proved, H7, H1, H8, genuine)
    42: (79,  0, 58,  4, 18),
    43: (94, 36, 56,  1,  1),
    44: (89,  0, 73, 16,  6),
    45: (58,  0, 53,  0,  5),
    46: (94, 86,  2,  4,  2),
}

seeds = [42, 43, 44, 45, 46]

print("=" * 70)
print("SEED-LEVEL STATISTICAL ANALYSIS: ISR beta=0.1 vs Random beta=0.1")
print("=" * 70)

# =============================================================================
# 2a. CMH Test (Cochran-Mantel-Haenszel)
# =============================================================================
print("\n" + "=" * 70)
print("2a. COCHRAN-MANTEL-HAENSZEL (CMH) TEST")
print("=" * 70)

# --- H7 rate comparison ---
tables_h7 = []
for s in seeds:
    ip, ih7, _, _, _ = isr_data[s]
    rp, rh7, _, _, _ = random_data[s]
    table = np.array([[ih7, ip - ih7],
                      [rh7, rp - rh7]])
    tables_h7.append(table)

st_h7 = StratifiedTable(tables_h7)
cmh_h7 = st_h7.test_null_odds()
ci_h7 = st_h7.oddsratio_pooled_confint()
print(f"\n--- H7 Rate: ISR vs Random (stratified by seed) ---")
print(f"  Pooled OR:  {st_h7.oddsratio_pooled:.4f}")
print(f"  95% CI:     [{ci_h7[0]:.4f}, {ci_h7[1]:.4f}]")
print(f"  CMH chi2:   {cmh_h7.statistic:.4f}")
print(f"  p-value:    {cmh_h7.pvalue:.6e}")

# --- Genuine rate comparison ---
tables_gen = []
for s in seeds:
    ip, _, _, _, ig = isr_data[s]
    rp, _, _, _, rg = random_data[s]
    table = np.array([[ig, ip - ig],
                      [rg, rp - rg]])
    tables_gen.append(table)

st_gen = StratifiedTable(tables_gen)
cmh_gen = st_gen.test_null_odds()
ci_gen = st_gen.oddsratio_pooled_confint()
print(f"\n--- Genuine Rate: ISR vs Random (stratified by seed) ---")
print(f"  Pooled OR:  {st_gen.oddsratio_pooled:.4f}")
print(f"  95% CI:     [{ci_gen[0]:.4f}, {ci_gen[1]:.4f}]")
print(f"  CMH chi2:   {cmh_gen.statistic:.4f}")
print(f"  p-value:    {cmh_gen.pvalue:.6e}")

# =============================================================================
# 2b. Permutation Test
# =============================================================================
print("\n" + "=" * 70)
print("2b. PERMUTATION TEST (seed-level rates as units)")
print("=" * 70)

np.random.seed(2024)
n_perm = 10000

# H7 rates per seed
isr_h7_rates = np.array([isr_data[s][1] / isr_data[s][0] for s in seeds])
rand_h7_rates = np.array([random_data[s][1] / random_data[s][0] for s in seeds])

obs_diff_h7 = np.mean(isr_h7_rates) - np.mean(rand_h7_rates)
combined_h7 = np.concatenate([isr_h7_rates, rand_h7_rates])

count_h7 = 0
for _ in range(n_perm):
    perm = np.random.permutation(combined_h7)
    perm_diff = np.mean(perm[:5]) - np.mean(perm[5:])
    if abs(perm_diff) >= abs(obs_diff_h7):
        count_h7 += 1

perm_p_h7 = count_h7 / n_perm

print(f"\n--- H7 Rate Permutation Test ---")
print(f"  ISR H7 rates per seed:    {[f'{r:.4f}' for r in isr_h7_rates]}")
print(f"  Random H7 rates per seed: {[f'{r:.4f}' for r in rand_h7_rates]}")
print(f"  ISR mean H7 rate:         {np.mean(isr_h7_rates):.4f}")
print(f"  Random mean H7 rate:      {np.mean(rand_h7_rates):.4f}")
print(f"  Observed difference:      {obs_diff_h7:+.4f}")
print(f"  Permutation p-value:      {perm_p_h7:.4f} (two-sided, {n_perm} perms)")

# Genuine rates per seed
isr_gen_rates = np.array([isr_data[s][4] / isr_data[s][0] for s in seeds])
rand_gen_rates = np.array([random_data[s][4] / random_data[s][0] for s in seeds])

obs_diff_gen = np.mean(isr_gen_rates) - np.mean(rand_gen_rates)
combined_gen = np.concatenate([isr_gen_rates, rand_gen_rates])

count_gen = 0
for _ in range(n_perm):
    perm = np.random.permutation(combined_gen)
    perm_diff = np.mean(perm[:5]) - np.mean(perm[5:])
    if abs(perm_diff) >= abs(obs_diff_gen):
        count_gen += 1

perm_p_gen = count_gen / n_perm

print(f"\n--- Genuine Rate Permutation Test ---")
print(f"  ISR genuine rates per seed:    {[f'{r:.4f}' for r in isr_gen_rates]}")
print(f"  Random genuine rates per seed: {[f'{r:.4f}' for r in rand_gen_rates]}")
print(f"  ISR mean genuine rate:         {np.mean(isr_gen_rates):.4f}")
print(f"  Random mean genuine rate:      {np.mean(rand_gen_rates):.4f}")
print(f"  Observed difference:           {obs_diff_gen:+.4f}")
print(f"  Permutation p-value:           {perm_p_gen:.4f} (two-sided, {n_perm} perms)")

# H7+H8 combined hacking
isr_hack_rates = np.array([(isr_data[s][1] + isr_data[s][3]) / isr_data[s][0] for s in seeds])
rand_hack_rates = np.array([(random_data[s][1] + random_data[s][3]) / random_data[s][0] for s in seeds])

obs_diff_hack = np.mean(isr_hack_rates) - np.mean(rand_hack_rates)
combined_hack = np.concatenate([isr_hack_rates, rand_hack_rates])

count_hack = 0
for _ in range(n_perm):
    perm = np.random.permutation(combined_hack)
    perm_diff = np.mean(perm[:5]) - np.mean(perm[5:])
    if abs(perm_diff) >= abs(obs_diff_hack):
        count_hack += 1

perm_p_hack = count_hack / n_perm

print(f"\n--- H7+H8 Combined Hacking Rate Permutation Test ---")
print(f"  ISR (H7+H8)/proved per seed:    {[f'{r:.4f}' for r in isr_hack_rates]}")
print(f"  Random (H7+H8)/proved per seed: {[f'{r:.4f}' for r in rand_hack_rates]}")
print(f"  ISR mean (H7+H8) rate:          {np.mean(isr_hack_rates):.4f}")
print(f"  Random mean (H7+H8) rate:       {np.mean(rand_hack_rates):.4f}")
print(f"  Observed difference:            {obs_diff_hack:+.4f}")
print(f"  Permutation p-value:            {perm_p_hack:.4f} (two-sided, {n_perm} perms)")

# =============================================================================
# 2c. Wilson CI for Per-Seed Percentages
# =============================================================================
print("\n" + "=" * 70)
print("2c. WILSON 95% CI FOR PER-SEED METRICS")
print("=" * 70)

print(f"\n{'Cond':<8} {'Seed':<5} {'n':<4} {'H7%':<22} {'Genuine%':<22}")
print("-" * 65)

for cond_name, data in [("ISR", isr_data), ("Random", random_data)]:
    for s in seeds:
        proved, h7, h1, h8, genuine = data[s]
        
        h7_lo, h7_hi = proportion_confint(h7, proved, alpha=0.05, method='wilson')
        gen_lo, gen_hi = proportion_confint(genuine, proved, alpha=0.05, method='wilson')
        
        h7_pct = h7 / proved * 100
        gen_pct = genuine / proved * 100
        
        print(f"{cond_name:<8} {s:<5} {proved:<4} "
              f"{h7_pct:5.1f}% [{h7_lo*100:5.1f},{h7_hi*100:5.1f}]  "
              f"{gen_pct:5.1f}% [{gen_lo*100:5.1f},{gen_hi*100:5.1f}]")
    print()

# =============================================================================
# 2d. 5-Seed Pooled Summary
# =============================================================================
print("=" * 70)
print("2d. 5-SEED POOLED SUMMARY + FISHER EXACT TEST")
print("=" * 70)

isr_proved_total = sum(isr_data[s][0] for s in seeds)
isr_h7_total = sum(isr_data[s][1] for s in seeds)
isr_gen_total = sum(isr_data[s][4] for s in seeds)
isr_h8_total = sum(isr_data[s][3] for s in seeds)
isr_h1_total = sum(isr_data[s][2] for s in seeds)

rand_proved_total = sum(random_data[s][0] for s in seeds)
rand_h7_total = sum(random_data[s][1] for s in seeds)
rand_gen_total = sum(random_data[s][4] for s in seeds)
rand_h8_total = sum(random_data[s][3] for s in seeds)
rand_h1_total = sum(random_data[s][2] for s in seeds)

print(f"\n  {'Metric':<20} {'ISR beta=0.1':<20} {'Random beta=0.1':<20}")
print(f"  {'-'*58}")
print(f"  {'Proved (total)':<20} {isr_proved_total:<20} {rand_proved_total:<20}")
print(f"  {'H7 (total)':<20} {isr_h7_total:<20} {rand_h7_total:<20}")
print(f"  {'H1 (total)':<20} {isr_h1_total:<20} {rand_h1_total:<20}")
print(f"  {'H8 (total)':<20} {isr_h8_total:<20} {rand_h8_total:<20}")
print(f"  {'Genuine (total)':<20} {isr_gen_total:<20} {rand_gen_total:<20}")
print(f"  {'H7 rate':<20} {isr_h7_total/isr_proved_total:.4f} ({isr_h7_total}/{isr_proved_total}){'':<3} {rand_h7_total/rand_proved_total:.4f} ({rand_h7_total}/{rand_proved_total})")
print(f"  {'Genuine rate':<20} {isr_gen_total/isr_proved_total:.4f} ({isr_gen_total}/{isr_proved_total}){'':<3} {rand_gen_total/rand_proved_total:.4f} ({rand_gen_total}/{rand_proved_total})")

# Pooled Fisher: H7
table_h7_pooled = np.array([[isr_h7_total, isr_proved_total - isr_h7_total],
                            [rand_h7_total, rand_proved_total - rand_h7_total]])
or_h7, p_h7 = stats.fisher_exact(table_h7_pooled)
print(f"\n  --- Pooled Fisher Exact: H7 ---")
print(f"  OR = {or_h7:.4f}, p = {p_h7:.6e}")

# Pooled Fisher: Genuine
table_gen_pooled = np.array([[isr_gen_total, isr_proved_total - isr_gen_total],
                             [rand_gen_total, rand_proved_total - rand_gen_total]])
or_gen, p_gen = stats.fisher_exact(table_gen_pooled)
print(f"\n  --- Pooled Fisher Exact: Genuine ---")
print(f"  OR = {or_gen:.4f}, p = {p_gen:.6e}")

# Wilson CI for pooled rates
isr_gen_lo, isr_gen_hi = proportion_confint(isr_gen_total, isr_proved_total, alpha=0.05, method='wilson')
rand_gen_lo, rand_gen_hi = proportion_confint(rand_gen_total, rand_proved_total, alpha=0.05, method='wilson')
isr_h7_lo, isr_h7_hi = proportion_confint(isr_h7_total, isr_proved_total, alpha=0.05, method='wilson')
rand_h7_lo, rand_h7_hi = proportion_confint(rand_h7_total, rand_proved_total, alpha=0.05, method='wilson')
print(f"\n  --- Pooled Wilson 95% CI ---")
print(f"  ISR H7 rate:       {isr_h7_total/isr_proved_total:.4f} [{isr_h7_lo:.4f}, {isr_h7_hi:.4f}]")
print(f"  Random H7 rate:    {rand_h7_total/rand_proved_total:.4f} [{rand_h7_lo:.4f}, {rand_h7_hi:.4f}]")
print(f"  ISR genuine rate:  {isr_gen_total/isr_proved_total:.4f} [{isr_gen_lo:.4f}, {isr_gen_hi:.4f}]")
print(f"  Random genuine:    {rand_gen_total/rand_proved_total:.4f} [{rand_gen_lo:.4f}, {rand_gen_hi:.4f}]")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(f"\n  {'Test':<42} {'Stat':<15} {'p-value':<12} {'Sig?'}")
print(f"  {'-'*75}")
print(f"  {'CMH: H7 (seed-stratified)':<42} {'OR='+f'{st_h7.oddsratio_pooled:.3f}':<15} {cmh_h7.pvalue:<12.4e} {'Yes' if cmh_h7.pvalue < 0.05 else 'No'}")
print(f"  {'CMH: Genuine (seed-stratified)':<42} {'OR='+f'{st_gen.oddsratio_pooled:.3f}':<15} {cmh_gen.pvalue:<12.4e} {'Yes' if cmh_gen.pvalue < 0.05 else 'No'}")
print(f"  {'Permutation: H7 rate (n=5 per group)':<42} {'d='+f'{obs_diff_h7:+.4f}':<15} {perm_p_h7:<12.4f} {'Yes' if perm_p_h7 < 0.05 else 'No'}")
print(f"  {'Permutation: Genuine rate':<42} {'d='+f'{obs_diff_gen:+.4f}':<15} {perm_p_gen:<12.4f} {'Yes' if perm_p_gen < 0.05 else 'No'}")
print(f"  {'Permutation: H7+H8 hacking rate':<42} {'d='+f'{obs_diff_hack:+.4f}':<15} {perm_p_hack:<12.4f} {'Yes' if perm_p_hack < 0.05 else 'No'}")
print(f"  {'Pooled Fisher: H7 (ignores clustering)':<42} {'OR='+f'{or_h7:.3f}':<15} {p_h7:<12.4e} {'Yes' if p_h7 < 0.05 else 'No'}")
print(f"  {'Pooled Fisher: Genuine (ignores clustering)':<42} {'OR='+f'{or_gen:.3f}':<15} {p_gen:<12.4e} {'Yes' if p_gen < 0.05 else 'No'}")

print("\n\nNote: With only 5 seeds per group, permutation tests have limited resolution")
print("(minimum achievable two-sided p = 2/C(10,5) = 2/252 = 0.0079).")
print("CMH properly accounts for seed-level clustering in sample-level comparisons.")
print("Pooled Fisher included for reference but ignores within-seed correlation.")
