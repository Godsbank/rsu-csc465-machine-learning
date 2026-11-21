"""Unit 1: probability, Bayes' theorem and a hypothesis test.

(a) A malaria rapid test: what is P(malaria | positive)?
(b) A normal distribution of examination scores.
(c) A two-sample t-test: did a new teaching method change the scores?
"""
import numpy as np
from scipy import stats

# (a) Bayes' theorem
p_m = 0.10                      # prior: 10% of patients tested have malaria
sens, spec = 0.95, 0.90         # P(+ | m) and P(- | no m)
p_pos = sens * p_m + (1 - spec) * (1 - p_m)
print(f"P(positive) = {p_pos:.3f}")
print(f"P(malaria | positive) = {sens * p_m / p_pos:.3f}")

# (b) a normal distribution: mean 55, standard deviation 12
scores = stats.norm(55, 12)
print(f"P(score >= 70) = {1 - scores.cdf(70):.3f}")
print(f"score at the 90th percentile = {scores.ppf(0.90):.1f}")

# (c) a t-test comparing two groups of 200 students
# H0: both teaching methods have the same average score.
# H1: the new teaching method produces a different average score.
group_size = 200
seed = 465

# The seed makes the generated random scores reproducible.
# Both groups use a normal distribution with standard deviation 12.
rng = np.random.default_rng(seed)
old = rng.normal(55, 12, group_size)
new = rng.normal(61, 12, group_size)

# An independent two-sample t-test compares the means of the two groups.
# t is the test statistic; p is the probability of this result assuming H0.
t, p = stats.ttest_ind(new, old)
print(f"fixed seed {seed}:")
print(f"means: old {old.mean():.1f}, new {new.mean():.1f}")
print(f"t = {t:.2f}, p-value = {p:.4f}",
      "-> reject H0 at 5%" if p < 0.05 else "-> do not reject H0")

# The result can vary when the random data changes. Repeat the simulation
# with several seeds and count how often the test rejects H0 at the 5% level.
seed_count = 100
rejections = 0

for seed in range(seed_count):
    rng = np.random.default_rng(seed)
    old = rng.normal(55, 12, group_size)
    new = rng.normal(61, 12, group_size)

    _, p = stats.ttest_ind(new, old)
    rejections += p < 0.05

print(f"seed results: rejected H0 in {rejections} of "
      f"{seed_count} runs ({100 * rejections / seed_count:.1f}%)")
