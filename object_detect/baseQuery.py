import numpy as np

# --------------------------
# np.random.rand()
# Uniform distribution between 0 and 1
# --------------------------
uniform_samples = np.random.rand(5)  # 5 numbers between 0 and 1
print("Uniform random samples (rand):", uniform_samples)

# Scale to range [a, b], e.g., [0, 10]
scaled_uniform = 10 * np.random.rand(5)
print("Scaled uniform samples [0,10]:", scaled_uniform)

# Scale to range [a, b], e.g., [5, 15]
scaled_uniform = 5 + 15 * np.random.rand(5)
print("Scaled uniform samples [5,15]:", scaled_uniform)

# --------------------------
# np.random.normal()
# Normal (Gaussian) distribution
# --------------------------
normal_samples = np.random.normal(loc=0, scale=1, size=5)  # mean=0, std=1
print("Normal random samples (mean=0, std=1):", normal_samples)

# Normal with custom mean=5, std=2
custom_normal = np.random.normal(loc=5, scale=2, size=5)
print("Normal random samples (mean=5, std=2):", custom_normal)

# Nomal with custom mean = 3, std = 0.5 with size given from outside variable
N = 10
custom_normal_var_size = np.random.normal(loc=3, scale=0.5, size=N)
print("Normal random samples (mean=3, std=0.5, size=N):", custom_normal_var_size)










