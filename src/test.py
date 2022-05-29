import numpy as np
import pdb

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

np.random.seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

n_steps_z = 50
series_z = generate_time_series(10000, n_steps_z + 10)
X_train_z, Y_train_z = series_z[:7000, :n_steps_z], series_z[:7000, -10:, 0]
X_valid_z, Y_valid_z = series_z[7000:9000, :n_steps_z], series_z[7000:9000, -10:, 0]
X_test_z, Y_test_z = series_z[9000:, :n_steps_z], series_z[9000:, -10:, 0]

pdb.set_trace()