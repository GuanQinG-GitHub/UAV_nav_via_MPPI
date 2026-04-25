# benchmark_vectorized_mem.py
import time
import os
import resource
import numpy as np


def get_rss_mb():
    # Read current RSS from /proc/self/status
    rss_kb = None
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
                break
    if rss_kb is None:
        return -1.0
    return rss_kb / 1024.0


def get_peak_rss_mb():
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_one_iteration_vectorized(x0, control_samples, dt):
    num_samples, horizon = control_samples.shape

    # state batch: shape (N, 3)
    x = np.tile(x0.reshape(1, 3), (num_samples, 1)).astype(np.float64)
    cost = np.zeros(num_samples, dtype=np.float64)

    for t in range(horizon):
        u = control_samples[:, t]

        x_pos = x[:, 0]
        x_vel = x[:, 1]
        x_acc = x[:, 2]

        x_next_pos = x_pos + dt * x_vel
        x_next_vel = x_vel + dt * x_acc
        x_next_acc = x_acc + dt * u

        cost += (
            1.0 * x_next_pos ** 2
            + 0.1 * x_next_vel ** 2
            + 0.01 * x_next_acc ** 2
            + 0.001 * u ** 2
        )

        x[:, 0] = x_next_pos
        x[:, 1] = x_next_vel
        x[:, 2] = x_next_acc

    return cost


def main():
    num_samples = 10000
    horizon = 50
    dt = 0.02
    seed = 0
    num_iterations = 10

    rng = np.random.RandomState(seed)
    x0 = np.array([1.0, 0.5, -0.2], dtype=np.float64)

    all_control_samples = [
        rng.normal(0.0, 1.0, (num_samples, horizon)).astype(np.float64)
        for _ in range(num_iterations + 1)
    ]

    print("=== Vectorized warm benchmark with memory ===")
    print("pid            =", os.getpid())
    print("num_samples    =", num_samples)
    print("horizon        =", horizon)
    print("dt             =", dt)
    print("iterations     =", num_iterations)
    print()

    # Theoretical array sizes
    one_control_mb = all_control_samples[0].nbytes / (1024.0 ** 2)
    state_mb = (num_samples * 3 * 8) / (1024.0 ** 2)
    cost_mb = (num_samples * 8) / (1024.0 ** 2)

    print("=== Theoretical array sizes ===")
    print("one control_samples array : {:.3f} MB".format(one_control_mb))
    print("state batch x             : {:.3f} MB".format(state_mb))
    print("cost array                : {:.3f} MB".format(cost_mb))
    print()

    print("RSS before warmup   : {:.3f} MB".format(get_rss_mb()))
    print("Peak RSS before     : {:.3f} MB".format(get_peak_rss_mb()))

    _ = run_one_iteration_vectorized(x0, all_control_samples[0], dt)

    print("RSS after warmup    : {:.3f} MB".format(get_rss_mb()))
    print("Peak RSS after      : {:.3f} MB".format(get_peak_rss_mb()))
    print()

    times = []
    final_costs = None

    for k in range(num_iterations):
        rss_before = get_rss_mb()
        peak_before = get_peak_rss_mb()

        t0 = time.perf_counter()
        final_costs = run_one_iteration_vectorized(x0, all_control_samples[k + 1], dt)
        t1 = time.perf_counter()

        rss_after = get_rss_mb()
        peak_after = get_peak_rss_mb()

        elapsed = t1 - t0
        times.append(elapsed)

        print(
            "Iteration {:02d}: {:.6f} s | RSS {:.3f} -> {:.3f} MB | Peak {:.3f} -> {:.3f} MB".format(
                k + 1, elapsed, rss_before, rss_after, peak_before, peak_after
            )
        )

    avg_time = sum(times) / len(times)

    print()
    print("Average iteration time: {:.6f} s".format(avg_time))
    print("Average cost          : {:.6f}".format(final_costs.mean()))
    print("Min cost              : {:.6f}".format(final_costs.min()))
    print("Max cost              : {:.6f}".format(final_costs.max()))
    print("Average rollouts/sec  : {:.2f}".format(num_samples / avg_time))
    print("Final RSS             : {:.3f} MB".format(get_rss_mb()))
    print("Final Peak RSS        : {:.3f} MB".format(get_peak_rss_mb()))


if __name__ == "__main__":
    main()
