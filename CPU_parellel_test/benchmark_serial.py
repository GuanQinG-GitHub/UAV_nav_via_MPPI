# benchmark_serial_mem.py
import time
import os
import resource
import numpy as np


def get_rss_mb():
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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rollout_triple_integrator(x0, u_seq, dt):
    x = x0.copy()
    cost = 0.0

    for u in u_seq:
        x_pos, x_vel, x_acc = x
        x_next = np.array([
            x_pos + dt * x_vel,
            x_vel + dt * x_acc,
            x_acc + dt * u,
        ], dtype=np.float64)

        cost += (
            1.0 * x_next[0] ** 2
            + 0.1 * x_next[1] ** 2
            + 0.01 * x_next[2] ** 2
            + 0.001 * u ** 2
        )
        x = x_next

    return cost


def run_one_iteration(x0, control_samples, dt):
    num_samples = control_samples.shape[0]
    costs = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        costs[i] = rollout_triple_integrator(x0, control_samples[i], dt)

    return costs


def main():
    num_samples = 5000
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

    print("=== Serial warm benchmark with memory ===")
    print("pid            =", os.getpid())
    print("num_samples    =", num_samples)
    print("horizon        =", horizon)
    print("dt             =", dt)
    print("iterations     =", num_iterations)
    print()

    one_control_mb = all_control_samples[0].nbytes / (1024.0 ** 2)
    x0_mb = x0.nbytes / (1024.0 ** 2)
    cost_mb = (num_samples * 8) / (1024.0 ** 2)

    print("=== Theoretical array sizes ===")
    print("one control_samples array : {:.3f} MB".format(one_control_mb))
    print("single state x0           : {:.6f} MB".format(x0_mb))
    print("cost array                : {:.3f} MB".format(cost_mb))
    print()

    print("RSS before warmup   : {:.3f} MB".format(get_rss_mb()))
    print("Peak RSS before     : {:.3f} MB".format(get_peak_rss_mb()))

    _ = run_one_iteration(x0, all_control_samples[0], dt)

    print("RSS after warmup    : {:.3f} MB".format(get_rss_mb()))
    print("Peak RSS after      : {:.3f} MB".format(get_peak_rss_mb()))
    print()

    times = []
    final_costs = None

    for k in range(num_iterations):
        rss_before = get_rss_mb()
        peak_before = get_peak_rss_mb()

        t0 = time.perf_counter()
        final_costs = run_one_iteration(x0, all_control_samples[k + 1], dt)
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
