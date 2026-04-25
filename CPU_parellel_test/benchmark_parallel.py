# benchmark_parallel_warm.py
import os
import time
import numpy as np
import multiprocessing as mp


def rollout_triple_integrator(args):
    x0, u_seq, dt = args

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


def run_one_iteration(pool, x0, control_samples, dt, chunksize=32):
    task_args = [(x0, control_samples[i], dt) for i in range(control_samples.shape[0])]
    costs = pool.map(rollout_triple_integrator, task_args, chunksize=chunksize)
    return np.asarray(costs, dtype=np.float64)


def main():
    num_samples = 5000
    horizon = 50
    dt = 0.02
    seed = 0
    num_iterations = 10
    nproc = min(mp.cpu_count(), os.cpu_count() or 1)

    rng = np.random.RandomState(seed)
    x0 = np.array([1.0, 0.5, -0.2], dtype=np.float64)

    # Pre-generate controls for all iterations to avoid counting random generation time
    all_control_samples = [
        rng.normal(0.0, 1.0, (num_samples, horizon))
        for _ in range(num_iterations + 1)  # +1 for warmup
    ]

    print("=== Parallel warm benchmark ===")
    print("num_samples    =", num_samples)
    print("horizon        =", horizon)
    print("dt             =", dt)
    print("iterations     =", num_iterations)
    print("processes      =", nproc)

    times = []
    final_costs = None

    # Pool creation is outside timed iterations
    with mp.Pool(processes=nproc) as pool:
        # Warmup
        _ = run_one_iteration(pool, x0, all_control_samples[0], dt)

        for k in range(num_iterations):
            t0 = time.perf_counter()
            final_costs = run_one_iteration(pool, x0, all_control_samples[k + 1], dt)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            times.append(elapsed)
            print("Iteration {:02d}: {:.6f} s".format(k + 1, elapsed))

    avg_time = sum(times) / len(times)

    print()
    print("Average iteration time: {:.6f} s".format(avg_time))
    print("Average cost          : {:.6f}".format(final_costs.mean()))
    print("Min cost              : {:.6f}".format(final_costs.min()))
    print("Max cost              : {:.6f}".format(final_costs.max()))
    print("Average rollouts/sec  : {:.2f}".format(num_samples / avg_time))


if __name__ == "__main__":
    main()
