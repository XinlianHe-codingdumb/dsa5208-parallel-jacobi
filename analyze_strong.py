"""Parse an Atlas PBS log and draw reproducible strong-scaling figures."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".plot_deps"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
SOURCE = ROOT / "dsa5208_strong.o1036561"
rows = []
repeat = None
content = SOURCE.read_text(encoding="utf-8", errors="replace")
assert "strong_scaling_complete" in content and "Exit Status: 0" in content
for line in content.splitlines():
    if line.startswith("repeat="):
        repeat = int(line.split("=", 1)[1])
    fields = line.split(",")
    if fields[0] not in ("serial", "omp", "mpi", "hybrid"):
        continue
    mode = fields[0]
    if mode in ("serial", "omp"):
        _, nx, ny, steps, threads, seconds, checksum = fields
        ranks = 1
    else:
        _, nx, ny, steps, ranks, threads, seconds, checksum, *_ = fields
    rows.append(
        dict(
            repeat=repeat,
            mode=mode,
            nx=int(nx),
            ny=int(ny),
            steps=int(steps),
            ranks=int(ranks),
            threads=int(threads),
            cores=int(ranks) * int(threads),
            seconds=float(seconds),
            checksum=float(checksum),
        )
    )

assert len(rows) == 42, f"expected 42 runs, got {len(rows)}"
assert set(row["repeat"] for row in rows) == {1, 2, 3}
assert all((row["nx"], row["ny"], row["steps"]) == (2048, 2048, 1000) for row in rows)
checksum_ref = rows[0]["checksum"]
assert all(abs(row["checksum"] - checksum_ref) < 1e-6 for row in rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["mode"], row["ranks"], row["threads"])].append(row["seconds"])
assert all(len(values) == 3 for values in groups.values())
baseline = statistics.median(groups[("serial", 1, 1)])
summary = []
for (mode, ranks, threads), values in sorted(groups.items()):
    median = statistics.median(values)
    cores = ranks * threads
    summary.append(dict(mode=mode, ranks=ranks, threads=threads, cores=cores,
                        median_seconds=median, min_seconds=min(values),
                        max_seconds=max(values), speedup=baseline / median,
                        efficiency=baseline / (median * cores)))

with (ROOT / "strong_scaling_summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
colors = {"omp": "#1764a3", "mpi": "#d1751c", "hybrid": "#39845a"}
for mode in ("omp", "mpi"):
    entries = sorted((r for r in summary if r["mode"] == mode), key=lambda r: r["cores"])
    xs = [r["cores"] for r in entries]
    for axis, key in zip(axes, ("median_seconds", "speedup", "efficiency")):
        axis.plot(xs, [r[key] for r in entries], "o-", color=colors[mode], label=mode.upper())
hybrids = [r for r in summary if r["mode"] == "hybrid"]
for i, r in enumerate(hybrids):
    x = r["cores"] + (i - 1) * 0.35
    for axis, key in zip(axes, ("median_seconds", "speedup", "efficiency")):
        axis.scatter(x, r[key], marker="D", color=colors["hybrid"],
                     label="Hybrid" if i == 0 else None, zorder=4)
axes[0].set_ylabel("Median runtime (s)")
axes[1].set_ylabel("Speedup vs serial")
axes[2].set_ylabel("Efficiency")
for axis in axes:
    axis.set_xlabel("Allocated cores")
    axis.set_xticks([1, 2, 4, 6, 12])
    axis.grid(alpha=0.25)
axes[1].plot([1, 12], [1, 12], "--", color="0.6", linewidth=1, label="Ideal")
axes[2].axhline(1, linestyle="--", color="0.6", linewidth=1)
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.suptitle("Atlas8 strong scaling: 2048 x 2048 grid, 1000 steps, 3 runs")
fig.savefig(ROOT / "strong_scaling.png", dpi=220)
plt.close(fig)

fig2, panels = plt.subplot_mosaic([["runtime", "runtime"], ["speedup", "efficiency"]],
                                  figsize=(9, 5), layout="constrained")
keys = {"runtime": "median_seconds", "speedup": "speedup", "efficiency": "efficiency"}
for name, axis in panels.items():
    key = keys[name]
    for mode in ("omp", "mpi"):
        entries = sorted((r for r in summary if r["mode"] == mode), key=lambda r: r["cores"])
        axis.plot([r["cores"] for r in entries], [r[key] for r in entries],
                  "o-", color=colors[mode], label=mode.upper())
    for i, r in enumerate(hybrids):
        axis.scatter(r["cores"] + (i - 1) * 0.35, r[key], marker="D",
                     color=colors["hybrid"], label="Hybrid" if i == 0 else None, zorder=4)
    axis.set_xticks([1, 2, 4, 6, 12])
    axis.set_xlabel("Allocated cores")
    axis.grid(alpha=0.25)
panels["runtime"].set_ylabel("Median runtime (s)")
panels["speedup"].set_ylabel("Speedup vs serial")
panels["efficiency"].set_ylabel("Efficiency")
panels["speedup"].plot([1, 12], [1, 12], "--", color="0.6", linewidth=1, label="Ideal")
panels["efficiency"].axhline(1, linestyle="--", color="0.6", linewidth=1)
panels["runtime"].legend(ncol=3, fontsize=9)
fig2.savefig(ROOT / "strong_scaling_report.png", dpi=220)
plt.close(fig2)

print(f"serial median: {baseline:.4f} s")
for row in summary:
    print(f"{row['mode']:6} {row['ranks']:2}x{row['threads']:<2} "
          f"{row['median_seconds']:.4f} s  speedup {row['speedup']:.2f}  "
          f"efficiency {row['efficiency']:.2f}")
