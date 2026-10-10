"""Flywheel metrics: track the data flywheel across generations.

The thesis of this repo is a virtuous cycle:

    agent (gen N) -> trajectories -> dataset -> agent (gen N+1) -> ...

If the cycle works, every generation should show: higher verification
pass rate (the agent is better), higher mean trajectory quality (the
agent writes cleaner solutions in fewer steps), and higher dataset
yield per raw trajectory (less waste). The FlywheelTracker records one
GenerationStats per training generation and renders the deltas. When
the deltas go flat or negative, the flywheel is stalling and the
collection strategy, not the model, is the bottleneck.

GenerationStats is also what a real training loop would log after each
data-collection batch, so the tracker doubles as the schema for the
production flywheel.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationStats:
    generation: int
    n_raw: int
    pass_rate: float
    mean_quality: float
    sft_yield: int
    pref_yield: int
    repaired: int = 0
    near_misses: int = 0

    @property
    def yield_per_raw(self) -> float:
        if self.n_raw == 0:
            return 0.0
        return (self.sft_yield + self.pref_yield) / self.n_raw


class FlywheelTracker:
    """Record per-generation stats and report the virtuous cycle."""

    def __init__(self) -> None:
        self.generations: list[GenerationStats] = []

    def record(self, stats: GenerationStats) -> None:
        self.generations.append(stats)
        self.generations.sort(key=lambda g: g.generation)

    def deltas(self) -> list[dict]:
        """Generation-over-generation changes for the headline metrics."""
        out = []
        for prev, cur in zip(self.generations, self.generations[1:]):
            out.append({
                "generation": cur.generation,
                "d_pass_rate": round(cur.pass_rate - prev.pass_rate, 4),
                "d_mean_quality": round(
                    cur.mean_quality - prev.mean_quality, 4),
                "d_yield_per_raw": round(
                    cur.yield_per_raw - prev.yield_per_raw, 4),
                "d_sft_yield": cur.sft_yield - prev.sft_yield,
            })
        return out

    def compounding(self) -> dict:
        """Headline numbers: total improvement from first to last."""
        if len(self.generations) < 2:
            return {"generations": len(self.generations)}
        first, last = self.generations[0], self.generations[-1]
        return {
            "generations": len(self.generations),
            "pass_rate": (round(first.pass_rate, 3),
                          round(last.pass_rate, 3)),
            "mean_quality": (round(first.mean_quality, 3),
                             round(last.mean_quality, 3)),
            "yield_per_raw": (round(first.yield_per_raw, 3),
                              round(last.yield_per_raw, 3)),
            "total_sft_rows": sum(g.sft_yield for g in self.generations),
            "total_repaired": sum(g.repaired for g in self.generations),
        }

    def report(self) -> str:
        lines = []
        lines.append("=" * 64)
        lines.append("FLYWHEEL REPORT")
        lines.append("=" * 64)
        lines.append(f"{'gen':>4} {'raw':>6} {'pass':>7} {'quality':>8} "
                     f"{'sft':>5} {'pref':>5} {'yield/raw':>9}")
        for g in self.generations:
            lines.append(
                f"{g.generation:>4} {g.n_raw:>6} {g.pass_rate:>7.3f} "
                f"{g.mean_quality:>8.3f} {g.sft_yield:>5} "
                f"{g.pref_yield:>5} {g.yield_per_raw:>9.3f}"
            )
        lines.append("")
        for d in self.deltas():
            arrow = lambda v: "+" if v >= 0 else ""
            lines.append(
                f"gen {d['generation']}: pass_rate {arrow(d['d_pass_rate'])}"
                f"{d['d_pass_rate']:.3f}, quality "
                f"{arrow(d['d_mean_quality'])}{d['d_mean_quality']:.3f}, "
                f"yield/raw {arrow(d['d_yield_per_raw'])}"
                f"{d['d_yield_per_raw']:.3f}"
            )
        lines.append("")
        c = self.compounding()
        if c.get("generations", 0) >= 2:
            lines.append(
                f"compounding over {c['generations']} generations: "
                f"pass_rate {c['pass_rate'][0]:.3f} -> "
                f"{c['pass_rate'][1]:.3f}, mean quality "
                f"{c['mean_quality'][0]:.3f} -> {c['mean_quality'][1]:.3f}, "
                f"yield/raw {c['yield_per_raw'][0]:.3f} -> "
                f"{c['yield_per_raw'][1]:.3f}"
            )
            lines.append(f"total SFT rows banked: {c['total_sft_rows']}, "
                         f"trajectories repaired: {c['total_repaired']}")
        lines.append("=" * 64)
        return "\n".join(lines)
