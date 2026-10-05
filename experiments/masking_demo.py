"""Five deterministic sensory examples; no signal meanings or learning."""

from dataclasses import asdict
import json

from systems.communication import Communication


def run_demo() -> dict:
    results = {}
    for name, distances in (("A", (1, 3)), ("B", (1, 2)), ("C", (2, 2)),
                            ("D", (2,)), ("E", ())):
        channel = Communication(16, 8)
        for sender, distance in enumerate(distances, start=1):
            channel.emit(sender + 6, sender, distance, 0)
        channel.advance()
        results[name] = {"distances_for_diagnostics_only": distances,
                         **channel.resolution_diagnostics(0, 0, 0)}
    return results


if __name__ == "__main__":
    print(json.dumps(run_demo(), default=asdict, indent=2))
