import argparse
import os
import json

from circuit_tracing_llama.viz.graph_viz import (
    build_attribution_graph,
    plot_attribution_graph
)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize pruned attribution components as a circuit graph"
    )
    parser.add_argument(
        "--pruned_path",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Attribution Circuit",
        help=""
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/vis/attribution_graph.png",
        help=""
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print(f"[1/3] Loading pruned components from {args.pruned_path}")
    with open(args.pruned_path, "r", encoding="utf-8") as f:
        pruned: dict = json.load(f)

    print("[2/3] Building attribution graph")
    G = build_attribution_graph(pruned, center_node="target_logit")
    print(f"[3/3] Plotting and saving to {args.save_path}")
    plot_attribution_graph(G, title=args.title, save_path=args.save_path)
    print("Done.")

if __name__ == "__main__":
    main()
