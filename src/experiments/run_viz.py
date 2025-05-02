import sys, os
import argparse, json
from src.viz.graph_viz import build_attribution_graph, plot_attribution_graph

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pruned_path", type=str, required=True)
    p.add_argument("--title",       type=str, default="Attribution Circuit")
    p.add_argument("--save_path",   type=str, default="outputs/vis/circuit.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    pruned = json.load(open(args.pruned_path, "r", encoding="utf-8"))
    G = build_attribution_graph(pruned, center_node="target_logit")
    plot_attribution_graph(G, title=args.title, save_path=args.save_path)
    print(f"[+] Saved circuit viz → {args.save_path}")

if __name__=="__main__":
    main()
