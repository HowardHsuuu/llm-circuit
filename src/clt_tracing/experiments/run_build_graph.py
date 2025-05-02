import argparse
from clt_tracing.graph_builder import build_attribution_graph

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clt",      type=str, required=True)
    p.add_argument("--layers",   type=int, nargs="+", required=True)
    p.add_argument("--out_graph",type=str, required=True)
    p.add_argument("--out_img",  type=str, required=True)
    args = p.parse_args()

    build_attribution_graph(
        args.clt, args.layers,
        args.out_graph, args.out_img
    )
