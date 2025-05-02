import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

def build_attribution_graph(
    clt_weights_path: str,
    layers: List[int],
    save_graph_path: str,
    save_image_path: str
):
    data = torch.load(clt_weights_path)
    W_dec = data["W_dec"]    # [L, L, F, F]
    W_out = data["W_out"]    # [L, F, V]
    L, _, F, _ = W_dec.shape
    V = W_out.shape[2]

    G = nx.DiGraph()
    for src in range(L):
        for tgt in range(L):
            W = W_dec[src, tgt]  # [F, F]
            for i in range(F):
                for j in range(F):
                    w = abs(W[j, i].item())
                    if w > 0:
                        G.add_edge(f"({src},{i})", f"({tgt},{j})", weight=w)

    for l in range(L):
        for i in range(F):
            for v in range(V):
                w = abs(W_out[l, i, v].item())
                if w > 0:
                    G.add_edge(f"({l},{i})", f"logit_{v}", weight=w)

    nx.write_gpickle(G, save_graph_path)
    print(f"[Graph] Saved graph → {save_graph_path}")
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [(w / max_w) * 5 for w in weights]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=50)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7)
    plt.axis("off")
    plt.savefig(save_image_path, dpi=300)
    plt.close()
    print(f"[Graph] Saved visualization → {save_image_path}")
