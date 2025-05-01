import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional

def build_attribution_graph(
    importances: Dict[str, float],
    center_node: str = "output_token"
) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node(center_node)
    for comp, score in importances.items():
        G.add_node(comp)
        G.add_edge(comp, center_node, weight=score)
    return G

def plot_attribution_graph(
    G: nx.DiGraph,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
):
    pos = nx.spring_layout(G, seed=42)  # 力导向布局 :contentReference[oaicite:0]{index=0}
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 归一化边宽以便可视化
    max_w = max(weights) if weights else 1.0
    widths = [ (w / max_w) * 5 for w in weights ]  # 最大 5 pt

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->',
        arrowsize=15,
        width=widths,
        edge_color='gray',
        connectionstyle='arc3,rad=0.1'
    )
    if title:
        plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
