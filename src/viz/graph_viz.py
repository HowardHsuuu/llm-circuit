import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional
import re

def build_attribution_graph(
    importances: Dict[str, float],
    center_node: str = "output_token"
) -> nx.DiGraph:
    """Build hierarchical attribution graph showing layer structure"""
    G = nx.DiGraph()
    G.add_node(center_node)
    
    layers = {}
    for comp, score in importances.items():
        match = re.match(r'(\w+)_L(\d+)', comp)
        if match:
            comp_type, layer_num = match.groups()
            layer_num = int(layer_num)
            if layer_num not in layers:
                layers[layer_num] = []
            layers[layer_num].append((comp, score))
        else:
            G.add_node(comp)
            G.add_edge(comp, center_node, weight=score)
    
    sorted_layers = sorted(layers.keys())
    
    for i, layer_num in enumerate(sorted_layers):
        layer_components = layers[layer_num]
        
        for comp, score in layer_components:
            G.add_node(comp)
            G.add_edge(comp, center_node, weight=score)
            
            if i == 0:
                G.add_node("input")
                G.add_edge("input", comp, weight=0.5)
        
        if i > 0:
            prev_layer = sorted_layers[i-1]
            prev_components = layers[prev_layer]
            
            for prev_comp, prev_score in prev_components:
                for curr_comp, curr_score in layer_components:
                    connection_weight = min(prev_score, curr_score) * 0.3
                    if connection_weight > 0.01:
                        G.add_edge(prev_comp, curr_comp, weight=connection_weight)
    
    return G

def plot_attribution_graph(
    G: nx.DiGraph,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """Plot improved attribution graph with hierarchical layout"""
    pos = nx.spring_layout(G, seed=42, k=3)
    
    nodes_by_layer = {}
    for node in G.nodes():
        if node == "output_token" or node == "target_logit":
            pos[node] = (0, 0)
        elif node == "input":
            pos[node] = (-4, 0)
        else:
            match = re.match(r'\w+_L(\d+)', node)
            if match:
                layer_num = int(match.group(1))
                if layer_num not in nodes_by_layer:
                    nodes_by_layer[layer_num] = []
                nodes_by_layer[layer_num].append(node)
    
    for layer_num, nodes in nodes_by_layer.items():
        x_offset = -2 + layer_num * 1.5
        for i, node in enumerate(nodes):
            y_offset = (i - len(nodes)/2) * 0.8
            pos[node] = (x_offset, y_offset)
    
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    if max_w <= 0:
        max_w = 1.0
    
    plt.figure(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=9)
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        width = max(0.5, (weight / max_w) * 8)
        
        if v == "output_token" or v == "target_logit":
            edge_color = 'red'
        elif u == "input":
            edge_color = 'blue'
        else:
            edge_color = 'green'
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            arrowstyle='->',
            arrowsize=15,
            width=width,
            edge_color=edge_color,
            connectionstyle='arc3,rad=0.1'
        )
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
