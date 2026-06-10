#!/usr/bin/env python3
"""
France Election Polarization Analysis - Network Analysis Module
Demonstrates retweet network construction, community detection, and echo chamber analysis.
Author: Sidra Jabeen Khan
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

def build_retweet_network(df):
    """
    Build retweet network from social media data.
    Nodes = users, Edges = retweet relationships (weighted by frequency)
    """
    print("=== BUILDING RETWEET NETWORK ===")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add all users as nodes
    all_users = set(df['user_id'].unique())
    if df['original_author'].notna().any():
        all_users.update(df['original_author'].dropna().unique())
    
    for user in all_users:
        G.add_node(user)
    
    # Add edges for retweets
    retweets = df[df['is_retweet'] == True]
    print(f"Found {len(retweets)} retweets out of {len(df)} total posts")
    
    # Count retweet frequencies
    edge_weights = defaultdict(int)
    for _, row in retweets.iterrows():
        if pd.notna(row['original_author']):
            edge = (row['user_id'], row['original_author'])
            edge_weights[edge] += 1
    
    # Add weighted edges
    for (source, target), weight in edge_weights.items():
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G

def analyze_network_structure(G):
    """
    Analyze network structure and identify polarization patterns.
    """
    print("\n=== NETWORK STRUCTURE ANALYSIS ===")
    
    # Basic metrics
    print(f"Density: {nx.density(G):.4f}")
    
    if G.number_of_edges() > 0:
        # Largest weakly connected component
        wcc = max(nx.weakly_connected_components(G), key=len)
        print(f"Largest WCC: {len(wcc)} nodes ({len(wcc)/G.number_of_nodes()*100:.1f}%)")
        
        # Largest strongly connected component
        scc = max(nx.strongly_connected_components(G), key=len)
        print(f"Largest SCC: {len(scc)} nodes ({len(scc)/G.number_of_nodes()*100:.1f}%)")
        
        # Clustering coefficient (for undirected version)
        G_undirected = G.to_undirected()
        print(f"Average clustering: {nx.average_clustering(G_undirected):.4f}")
        
        # Degree distribution
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        print(f"Average in-degree: {np.mean(in_degrees):.2f}")
        print(f"Average out-degree: {np.mean(out_degrees):.2f}")
        print(f"Max in-degree: {max(in_degrees)}")
        print(f"Max out-degree: {max(out_degrees)}")
    
    return {
        'density': nx.density(G),
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'avg_clustering': nx.average_clustering(G.to_undirected()) if G.number_of_edges() > 0 else 0
    }

def detect_communities(G):
    """
    Detect communities in the retweet network using modularity optimization.
    Communities indicate echo chambers or polarized clusters.
    """
    print("\n=== COMMUNITY DETECTION ===")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Use greedy modularity communities (Louvain-like, no external dependencies)
    communities = nx.community.greedy_modularity_communities(G_undirected)
    
    print(f"Detected {len(communities)} communities")
    
    community_sizes = [len(c) for c in communities]
    print(f"Community sizes: {sorted(community_sizes, reverse=True)[:10]}")
    
    # Calculate modularity
    modularity = nx.community.modularity(G_undirected, communities)
    print(f"Modularity: {modularity:.4f}")
    
    # Higher modularity = more polarized network structure
    if modularity > 0.4:
        print("Network shows STRONG community structure (high polarization)")
    elif modularity > 0.3:
        print("Network shows MODERATE community structure (moderate polarization)")
    else:
        print("Network shows WEAK community structure (low polarization)")
    
    return communities, modularity

def identify_influencers(G, top_n=10):
    """
    Identify key influencers using centrality metrics.
    """
    print(f"\n=== TOP {top_n} INFLUENCERS ===")
    
    if G.number_of_edges() == 0:
        print("No edges in network - cannot calculate centrality")
        return {}
    
    # In-degree centrality (most retweeted)
    in_degree_cent = nx.in_degree_centrality(G)
    
    # PageRank (influence considering network structure)
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = nx.pagerank(G)
    
    # Betweenness centrality (bridge between communities)
    G_undirected = G.to_undirected()
    try:
        betweenness = nx.betweenness_centrality(G_undirected, weight='weight')
    except:
        betweenness = nx.betweenness_centrality(G_undirected)
    
    # Combine metrics
    influencer_scores = {}
    for node in G.nodes():
        influencer_scores[node] = {
            'in_degree_centrality': in_degree_cent.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'composite_score': (in_degree_cent.get(node, 0) + 
                              pagerank.get(node, 0) + 
                              betweenness.get(node, 0)) / 3
        }
    
    # Sort by composite score
    top_influencers = sorted(influencer_scores.items(), 
                            key=lambda x: x[1]['composite_score'], 
                            reverse=True)[:top_n]
    
    for i, (user, scores) in enumerate(top_influencers, 1):
        print(f"{i}. {user}: composite={scores['composite_score']:.4f}, "
              f"in-degree={scores['in_degree_centrality']:.4f}, "
              f"pagerank={scores['pagerank']:.4f}")
    
    return influencer_scores

def analyze_echo_chambers(G, communities):
    """
    Analyze echo chamber characteristics within communities.
    """
    print("\n=== ECHO CHAMBER ANALYSIS ===")
    
    G_undirected = G.to_undirected()
    
    echo_chamber_metrics = []
    for i, community in enumerate(communities):
        # Subgraph for this community
        subgraph = G_undirected.subgraph(community)
        
        if len(community) > 1:
            # Internal density
            internal_density = nx.density(subgraph)
            
            # Average clustering
            avg_clustering = nx.average_clustering(subgraph)
            
            # Internal vs external edges
            internal_edges = subgraph.number_of_edges()
            
            external_edges = 0
            for node in community:
                for neighbor in G_undirected.neighbors(node):
                    if neighbor not in community:
                        external_edges += 1
            
            echo_chamber_score = avg_clustering * (internal_edges / (internal_edges + external_edges + 1))
            
            echo_chamber_metrics.append({
                'community_id': i,
                'size': len(community),
                'internal_density': internal_density,
                'avg_clustering': avg_clustering,
                'internal_edges': internal_edges,
                'external_edges': external_edges,
                'echo_chamber_score': echo_chamber_score
            })
            
            print(f"Community {i} ({len(community)} nodes): "
                  f"clustering={avg_clustering:.3f}, "
                  f"echo_score={echo_chamber_score:.3f}")
    
    return echo_chamber_metrics

def visualize_network(G, communities, influencer_scores):
    """
    Visualize the retweet network with communities and influencers highlighted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare layout
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        pos = nx.spring_layout(G.to_undirected(), k=0.5, iterations=50, seed=42)
        
        # Color nodes by community
        node_colors = []
        node_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_community[node] = i
        
        color_map = plt.cm.tab10
        for node in G.nodes():
            comm_id = node_community.get(node, -1)
            if comm_id >= 0:
                node_colors.append(color_map(comm_id % 10))
            else:
                node_colors.append('gray')
        
        # Size nodes by influence
        node_sizes = []
        for node in G.nodes():
            score = influencer_scores.get(node, {}).get('composite_score', 0)
            node_sizes.append(50 + score * 1000)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=axes[0])
        nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=False, ax=axes[0])
        axes[0].set_title('Retweet Network: Communities and Influencers')
        axes[0].axis('off')
        
        # Degree distribution
        in_degrees = [d for n, d in G.in_degree()]
        axes[1].hist(in_degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].set_title('In-Degree Distribution (Retweet Count)')
        axes[1].set_xlabel('In-Degree')
        axes[1].set_ylabel('Frequency')
        axes[1].set_yscale('log')
    else:
        axes[0].text(0.5, 0.5, 'Network too sparse to visualize', 
                    ha='center', va='center', fontsize=14)
        axes[0].axis('off')
        axes[1].text(0.5, 0.5, 'No degree data available', 
                    ha='center', va='center', fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('network_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'network_analysis_results.png'")

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('france_election_2022_processed.csv')
    
    # Build network
    G = build_retweet_network(df)
    
    # Analyze structure
    metrics = analyze_network_structure(G)
    
    # Detect communities
    communities, modularity = detect_communities(G)
    
    # Identify influencers
    influencer_scores = identify_influencers(G, top_n=10)
    
    # Analyze echo chambers
    echo_metrics = analyze_echo_chambers(G, communities)
    
    # Visualize
    visualize_network(G, communities, influencer_scores)
    
    print("\n=== NETWORK SUMMARY ===")
    print(f"Modularity (polarization indicator): {modularity:.4f}")
    print(f"Communities detected: {len(communities)}")
    print(f"Echo chamber analysis complete for {len(echo_metrics)} communities")
