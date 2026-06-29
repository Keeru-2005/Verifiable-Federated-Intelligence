import networkx as nx
import pandas as pd
import numpy as np
import logging
import math
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_simplified_graph(df):
    df = df.sort_values(by=['sender', 'timestamp'])
    df['temporal_delta'] = df.groupby('sender')['timestamp'].diff().fillna(0)
    df['amount_log']     = df['amount'].apply(lambda x: math.log1p(x) if x > 0 else 0)

    pair_stats = df.groupby(['sender', 'receiver']).agg(
        total_amount = ('amount',        'sum'),
        avg_amount   = ('amount',        'mean'),
        max_amount   = ('amount',        'max'),
        min_amount   = ('amount',        'min'),
        txn_count    = ('amount',        'count'),
        first_ts     = ('timestamp',     'min'),
        last_ts      = ('timestamp',     'max'),
        fraud_flag   = ('is_laundering', 'max'),
        std_amount   = ('amount',        'std'),
    ).reset_index().fillna(0)

    pair_stats['time_span'] = (pair_stats['last_ts'] - pair_stats['first_ts']).clip(lower=1)
    pair_stats['velocity']  = pair_stats['total_amount'] / pair_stats['time_span']

    logging.info(f"Unique pairs before filtering: {len(pair_stats)}")

    amount_threshold = pair_stats['total_amount'].quantile(0.25)
    fraud_accounts   = set(df[df['is_laundering'] == 1]['sender'].unique()) | set(df[df['is_laundering'] == 1]['receiver'].unique())

    method1_mask = (
        (pair_stats['fraud_flag'] == 1) |
        (pair_stats['sender'].isin(fraud_accounts)) |
        (pair_stats['receiver'].isin(fraud_accounts)) |
        (pair_stats['total_amount'] >= amount_threshold)
    )
    before = len(pair_stats)
    pair_stats = pair_stats[method1_mask].reset_index(drop=True)
    logging.info(f"Fraud-Preserving Threshold: {before} → {len(pair_stats)} edges")

    avg_txn_count    = pair_stats['txn_count'].mean()
    median_time_span = pair_stats['time_span'].median()

    method2_mask = (
        (pair_stats['fraud_flag'] == 1) |
        (pair_stats['txn_count'] >= avg_txn_count) |
        ((pair_stats['txn_count'] > 1) & (pair_stats['time_span'] <= median_time_span))
    )
    before = len(pair_stats)
    pair_stats = pair_stats[method2_mask].reset_index(drop=True)
    logging.info(f"Temporal Burst Detection:   {before} → {len(pair_stats)} edges")

    out_flow     = df.groupby('sender')['amount'].sum()
    in_flow      = df.groupby('receiver')['amount'].sum()
    all_accounts = set(out_flow.index) | set(in_flow.index)
    asymmetry    = {}
    for acc in all_accounts:
        out_val          = float(out_flow.get(acc, 0))
        in_val           = float(in_flow.get(acc, 0))
        total            = out_val + in_val
        asymmetry[acc]   = abs(out_val - in_val) / max(total, 1)

    asym_threshold     = np.percentile(list(asymmetry.values()), 60)
    high_asym_accounts = {acc for acc, score in asymmetry.items() if score >= asym_threshold}

    method3_mask = (
        (pair_stats['fraud_flag'] == 1) |
        (pair_stats['sender'].isin(high_asym_accounts)) |
        (pair_stats['receiver'].isin(high_asym_accounts))
    )
    before     = len(pair_stats)
    pair_stats = pair_stats[method3_mask].reset_index(drop=True)
    logging.info(f"Flow Asymmetry Filter:      {before} → {len(pair_stats)} edges")

    G = nx.from_pandas_edgelist(
        pair_stats,
        source='sender',
        target='receiver',
        edge_attr=['total_amount', 'avg_amount', 'max_amount', 'txn_count', 'velocity', 'fraud_flag'],
        create_using=nx.DiGraph()
    )

    df['_pk']    = list(zip(df['sender'], df['receiver']))
    kept_pairs   = set(zip(pair_stats['sender'], pair_stats['receiver']))
    df_filtered  = df[df['_pk'].isin(kept_pairs)].drop(columns=['_pk'])

    logging.info(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logging.info(f"Transactions kept: {len(df_filtered)} of {len(df)}")
    return G, df_filtered, pair_stats


def extract_features(df):
    logging.info(f"Starting graph feature extraction for {len(df)} transactions...")

    # ── simplified graph (3-method filtering) ──
    G, df, pair_stats = build_simplified_graph(df)

    # ── structural node features ──
    logging.info("Calculating PageRank (damping=0.85)...")
    pagerank_dict    = nx.pagerank(G, alpha=0.85, weight='total_amount')
    logging.info("Calculating In/Out Degrees...")
    in_degree_dict   = dict(G.in_degree(weight=None))
    out_degree_dict  = dict(G.out_degree(weight=None))
    logging.info("Calculating Clustering Coefficient...")
    clustering_dict  = nx.clustering(nx.DiGraph(G))
    logging.info("Calculating Betweenness Centrality...")
    k                = min(100, G.number_of_nodes())
    betweenness_dict = nx.betweenness_centrality(G, k=k, weight='total_amount', seed=42)

    for node in G.nodes():
        G.nodes[node]['pagerank']               = pagerank_dict.get(node, 0)
        G.nodes[node]['in_degree']              = in_degree_dict.get(node, 0)
        G.nodes[node]['out_degree']             = out_degree_dict.get(node, 0)
        G.nodes[node]['clustering_coefficient'] = clustering_dict.get(node, 0)
        G.nodes[node]['betweenness_centrality'] = betweenness_dict.get(node, 0)

    node_features = pd.DataFrame({
        'account':                list(G.nodes()),
        'pagerank':               [pagerank_dict.get(n, 0)    for n in G.nodes()],
        'in_degree':              [in_degree_dict.get(n, 0)   for n in G.nodes()],
        'out_degree':             [out_degree_dict.get(n, 0)  for n in G.nodes()],
        'clustering_coefficient': [clustering_dict.get(n, 0)  for n in G.nodes()],
        'betweenness_centrality': [betweenness_dict.get(n, 0) for n in G.nodes()],
    })

    # ── temporal bipartite graph features ──
    logging.info("Building Temporal Bipartite Graph features...")
    try:
        data_path = os.path.dirname(os.path.abspath(__file__))
        if data_path not in sys.path:
            sys.path.insert(0, data_path)

        from temporal_bipartite_graph import (
            build_temporal_bipartite_graph,
            compute_snapshot_features,
            build_final_graph_for_gat
        )

        snapshots, df = build_temporal_bipartite_graph(df)
        snapshots, account_features_over_time = compute_snapshot_features(snapshots, df)
        G_temporal = build_final_graph_for_gat(snapshots, account_features_over_time)

        temporal_feature_keys = [
            'cum_sent', 'cum_received', 'txn_count_out', 'txn_count_in',
            'asymmetry', 'velocity', 'burst_score', 'fraud_exposure',
            'delta_asymmetry', 'delta_velocity', 'delta_burst',
            'delta_fraud_exposure', 'max_burst_score', 'time_to_peak_burst'
        ]

        temporal_rows = []
        for node in G_temporal.nodes():
            acc = node.replace('acc_', '')
            row = {'account': acc}
            for feat in temporal_feature_keys:
                row[feat] = G_temporal.nodes[node].get(feat, 0.0)
            temporal_rows.append(row)

        temporal_features_df = pd.DataFrame(temporal_rows)
        node_features = node_features.merge(temporal_features_df, on='account', how='left').fillna(0)
        logging.info(f"Temporal features added: {len(temporal_feature_keys)} new features per account")

        # use temporal enriched graph for GAT
        for node in G_temporal.nodes():
            acc = node.replace('acc_', '')
            if G.has_node(acc):
                for feat in temporal_feature_keys:
                    G.nodes[acc][feat] = G_temporal.nodes[node].get(feat, 0.0)

    except Exception as e:
        logging.warning(f"Temporal bipartite graph failed: {e} — continuing without temporal features.")

    # ── GAT encoder ──
    logging.info("Running GAT encoder for learned edge weights + pruning...")

    models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)

    try:
        from graph_encoder import GraphEncoder

        fraud_senders = set(df[df['is_laundering'] == 1]['sender'].unique())
        node_labels   = {n: (1 if n in fraud_senders else 0) for n in G.nodes()}

        encoder    = GraphEncoder(dimensions=16, heads=4, epochs=200, prune_threshold=0.1, lr=0.01)
        embeddings = encoder.fit_transform(G, df=df, node_labels=node_labels)

        node_attn  = encoder.get_node_attention()
        kept_edges = set(encoder.get_pruned_edges())

        node_features['gat_attention_score'] = node_features['account'].map(lambda acc: node_attn.get(acc, 0.0))

        for dim in range(encoder.dimensions):
            node_features[f'gat_emb_{dim}'] = node_features['account'].map(
                lambda acc, d=dim: float(embeddings.get(str(acc), np.zeros(encoder.dimensions))[d])
            )

        before = len(df)
        df['_edge_key'] = list(zip(df['sender'], df['receiver']))
        df = df[df['_edge_key'].apply(lambda e: e in kept_edges)].drop(columns=['_edge_key'])
        logging.info(f"Transactions after GAT pruning: {len(df)} (removed {before - len(df)})")

    except ImportError:
        logging.warning("GraphEncoder not found — skipping GAT.")
        node_features['gat_attention_score'] = 0.0

    # ── merge all features ──
    logging.info("Merging node features to transactions...")

    df = df.merge(
        node_features.add_suffix('_sender'),
        left_on='sender', right_on='account_sender', how='left'
    ).drop(columns=['account_sender'])

    df = df.merge(
        node_features.add_suffix('_receiver'),
        left_on='receiver', right_on='account_receiver', how='left'
    ).drop(columns=['account_receiver'])

    df = df.fillna(0)
    logging.info("Feature extraction complete.")
    return df