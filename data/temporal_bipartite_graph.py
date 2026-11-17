import networkx as nx
import pandas as pd
import numpy as np
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_temporal_bipartite_graph(df):
    """
    Build cumulative temporal bipartite snapshots (account -> transaction -> account).

    IMPORTANT: no label column is read here.  This function must remain usable on
    the holdout split, where labels are not available at inference time.
    """
    logging.info("Building Temporal Bipartite Graph...")
    logging.info(f"  Total transactions: {len(df)}")

    df = df.sort_values('timestamp').reset_index(drop=True)

    if 'amount_log' not in df.columns:
        df['amount_log'] = df['amount'].apply(lambda x: math.log1p(x) if x > 0 else 0)
    if 'temporal_delta' not in df.columns:
        df['temporal_delta'] = df.groupby('sender')['timestamp'].diff().fillna(0)

    unique_steps = sorted(df['timestamp'].unique())
    logging.info(f"  Unique timestamps: {len(unique_steps)} → building {len(unique_steps)} snapshots")

    # The downstream GAT graph only needs the final cumulative snapshot.  The old
    # implementation copied the entire growing graph once per timestamp and kept
    # every copy; at ~1M transactions that is several GB of redundant graphs.
    # Build the final cumulative graph in one pass and keep only that snapshot.
    G_cumulative = nx.DiGraph()

    account_ids = pd.unique(pd.concat([df['sender'], df['receiver']], ignore_index=True))
    G_cumulative.add_nodes_from((f"acc_{acc}", {'node_type': 0}) for acc in account_ids)

    txn_nodes = [f"txn_{txn_id}" for txn_id in df['txn_id']]
    sender_nodes = [f"acc_{acc}" for acc in df['sender']]
    receiver_nodes = [f"acc_{acc}" for acc in df['receiver']]

    G_cumulative.add_nodes_from(
        (
            node,
            {
                'node_type': 1,
                'amount': float(amount),
                'amount_log': float(amount_log),
                'temporal_delta': float(temporal_delta),
                'timestamp': float(timestamp),
                'sender_acc': sender_node,
                'receiver_acc': receiver_node,
            },
        )
        for node, amount, amount_log, temporal_delta, timestamp, sender_node, receiver_node in zip(
            txn_nodes,
            df['amount'].to_numpy(),
            df['amount_log'].to_numpy(),
            df['temporal_delta'].to_numpy(),
            df['timestamp'].to_numpy(),
            sender_nodes,
            receiver_nodes,
        )
    )
    G_cumulative.add_edges_from((s, t, {'edge_type': 'sent'}) for s, t in zip(sender_nodes, txn_nodes))
    G_cumulative.add_edges_from((t, r, {'edge_type': 'received'}) for t, r in zip(txn_nodes, receiver_nodes))

    snapshots = {unique_steps[-1]: G_cumulative}

    logging.info(f"  Snapshots built: {len(unique_steps)} (final cumulative graph retained)")
    logging.info(f"  Final snapshot — nodes: {G_cumulative.number_of_nodes()}, "
                 f"edges: {G_cumulative.number_of_edges()}")
    return snapshots, df


def compute_snapshot_features(snapshots, df):
    logging.info("Computing temporal node features across snapshots...")

    unique_steps = sorted(snapshots.keys())
    if not unique_steps:
        return snapshots, {}

    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    sender_cum = df_sorted.groupby(['timestamp', 'sender']).agg(
        step_sent=('amount', 'sum'),
        step_count=('amount', 'count'),
    ).reset_index()
    receiver_cum = df_sorted.groupby(['timestamp', 'receiver']).agg(
        step_received=('amount', 'sum'),
        step_count_recv=('amount', 'count'),
    ).reset_index()

    sender_cum = sender_cum.sort_values(['sender', 'timestamp'])
    receiver_cum = receiver_cum.sort_values(['receiver', 'timestamp'])

    sender_cum['cum_sent'] = sender_cum.groupby('sender')['step_sent'].cumsum()
    sender_cum['cum_count_out'] = sender_cum.groupby('sender')['step_count'].cumsum()
    receiver_cum['cum_received'] = receiver_cum.groupby('receiver')['step_received'].cumsum()
    receiver_cum['cum_count_in'] = receiver_cum.groupby('receiver')['step_count_recv'].cumsum()

    sender_lookup = sender_cum.set_index(['sender', 'timestamp'])
    receiver_lookup = receiver_cum.set_index(['receiver', 'timestamp'])

    step_list = sorted(unique_steps)
    account_features_over_time = {}

    all_accounts = set(df_sorted['sender'].unique()) | set(df_sorted['receiver'].unique())
    logging.info(f"  Computing features for {len(all_accounts)} accounts across {len(unique_steps)} steps...")

    # Pre-index per-account timestamps once instead of filtering inside the loop.
    sender_steps = {acc: grp['timestamp'].values for acc, grp in sender_cum.groupby('sender')}
    receiver_steps = {acc: grp['timestamp'].values for acc, grp in receiver_cum.groupby('receiver')}

    for step_idx, step in enumerate(step_list):
        G = snapshots.get(step)

        recent_steps = step_list[max(0, step_idx - 3): step_idx + 1]
        recent_df = df_sorted[df_sorted['timestamp'].isin(recent_steps)]
        recent_counts = recent_df.groupby('sender').size().to_dict()

        for acc in all_accounts:
            s_idx = (acc, step)
            r_idx = (acc, step)

            if s_idx in sender_lookup.index:
                s_row = sender_lookup.loc[s_idx]
                cum_sent = float(s_row['cum_sent'])
                txn_count_out = float(s_row['cum_count_out'])
            else:
                acc_steps = sender_steps.get(acc)
                past_steps = acc_steps[acc_steps <= step] if acc_steps is not None else []
                if len(past_steps) > 0:
                    s_row = sender_lookup.loc[(acc, past_steps.max())]
                    cum_sent = float(s_row['cum_sent'])
                    txn_count_out = float(s_row['cum_count_out'])
                else:
                    cum_sent = txn_count_out = 0.0

            if r_idx in receiver_lookup.index:
                r_row = receiver_lookup.loc[r_idx]
                cum_received = float(r_row['cum_received'])
                txn_count_in = float(r_row['cum_count_in'])
            else:
                acc_steps = receiver_steps.get(acc)
                past_steps = acc_steps[acc_steps <= step] if acc_steps is not None else []
                if len(past_steps) > 0:
                    r_row = receiver_lookup.loc[(acc, past_steps.max())]
                    cum_received = float(r_row['cum_received'])
                    txn_count_in = float(r_row['cum_count_in'])
                else:
                    cum_received = txn_count_in = 0.0

            total_flow = cum_sent + cum_received
            asymmetry = abs(cum_sent - cum_received) / max(total_flow, 1)
            velocity = cum_sent / max(float(step), 1)
            avg_out = txn_count_out / max(float(step_idx + 1), 1)
            recent_out = float(recent_counts.get(acc, 0))
            burst_score = recent_out / max(avg_out, 1)

            if acc not in account_features_over_time:
                account_features_over_time[acc] = {}

            account_features_over_time[acc][step] = {
                'cum_sent': cum_sent,
                'cum_received': cum_received,
                'txn_count_out': txn_count_out,
                'txn_count_in': txn_count_in,
                'asymmetry': asymmetry,
                'velocity': velocity,
                'burst_score': burst_score,
            }

            node_key = f"acc_{acc}"
            if G is not None and G.has_node(node_key):
                for feat, val in account_features_over_time[acc][step].items():
                    G.nodes[node_key][feat] = val

    logging.info(f"  Temporal features computed for {len(account_features_over_time)} accounts")
    return snapshots, account_features_over_time


def build_final_graph_for_gat(snapshots, account_features_over_time):
    logging.info("Building final enriched graph for GAT...")

    G_final = nx.DiGraph()
    if not snapshots or not account_features_over_time:
        logging.warning("  No snapshots/accounts available; returning empty graph.")
        return G_final

    final_snapshot = snapshots[max(snapshots.keys())]

    for acc, time_features in account_features_over_time.items():
        node_key = f"acc_{acc}"
        steps = sorted(time_features.keys())
        if not steps:
            continue
        final_f = time_features[steps[-1]]
        first_f = time_features[steps[0]]

        burst_scores = [time_features[s]['burst_score'] for s in steps]
        max_burst = max(burst_scores)
        peak_burst_step = steps[int(np.argmax(burst_scores))]
        time_to_peak = float(peak_burst_step) / max(float(steps[-1]), 1)

        G_final.add_node(
            node_key,
            node_type=0,
            cum_sent=final_f['cum_sent'],
            cum_received=final_f['cum_received'],
            txn_count_out=final_f['txn_count_out'],
            txn_count_in=final_f['txn_count_in'],
            asymmetry=final_f['asymmetry'],
            velocity=final_f['velocity'],
            burst_score=final_f['burst_score'],
            delta_asymmetry=final_f['asymmetry'] - first_f['asymmetry'],
            delta_velocity=final_f['velocity'] - first_f['velocity'],
            delta_burst=final_f['burst_score'] - first_f['burst_score'],
            max_burst_score=max_burst,
            time_to_peak_burst=time_to_peak,
        )

    for node, data in final_snapshot.nodes(data=True):
        if data.get('node_type') != 1:
            continue
        sender_acc = data.get('sender_acc')
        receiver_acc = data.get('receiver_acc')
        if sender_acc and receiver_acc and G_final.has_node(sender_acc) and G_final.has_node(receiver_acc):
            G_final.add_edge(sender_acc, receiver_acc, weight=float(data.get('amount', 1.0)))

    logging.info(f"  Final enriched graph: {G_final.number_of_nodes()} nodes, "
                 f"{G_final.number_of_edges()} edges")
    return G_final
