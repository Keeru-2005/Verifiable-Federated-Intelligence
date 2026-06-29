import networkx as nx
import pandas as pd
import numpy as np
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_temporal_bipartite_graph(df):
    logging.info("Building Temporal Bipartite Graph...")
    logging.info(f"  Total transactions: {len(df)}")

    df = df.sort_values('timestamp').reset_index(drop=True)

    if 'amount_log' not in df.columns:
        df['amount_log'] = df['amount'].apply(lambda x: math.log1p(x) if x > 0 else 0)
    if 'temporal_delta' not in df.columns:
        df['temporal_delta'] = df.groupby('sender')['timestamp'].diff().fillna(0)

    unique_steps = sorted(df['timestamp'].unique())
    logging.info(f"  Unique timestamps: {len(unique_steps)} → building {len(unique_steps)} snapshots")

    snapshots = {}
    G_cumulative = nx.DiGraph()

    for step in unique_steps:
        step_txns = df[df['timestamp'] == step]

        for _, row in step_txns.iterrows():
            acc_sender   = f"acc_{row['sender']}"
            acc_receiver = f"acc_{row['receiver']}"
            txn_node     = f"txn_{row['txn_id']}"

            if not G_cumulative.has_node(acc_sender):
                G_cumulative.add_node(acc_sender, node_type=0)
            if not G_cumulative.has_node(acc_receiver):
                G_cumulative.add_node(acc_receiver, node_type=0)

            G_cumulative.add_node(txn_node,
                node_type      = 1,
                amount         = float(row['amount']),
                amount_log     = float(row['amount_log']),
                temporal_delta = float(row['temporal_delta']),
                is_laundering  = int(row['is_laundering']),
                timestamp      = float(row['timestamp']),
            )

            G_cumulative.add_edge(acc_sender,   txn_node,     edge_type='sent')
            G_cumulative.add_edge(txn_node,     acc_receiver, edge_type='received')

        snapshots[step] = G_cumulative.copy()

    logging.info(f"  Snapshots built: {len(snapshots)}")
    logging.info(f"  Final snapshot — nodes: {G_cumulative.number_of_nodes()}, edges: {G_cumulative.number_of_edges()}")
    return snapshots, df


def compute_snapshot_features(snapshots, df):
    logging.info("Computing temporal node features across snapshots...")

    unique_steps = sorted(snapshots.keys())

    # precompute all cumulative stats efficiently using vectorized pandas
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # cumulative sender stats up to each step
    sender_cum   = df_sorted.groupby(['timestamp', 'sender']).agg(
        step_sent  = ('amount', 'sum'),
        step_count = ('amount', 'count'),
        step_fraud = ('is_laundering', 'sum'),
    ).reset_index()

    receiver_cum = df_sorted.groupby(['timestamp', 'receiver']).agg(
        step_received    = ('amount', 'sum'),
        step_count_recv  = ('amount', 'count'),
        step_fraud_recv  = ('is_laundering', 'sum'),
    ).reset_index()

    # build cumulative sums across time per account
    sender_cum   = sender_cum.sort_values(['sender', 'timestamp'])
    receiver_cum = receiver_cum.sort_values(['receiver', 'timestamp'])

    sender_cum['cum_sent']      = sender_cum.groupby('sender')['step_sent'].cumsum()
    sender_cum['cum_count_out'] = sender_cum.groupby('sender')['step_count'].cumsum()
    sender_cum['cum_fraud_out'] = sender_cum.groupby('sender')['step_fraud'].cumsum()

    receiver_cum['cum_received']   = receiver_cum.groupby('receiver')['step_received'].cumsum()
    receiver_cum['cum_count_in']   = receiver_cum.groupby('receiver')['step_count_recv'].cumsum()
    receiver_cum['cum_fraud_recv'] = receiver_cum.groupby('receiver')['step_fraud_recv'].cumsum()

    # build lookup dicts for fast access
    sender_lookup   = sender_cum.set_index(['sender', 'timestamp'])
    receiver_lookup = receiver_cum.set_index(['receiver', 'timestamp'])

    # for burst score — recent window = last 3 steps
    step_list = sorted(unique_steps)

    account_features_over_time = {}

    all_accounts = set(df_sorted['sender'].unique()) | set(df_sorted['receiver'].unique())

    logging.info(f"  Computing features for {len(all_accounts)} accounts across {len(unique_steps)} steps...")

    for step_idx, step in enumerate(step_list):
        G = snapshots[step]

        # recent burst window
        recent_steps  = step_list[max(0, step_idx - 3): step_idx + 1]
        recent_df     = df_sorted[df_sorted['timestamp'].isin(recent_steps)]
        recent_counts = recent_df.groupby('sender').size().to_dict()

        for acc in all_accounts:
            # get latest cumulative stats up to this step
            s_idx = (acc, step)
            r_idx = (acc, step)

            # find most recent available step for this account
            if s_idx in sender_lookup.index:
                s_row = sender_lookup.loc[s_idx]
                cum_sent      = float(s_row['cum_sent'])
                txn_count_out = float(s_row['cum_count_out'])
                cum_fraud_out = float(s_row['cum_fraud_out'])
            else:
                # get last available step
                acc_steps = sender_cum[sender_cum['sender'] == acc]['timestamp']
                past_steps = acc_steps[acc_steps <= step]
                if len(past_steps) > 0:
                    last_step = past_steps.max()
                    s_row = sender_lookup.loc[(acc, last_step)]
                    cum_sent      = float(s_row['cum_sent'])
                    txn_count_out = float(s_row['cum_count_out'])
                    cum_fraud_out = float(s_row['cum_fraud_out'])
                else:
                    cum_sent = txn_count_out = cum_fraud_out = 0.0

            if r_idx in receiver_lookup.index:
                r_row = receiver_lookup.loc[r_idx]
                cum_received  = float(r_row['cum_received'])
                txn_count_in  = float(r_row['cum_count_in'])
                cum_fraud_recv = float(r_row['cum_fraud_recv'])
            else:
                acc_steps = receiver_cum[receiver_cum['receiver'] == acc]['timestamp']
                past_steps = acc_steps[acc_steps <= step]
                if len(past_steps) > 0:
                    last_step = past_steps.max()
                    r_row = receiver_lookup.loc[(acc, last_step)]
                    cum_received   = float(r_row['cum_received'])
                    txn_count_in   = float(r_row['cum_count_in'])
                    cum_fraud_recv = float(r_row['cum_fraud_recv'])
                else:
                    cum_received = txn_count_in = cum_fraud_recv = 0.0

            total_flow     = cum_sent + cum_received
            asymmetry      = abs(cum_sent - cum_received) / max(total_flow, 1)
            velocity       = cum_sent / max(float(step), 1)
            avg_out        = txn_count_out / max(float(step_idx + 1), 1)
            recent_out     = float(recent_counts.get(acc, 0))
            burst_score    = recent_out / max(avg_out, 1)
            total_count    = txn_count_out + txn_count_in
            fraud_exposure = (cum_fraud_out + cum_fraud_recv) / max(total_count, 1)

            if acc not in account_features_over_time:
                account_features_over_time[acc] = {}

            account_features_over_time[acc][step] = {
                'cum_sent':       cum_sent,
                'cum_received':   cum_received,
                'txn_count_out':  txn_count_out,
                'txn_count_in':   txn_count_in,
                'asymmetry':      asymmetry,
                'velocity':       velocity,
                'burst_score':    burst_score,
                'fraud_exposure': fraud_exposure,
            }

            node_key = f"acc_{acc}"
            if G.has_node(node_key):
                for feat, val in account_features_over_time[acc][step].items():
                    G.nodes[node_key][feat] = val

    logging.info(f"  Temporal features computed for {len(account_features_over_time)} accounts")
    return snapshots, account_features_over_time


def build_final_graph_for_gat(snapshots, account_features_over_time):
    logging.info("Building final enriched graph for GAT...")

    final_snapshot = snapshots[max(snapshots.keys())]

    G_final    = nx.DiGraph()
    unique_steps = sorted(list(list(account_features_over_time.values())[0].keys()))

    for acc, time_features in account_features_over_time.items():
        node_key = f"acc_{acc}"
        steps    = sorted(time_features.keys())
        final_f  = time_features[steps[-1]]
        first_f  = time_features[steps[0]]

        burst_scores    = [time_features[s]['burst_score'] for s in steps]
        max_burst       = max(burst_scores)
        peak_burst_step = steps[int(np.argmax(burst_scores))]
        time_to_peak    = float(peak_burst_step) / max(float(steps[-1]), 1)

        G_final.add_node(node_key,
            node_type            = 0,
            cum_sent             = final_f['cum_sent'],
            cum_received         = final_f['cum_received'],
            txn_count_out        = final_f['txn_count_out'],
            txn_count_in         = final_f['txn_count_in'],
            asymmetry            = final_f['asymmetry'],
            velocity             = final_f['velocity'],
            burst_score          = final_f['burst_score'],
            fraud_exposure       = final_f['fraud_exposure'],
            delta_asymmetry      = final_f['asymmetry']      - first_f['asymmetry'],
            delta_velocity       = final_f['velocity']       - first_f['velocity'],
            delta_burst          = final_f['burst_score']    - first_f['burst_score'],
            delta_fraud_exposure = final_f['fraud_exposure'] - first_f['fraud_exposure'],
            max_burst_score      = max_burst,
            time_to_peak_burst   = time_to_peak,
        )

    for u, v, data in final_snapshot.edges(data=True):
        u_type = final_snapshot.nodes[u].get('node_type', -1)
        v_type = final_snapshot.nodes[v].get('node_type', -1)

        if u_type == 0 and v_type == 1:
            acc_v = v.replace('txn_', 'acc_')
            if G_final.has_node(u) and G_final.has_node(acc_v):
                G_final.add_edge(u, acc_v, **data)
        elif u_type == 1 and v_type == 0:
            acc_u = u.replace('txn_', 'acc_')
            if G_final.has_node(acc_u) and G_final.has_node(v):
                G_final.add_edge(acc_u, v, **data)

    logging.info(f"  Final enriched graph: {G_final.number_of_nodes()} nodes, {G_final.number_of_edges()} edges")
    return G_final