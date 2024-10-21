import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pm4py
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm


def get_dfg_from_df(df, activity_key='ActivityID', case_id_key='CaseID', timestamp_key='timestamp'):
    # Obtain the DFG data
    dfg_data = pm4py.algo.discovery.dfg.adapters.pandas.df_statistics.get_dfg_graph(
        df,
        activity_key=activity_key,
        case_id_glue=case_id_key,
        timestamp_key=timestamp_key
    )

    return dfg_data


def unionize_dfg_sources(dfg_sources, threshold=0):
    # Create an empty Directed Graph
    G_union = nx.DiGraph()
    max_features_length = 0
    # First, find the maximum weight across all DFGs
    all_weights = [weight for dfg_data, _ in dfg_sources for (_, _), weight in dfg_data.items()]
    mean_weight = sum(all_weights)/len(all_weights)

    # Compute the actual threshold based on the max weight and the input threshold percentage
    actual_threshold = (threshold/100) * mean_weight
    for dfg_data, source_label in dfg_sources:
        for (source, target), weight in dfg_data.items():
            # Add edges with a 'source' attribute to separate graphs
            if weight >= actual_threshold:
                edge_key = (int(source), int(target))
                if G_union.has_edge(*edge_key):
                    # If the edge already exists, concatenate the features
                    existing_data = G_union.get_edge_data(*edge_key)
                    existing_data['features'].append([source_label, weight])
                else:
                    # If the edge doesn't exist, add it with the new features
                    G_union.add_edge(edge_key[0], edge_key[1], features=[[source_label, weight]])
    # Calculate the maximum length of features lists
    for _, _, data in G_union.edges(data=True):
        max_features_length = max(max_features_length, len(data['features']))

    # Pad features lists to ensure each edge has the same number of inner lists
    for _, _, data in G_union.edges(data=True):
        data['features'] += [[0, 0]] * (max_features_length - len(data['features']))

    return G_union


def prepare_data(graph, onehot_encoder, scaler):
    # Add degree to node features
    degree_dict = dict(graph.degree())
    degrees = torch.tensor([degree_dict.get(int(node), 0) for node in list(graph.nodes)]).unsqueeze(1).float()

    # Assuming nodes are represented by their indices
    x = torch.tensor(list(graph.nodes)).unsqueeze(0).t().float()

    # Concatenating node degrees as an additional feature
    x_features = torch.cat((x, degrees), dim=1)

    edges = [(list(graph.nodes).index(source), list(graph.nodes).index(target)) for source, target
             in graph.edges()]
    # Create edge index tensor
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

    # Prepare edge features
    edge_features = []
    source_features = []
    weight_features = []

    # Collect all 'source' and 'weight' features for one-hot encoding and scaling
    for source, target, data in graph.edges(data=True):
        for feature in data['features']:
            source_features.append([str(feature[0])])
            weight_features.append([feature[1]])

    # One-hot encode the source features
    onehot_encoded = onehot_encoder.transform(source_features)
    # Scale weights features
    scaled_weights = scaler.transform(weight_features)

    for i, (source, target, data) in enumerate(graph.edges(data=True)):
        onehot_combined = []
        scaled_weight_combined = []

        # Collect features for this edge
        for feature in data['features']:
                onehot = onehot_encoded[source_features.index([str(feature[0])])].toarray().flatten()
                scaled_weight = scaled_weights[weight_features.index([feature[1]])]

                onehot_combined.append(onehot)
                scaled_weight_combined.append(scaled_weight)

        # Flatten and combine all features for this edge
        combined_features = np.concatenate(onehot_combined + scaled_weight_combined).flatten()
        edge_features.append(torch.tensor(combined_features).float())

    # Stack all edge features into a tensor
    edge_attr = torch.stack(edge_features)

    return x_features, edge_index, edge_attr


def data_generator(graph, X, y_act, y_times, onehot_encoder, scaler, training=True):
    # Prepare merged data
    nodes = X[:, :, 0]
    graph_data = []

    source_features = []
    weight_features = []

    if training:
        # Collect all 'source' and 'weight' features for one-hot encoding and scaling
        for source, target, data in graph.edges(data=True):
            for feature in data['features']:
                source_features.append([str(feature[0])])
                weight_features.append([feature[1]])

        # Fit the onehot_encoder on the collected source features
        onehot_encoder.fit(np.array(source_features).reshape(-1, 1))

        # Fit the scaler on the collected weight features
        scaler.fit(np.array(weight_features).reshape(-1, 1))

    x, edge_index, edge_attr = prepare_data(graph, onehot_encoder, scaler)

    for i in tqdm(range(len(nodes)), desc='Generating graph data'):
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        d.lstm_input = torch.from_numpy(X[i, :, ]).unsqueeze(0)
        d.y_act = torch.from_numpy(y_act[i]).unsqueeze(0)
        d.y_times = torch.from_numpy(y_times[i]).unsqueeze(0)
        graph_data.append(d)

    return graph_data, len(edge_attr[0])


def data_generator_ancien(graph, X, y_act, y_times):
    # Prepare merged data
    nodes = X[:, :, 0]
    subgraph_data = []

    edge_label_encoder = LabelEncoder()
    unique_object_types = set()
    # Iterate over edges to collect unique first features
    for source, target, data in graph.edges(data=True):
        for feature in data['features']:
            unique_object_types.add(str(feature[0]))
    # Fit the label encoder on the unique first features
    edge_label_encoder.fit(list(unique_object_types))

    for i in tqdm(range(len(nodes)), desc='Generating graph data'):
        subgraph = graph.subgraph(nodes[i])
        # We eliminate no edge graphs
        if len(subgraph.edges) > 0:
            x, edge_index, edge_attr = prepare_data(subgraph, nodes[i], edge_label_encoder)
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            d.lstm_input = torch.from_numpy(X[i, :, 1:]).unsqueeze(0)
            d.y_act = torch.from_numpy(y_act[i]).unsqueeze(0)
            d.y_times = torch.from_numpy(y_times[i]).unsqueeze(0)
            subgraph_data.append(d)
    return subgraph_data, len(edge_attr[0])
