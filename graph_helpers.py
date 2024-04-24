import torch
from sklearn.preprocessing import LabelEncoder
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
    for dfg_data, source_label in dfg_sources:
        for (source, target), weight in dfg_data.items():
            # Add edges with a 'source' attribute to separate graphs
            if weight >= threshold:
                edge_key = (int(source), int(target))
                if G_union.has_edge(*edge_key):
                    # If the edge already exists, update its features
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


def multigraph_transform(G_multigraph):
    G = nx.Graph()
    edges = dict()
    max_num_features = 0
    max_inner_dimensions = 0

    for (source, target, data) in G_multigraph.edges(data=True):
        features = list(data.values())
        if source < target:
            key = (source, target)
            direction = 1  # Direction from source to target
        else:
            key = (target, source)
            direction = 0  # Direction from target to source
        if key in edges:
            edges[key].append(features + [direction])
        else:
            edges[key] = [features + [direction]]
        max_num_features = max(max_num_features, len(edges[key]))
        max_inner_dimensions = max(max_inner_dimensions, len(features))

    for edge, features_list in edges.items():
        num_padding = max_num_features - len(features_list)
        padded_features = [f + [0] * (max_inner_dimensions - len(f)) for f in features_list] + [
            [0] * (max_inner_dimensions + 1)] * num_padding
        G.add_edge(edge[0], edge[1], features=padded_features)

    return G


def prepare_data(graph, seq, edge_label_encoder):
    # Label encode node IDs (activity names)
    label_encoder = LabelEncoder()
    node_ids = list(graph.nodes())
    node_ids_encoded = label_encoder.fit_transform(node_ids)

    node_id_to_encoded = dict(zip(node_ids, node_ids_encoded))

    x = torch.tensor(node_ids_encoded).unsqueeze(0).t().float()

    # Replace edge node names with corresponding node IDs
    # edges = [(label_encoder.transform([source])[0], label_encoder.transform([target])[0]) for source, target
    #          in graph.edges()]
    edges = [(list(seq).index(source), list(seq).index(target)) for source, target
             in graph.edges()]

    # Create edge index tensor
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

    # Get edge features
    edge_features = []
    for source, target, data in graph.edges(data=True):
        encoded_data = []
        for i, feature in enumerate(data['features']):
            # Label encode each feature of the inner list
            first_feature_encoded = edge_label_encoder.transform([str(feature[0])])[0]
            encoded_features = [first_feature_encoded] + feature[1:]
            encoded_data.extend(encoded_features)
        edge_features.append(torch.tensor(encoded_data).float())
    edge_attr = torch.stack(edge_features)

    return x, edge_index, edge_attr, node_id_to_encoded


def data_generator(graph, X, y_act, y_times):
    # Prepare merged data
    nodes = X[:, :, 0]
    subgraphs = [graph.subgraph(a) for a in nodes]
    subgraph_data = []

    edge_label_encoder = LabelEncoder()
    unique_first_features = set()
    # Iterate over edges to collect unique first features
    for source, target, data in graph.edges(data=True):
        for feature in data['features']:
            unique_first_features.add(str(feature[0]))
    # Fit the label encoder on the unique first features
    edge_label_encoder.fit(list(unique_first_features))

    for i in tqdm(range(len(subgraphs)), desc='Generating graph data'):
        subgraph = subgraphs[i]
        # We eliminate no edge graphs
        if len(subgraph.edges) > 0:
            _, edge_index, edge_attr, _ = prepare_data(subgraph, nodes[i], edge_label_encoder)
            degrees = torch.tensor([graph.degree[node] for node in nodes[i]]).unsqueeze(1).float()
            x = torch.tensor(nodes[i]).unsqueeze(0).t().float()
            x = torch.cat((x, degrees), dim=1)
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            d.lstm_input = torch.from_numpy(X[i, :, 1:]).unsqueeze(0)
            d.y_act = torch.from_numpy(y_act[i]).unsqueeze(0)
            d.y_times = torch.from_numpy(y_times[i]).unsqueeze(0)
            subgraph_data.append(d)
    return subgraph_data, len(edge_attr[0])
