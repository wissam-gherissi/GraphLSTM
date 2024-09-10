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


def prepare_data(graph, edge_label_encoder):
    # Add degree to node features
    degree_dict = dict(graph.degree())
    degrees = torch.tensor(
        [degree_dict[int(node)] if node in degree_dict.keys() else 0 for node in list(graph.nodes)]).unsqueeze(
        1).float()

    x = torch.tensor(list(graph.nodes)).unsqueeze(0).t().float()
    x_old = torch.cat((x, degrees), dim=1)
    x = torch.tensor(list(graph.degree())).float()

    # Replace edge node names with corresponding node IDs
    # edges = [(label_encoder.transform([source])[0], label_encoder.transform([target])[0]) for source, target
    #          in graph.edges()]

    edges = [(list(graph.nodes).index(source), list(graph.nodes).index(target)) for source, target
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

    return x, edge_index, edge_attr


def data_generator(graph, X, y_act, y_times, edge_label_encoder):
    # Prepare merged data
    nodes = X[:, :, 0]
    graph_data = []

    unique_first_features = set()
    # Iterate over edges to collect unique first features
    for source, target, data in graph.edges(data=True):
        for feature in data['features']:
            unique_first_features.add(str(feature[0]))
    # Fit the label encoder on the unique first features
    edge_label_encoder.fit(list(unique_first_features))
    x, edge_index, edge_attr = prepare_data(graph, edge_label_encoder)

    for i in tqdm(range(len(nodes)), desc='Generating graph data'):
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        d.lstm_input = torch.from_numpy(X[i, :,]).unsqueeze(0)
        d.y_act = torch.from_numpy(y_act[i]).unsqueeze(0)
        d.y_times = torch.from_numpy(y_times[i]).unsqueeze(0)
        graph_data.append(d)
    return graph_data, len(edge_attr[0])


def data_generator_ancien(graph, X, y_act, y_times):
    # Prepare merged data
    nodes = X[:, :, 0]
    subgraph_data = []

    edge_label_encoder = LabelEncoder()
    unique_first_features = set()
    # Iterate over edges to collect unique first features
    for source, target, data in graph.edges(data=True):
        for feature in data['features']:
            unique_first_features.add(str(feature[0]))
    # Fit the label encoder on the unique first features
    edge_label_encoder.fit(list(unique_first_features))

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
