import os.path

import matplotlib
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


from model_utils import LSTMGATModel, GATModel, LSTMModel, initialize_models, initialize_optimizers, train_model, \
    evaluate_model, load_model, save_results
from preprocess import preprocess_and_prepare_graphs, load_data

matplotlib.use('TkAgg')  # or 'TkAgg', 'Qt5Agg', etc., depending on your environment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PREPROCESS = True
MODEL_USED = "GRAPH"
OCEL = "order-management"
MAIN_OT = "items"
OCDFG = ["items", "orders", "packages"]
TASK_TYPE = "all"


def main_function(preprocess, model_used, ocel, main_ot, ocdfg, num_epochs=20, num_layers=1, dropout=0,
                  graph_hidden_dim=8,
                  graph_embedding_dim=8, lstm_hidden_dim=100,
                  learning_rate_graph=0.0003,
                  learning_rate_lstm=0.001, threshold=0, patience=5, task_type=TASK_TYPE):
    if preprocess:
        preprocess_and_prepare_graphs(model_used, ocel, main_ot, ocdfg, threshold)
    training_input, test_input, Maxlen, time_target_means, Char_indices, num_edge_features = load_data(model_used, ocel,
                                                                                                       main_ot)
    num_target_chars = len(Char_indices)
    total_size = len(training_input)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    generator1 = torch.Generator().manual_seed(1)
    train_data, val_data = random_split(training_input, [train_size, val_size], generator=generator1)
    train_data_loaded = DataLoader(train_data, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_data_loaded = DataLoader(test_input, batch_size=64, shuffle=False)

    num_node_features = 2
    num_time_features = 5

    if MODEL_USED == "GRAPH":
        input_dim = graph_embedding_dim + num_time_features
    elif MODEL_USED == "LSTM":
        input_dim = num_time_features + 1

    lstm_gat_model = initialize_models(num_node_features, dropout, graph_hidden_dim, graph_embedding_dim,
                                       num_edge_features, num_layers, input_dim, lstm_hidden_dim, num_target_chars)
    optimizer_lstm, optimizer_graph = initialize_optimizers(lstm_gat_model, learning_rate_lstm, learning_rate_graph)

    _ = train_model(lstm_gat_model, model_used, train_data_loaded, val_data_loader, num_epochs, optimizer_lstm,
                                 optimizer_graph, MODELS_DIR,  patience, task_type)
    best_model_path = os.path.join(MODELS_DIR, f'best_model_{MODEL_USED}.pth')
    # Initialize the components of the LSTMGATModel
    gat_model = GATModel(num_node_features, dropout, graph_hidden_dim, graph_embedding_dim, num_edge_features,
                         num_layers)
    lstm_model = LSTMModel(input_dim, lstm_hidden_dim, num_target_chars)

    # Create the combined model
    lstm_gat_model = LSTMGATModel(gat_model, lstm_model)
    lstm_gat_model = load_model(lstm_gat_model, best_model_path, device)

    metrics = evaluate_model(lstm_gat_model, model_used, test_data_loaded, time_target_means, task_type)
    save_results(metrics, RESULTS_DIR, model_used, task_type)

    return metrics["Accuracy"]  # Returning accuracy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Hyperparameters
    LSTM_HIDDEN_DIM = 100
    NUM_EPOCHS = 2
    NUM_LAYERS = 3
    GRAPH_HIDDEN_SIZE = 24
    GRAPH_EMBEDDING_SIZE = 24
    LR_GRAPH = 0.0003
    LR_LSTM = 0.0003
    THRESHOLD = 0
    PATIENCE = 10
    DROPOUT = 0

    PARAMS = [NUM_EPOCHS, NUM_LAYERS, DROPOUT, GRAPH_HIDDEN_SIZE, GRAPH_EMBEDDING_SIZE, LSTM_HIDDEN_DIM,
              LR_GRAPH, LR_LSTM, THRESHOLD, PATIENCE, TASK_TYPE]

    MODELS_DIR = os.path.join('.', 'models', OCEL, MAIN_OT, f'best_model_{MODEL_USED}.pt')
    os.makedirs(MODELS_DIR, exist_ok=True)

    RESULTS_DIR = os.path.join('.', 'results', OCEL, MAIN_OT)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    PREPROCESS = True
    MODEL_USED = "GRAPH"
    OCEL = "order-management"
    MAIN_OT = "orders"
    OCDFG = ["items", "orders", "packages"]
    TASK_TYPE = "all"

    acc = main_function(PREPROCESS, MODEL_USED, OCEL, MAIN_OT, OCDFG, *PARAMS)
    print(acc)

    exit()
