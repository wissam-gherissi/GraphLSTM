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


def main_function(preprocess, model_used, models_dir, results_dir, ocel, main_ot, ocdfg, num_epochs=20, num_layers=1,
                  dropout=0,
                  graph_hidden_dim=8,
                  graph_embedding_dim=8, lstm_hidden_dim=100,
                  learning_rate_graph=0.0003,
                  learning_rate_lstm=0.001, threshold=0, patience=5, task_type="all"):
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

    if model_used == "GRAPH":
        input_dim = graph_embedding_dim + num_time_features
    elif model_used == "LSTM":
        input_dim = num_time_features + 1

    lstm_gat_model = initialize_models(num_node_features, dropout, graph_hidden_dim, graph_embedding_dim,
                                       num_edge_features, num_layers, input_dim, lstm_hidden_dim, num_target_chars)
    optimizer_lstm, optimizer_graph = initialize_optimizers(lstm_gat_model, learning_rate_lstm, learning_rate_graph)

    _ = train_model(lstm_gat_model, model_used, threshold, train_data_loaded, val_data_loader, num_epochs, optimizer_lstm,
                    optimizer_graph, models_dir, patience, task_type)
    best_model_path = os.path.join(models_dir, f'best_model_{model_used}_{threshold}.pth')
    # Initialize the components of the LSTMGATModel
    gat_model = GATModel(num_node_features, dropout, graph_hidden_dim, graph_embedding_dim, num_edge_features,
                         num_layers)
    lstm_model = LSTMModel(input_dim, lstm_hidden_dim, num_target_chars)

    # Create the combined model
    lstm_gat_model = LSTMGATModel(gat_model, lstm_model)
    lstm_gat_model = load_model(lstm_gat_model, best_model_path, device)

    metrics = evaluate_model(lstm_gat_model, model_used, test_data_loaded, time_target_means, task_type)
    save_results(metrics, results_dir, model_used, task_type, threshold)

    return metrics  # Returning accuracy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Hyperparameters
    LSTM_HIDDEN_DIM = 100
    NUM_EPOCHS = 500
    NUM_LAYERS = 3
    GRAPH_HIDDEN_SIZE = 256
    GRAPH_EMBEDDING_SIZE = 256
    LR_GRAPH = 0.0003
    LR_LSTM = 0.0003
    PATIENCE = 1
    DROPOUT = 0

    PREPROCESS = True
    OCEL = "order-management"
    OCDFG = ["items", "orders", "packages"]
    TASK_TYPE = "all"

    THRESHOLD_OPTIONS = [0, 10, 100, 500]  # Example values
    MODEL_USED_OPTIONS = ["LSTM", "GRAPH"]  # Example values
    MAIN_OT_OPTIONS = ["packages", "orders", "items"]  # Example values
    TASK_TYPE_OPTIONS = ["all", "next", "classification", "regression", "remaining"]

    for MODEL_USED in MODEL_USED_OPTIONS:
        for MAIN_OT in MAIN_OT_OPTIONS:
            for TASK_TYPE in TASK_TYPE_OPTIONS:
                # Only iterate over thresholds if the model is GRAPH
                thresholds = THRESHOLD_OPTIONS if MODEL_USED == "GRAPH" else [0]

                for THRESHOLD in thresholds:
                    # Define PARAMS with the current value of THRESHOLD
                    PARAMS = [NUM_EPOCHS, NUM_LAYERS, DROPOUT, GRAPH_HIDDEN_SIZE, GRAPH_EMBEDDING_SIZE, LSTM_HIDDEN_DIM,
                              LR_GRAPH, LR_LSTM, THRESHOLD, PATIENCE, TASK_TYPE]

                    # Set up directories for models and results
                    MODELS_DIR = os.path.join('.', 'models', OCEL, MAIN_OT, TASK_TYPE)
                    RESULTS_DIR = os.path.join('.', 'results', OCEL, MAIN_OT, TASK_TYPE)
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    os.makedirs(RESULTS_DIR, exist_ok=True)

                    # Run main_function with the current combination
                    metrics = main_function(PREPROCESS, MODEL_USED, MODELS_DIR, RESULTS_DIR, OCEL, MAIN_OT, OCDFG,
                                            *PARAMS)

                    # Print the results for this combination
                    print(f"Model: {MODEL_USED}, Task: {TASK_TYPE}, Main OT: {MAIN_OT}, Threshold: {THRESHOLD}, "
                          f"Metrics: {metrics}")
    exit()
