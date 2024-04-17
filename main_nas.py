import pickle
from itertools import product

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split

from model_func_nas import LSTMGATModel, GATModel
from graph_helpers import get_dfg_from_df, unionize_dfg_sources, multigraph_transform, data_generator
from helper_func import preprocess_data, prepare_character_encoding, construct_sequences, \
    prepare_lstm_input_and_targets

from src.search_space import SearchInstance, create_search_space

ACTIVITY_KEY = 'ActivityID'
CASE_ID_KEY = 'CaseID'
TIMESTAMP_KEY = 'timestamp'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_and_prepare_graphs(main_eventlog, *additional_objects):

    model_used = "graph"

    # Read main CSV file
    df_main = pd.read_csv(f'./data/{main_eventlog}.csv')

    # Preprocess data into cases and calculate temporal features and number of related objects
    data_df = preprocess_data(df_main)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data_df, test_size=0.33)

    train_cases = np.array(train_data['caseid'])
    df1 = df_main[df_main['CaseID'].isin(train_cases)].copy()

    # Find the maximum line size
    maxlen = max(len(line) for line in data_df['lines'])

    # Prepare character encoding
    chars, char_indices, indices_char, char_act = prepare_character_encoding(data_df['lines'])

    train_sequences = construct_sequences(train_data)

    test_sequences = construct_sequences(test_data)

    # Initialize the scaler
    scaler = MinMaxScaler()


    # Prepare LSTM input and targets for training data
    X_train_lstm, y_act_train, y_times_train, time_target_means = prepare_lstm_input_and_targets(
        train_sequences,
        maxlen,
        char_indices,
        char_act, scaler)

    # Prepare LSTM input and targets for test data
    X_test_lstm, y_act_test, y_times_test, _ = prepare_lstm_input_and_targets(test_sequences, maxlen, char_indices,
                                                                              char_act, scaler)
    training_input = []
    test_input = []
    if model_used=="LSTM":
        for i in range(len(X_train_lstm)):
            d = Data()
            d.lstm_input = torch.from_numpy(X_train_lstm[i]).unsqueeze(0)
            d.y_act = torch.from_numpy(y_act_train[i]).unsqueeze(0)
            d.y_times = torch.from_numpy(y_times_train[i]).unsqueeze(0)
            training_input.append(d)
        for i in range(len(X_test_lstm)):
            d = Data()
            d.lstm_input = torch.from_numpy(X_test_lstm[i]).unsqueeze(0)
            d.y_act = torch.from_numpy(y_act_test[i]).unsqueeze(0)
            d.y_times = torch.from_numpy(y_times_test[i]).unsqueeze(0)
            test_input.append(d)
        config = [maxlen, time_target_means, char_indices]

    elif model_used=="graph":

        # Graph preprocessing
        print("Graph preprocessing...")
        dfg_sources = [(get_dfg_from_df(df1, activity_key=ACTIVITY_KEY, case_id_key=CASE_ID_KEY,
                                        timestamp_key=TIMESTAMP_KEY), main_eventlog)]
        for i, object_name in enumerate(additional_objects):
            df_dfg = pd.read_csv(f'./data/{object_name}.csv')
            dfg_data = get_dfg_from_df(df_dfg, activity_key=ACTIVITY_KEY, case_id_key=CASE_ID_KEY,
                                       timestamp_key=TIMESTAMP_KEY)
            dfg_sources.append((dfg_data, f"additional_el{i + 1}"))

        # Unionize DFG sources
        G_union = unionize_dfg_sources(dfg_sources, threshold=0)
        # Transform to simple undirected graph with edge features
        G_simple = multigraph_transform(G_union)
        # Add padding node
        G_simple.add_node(0)

        # Prepare training data
        print("Train data generation...")
        training_input, num_edge_features = data_generator(G_simple, X_train_lstm, y_act_train, y_times_train)
        print("Test data generation...")
        test_input, _ = data_generator(G_simple, X_test_lstm, y_act_test, y_times_test)

        config = [maxlen, time_target_means, char_indices, num_edge_features]
    # Save training and test inputs to pickle files

    with open(f'./pickle_files/trainset_{model_used}_{main_eventlog}.pkl', 'wb') as train_file:
        pickle.dump(training_input, train_file)

    with open(f'./pickle_files/testset_{model_used}_{main_eventlog}.pkl', 'wb') as test_file:
        pickle.dump(test_input, test_file)

    with open(f'./pickle_files/config_{model_used}_{main_eventlog}.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)


def houci_function(num_epochs, num_layers, graph_hidden_dim, graph_embedding_dim, lstm_hidden_dim, learning_rate_graph,
                   learning_rate_lstm):
    main_eventlog = "orders_complete"
    model_used = "graph"

    with open(f'./pickle_files/trainset_{model_used}_{main_eventlog}.pkl', 'rb') as train_file:
        training_input = pickle.load(train_file)

    # Load test input from pickle file
    with open(f'./pickle_files/testset_{model_used}_{main_eventlog}.pkl', 'rb') as test_file:
        test_input = pickle.load(test_file)

    # Load test input from pickle file
    with open(f'./pickle_files/config_{model_used}_{main_eventlog}.pkl', 'rb') as config_file:
        if model_used=="graph":
            Maxlen, time_target_means, Char_indices, num_edge_features = pickle.load(config_file)
        elif model_used=="LSTM":
            Maxlen, time_target_means, Char_indices = pickle.load(config_file)
            num_edge_features = 0
    # Split the training data into training and validation sets
    total_size = len(training_input)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_data, val_data = random_split(training_input, [train_size, val_size])

    # Create DataLoader instances for training and validation sets
    train_data_loaded = DataLoader(train_data, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_data_loaded = DataLoader(test_input, batch_size=64, shuffle=False)

    # Number of features
    num_node_features = 1
    num_time_features = 5

    if model_used == "graph":
        input_dim = graph_embedding_dim + num_time_features
    elif model_used == "LSTM":
        input_dim = num_time_features + 1

    gat_model = GATModel(num_node_features, graph_hidden_dim, graph_embedding_dim, num_edge_features, num_layers)
    lstm_gat_model = LSTMGATModel(gat_model, input_dim, lstm_hidden_dim, len(Char_indices))

    lstm_gat_model = lstm_gat_model.to(device)
    # Define your optimizer
    # Separate parameters for LSTM and GAT parts of the model
    lstm_parameters = list(lstm_gat_model.shared_lstm.parameters()) + list(lstm_gat_model.act_lstm.parameters()) + \
                      list(lstm_gat_model.time_lstm.parameters()) + list(lstm_gat_model.timeR_lstm.parameters())
    gat_parameters = list(lstm_gat_model.gat_model.parameters())

    optimizer_lstm = optim.NAdam(lstm_parameters, lr=learning_rate_lstm)
    optimizer_graph = optim.NAdam(gat_parameters, lr=learning_rate_graph)

    # Define your loss functions
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.L1Loss()

    # Define early stopping parameters
    # patience = 5
    # best_val_loss = float('inf')
    # counter = 0

    # Training loop
    print("Training...")
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        lstm_gat_model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, data in enumerate(train_data_loaded):
            optimizer_lstm.zero_grad()  # Zero the gradients
            optimizer_graph.zero_grad()  # Zero the gradients
            data = data.to(device)
            # Forward pass
            act_output, time_output, timeR_output = lstm_gat_model(data, model_used)
            # make_dot(act_output.mean(), params=dict(lstm_gat_model.named_parameters()))
            # Compute the classification loss
            classification_loss = classification_criterion(act_output, data.y_act)

            # Compute the regression losses
            regression_loss1 = regression_criterion(time_output, data.y_times[:, 0].unsqueeze(1))
            regression_loss2 = regression_criterion(timeR_output, data.y_times[:, 1].unsqueeze(1))

            # Combine the losses
            loss = classification_loss + regression_loss1 + regression_loss2

            # Backward pass
            loss.backward()

            # Update weights
            optimizer_lstm.step()
            optimizer_graph.step()

            # Update running loss
            running_loss += loss.item()

        # Compute average training loss for the epoch
        epoch_train_loss = running_loss / len(train_data_loaded)
        train_losses.append(epoch_train_loss)

        # Validation step
        with torch.no_grad():
            val_loss = 0.0
            for data in val_data_loader:
                data = data.to(device)
                # Forward pass
                act_output, time_output, timeR_output = lstm_gat_model(data, model_used)
                # Compute validation loss
                classification_loss = classification_criterion(act_output, data.y_act)
                regression_loss1 = regression_criterion(time_output, data.y_times[:, 0].unsqueeze(1))
                regression_loss2 = regression_criterion(timeR_output, data.y_times[:, 1].unsqueeze(1))
                val_loss += (classification_loss + regression_loss1 + regression_loss2).item()

        # Compute average validation loss for the epoch
        epoch_val_loss = val_loss / len(val_data_loader)
        val_losses.append(epoch_val_loss)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

        # # Save best model
        # if epoch_val_loss < best_val_loss:
        #     best_val_loss = epoch_val_loss
        #     torch.save(lstm_gat_model.state_dict(), f'best_model_{main_eventlog}.pt')
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         # print(f'Early stopping at epoch {epoch}')
        #         break

    print('Finished Training')

    # Evaluation

    # Set the model to evaluation mode
    lstm_gat_model.eval()

    # Lists to store predictions and ground truths
    act_preds = []
    time_preds = []
    timeR_preds = []
    act_targets = []
    time_targets = []
    timeR_targets = []

    # Iterate over the test data loader
    with torch.no_grad():
        for data in test_data_loaded:
            data = data.to(device)
            # Forward pass
            act_output, time_output, timeR_output = lstm_gat_model(data, model_used)

            # Convert predictions and targets to numpy arrays
            act_preds.extend(torch.argmax(act_output, dim=-1).cpu().numpy())
            time_preds.extend(time_output.cpu().numpy())
            timeR_preds.extend(timeR_output.cpu().numpy())
            act_targets.extend(torch.argmax(data.y_act, dim=-1).cpu().numpy())
            time_targets.extend(data.y_times[:, 0].cpu().numpy())
            timeR_targets.extend(data.y_times[:, 1].cpu().numpy())

    # Calculate classification metrics
    accuracy = accuracy_score(act_targets, act_preds)
    precision = precision_score(act_targets, act_preds, average='weighted')
    recall = recall_score(act_targets, act_preds, average='weighted')
    f1 = f1_score(act_targets, act_preds, average='weighted')

    # Calculate regression metrics
    mae_time = mean_absolute_error(time_targets, time_preds) * time_target_means[0]
    mse_time = mean_squared_error(time_targets, time_preds)
    rmse_time = np.sqrt(mse_time)
    mae_timeR = mean_absolute_error(timeR_targets, timeR_preds) * time_target_means[1]
    mse_timeR = mean_squared_error(timeR_targets, timeR_preds)
    rmse_timeR = np.sqrt(mse_timeR)

    # Print the metrics
    # print("Classification Metrics:")
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-score:", f1)
    #
    # print("\nRegression Metrics (Time):")
    # print("MAE:", mae_time)
    # print("MSE:", mse_time)
    # print("RMSE:", rmse_time)
    #
    # print("\nRegression Metrics (TimeR):")
    # print("MAE:", mae_timeR)
    # print("MSE:", mse_timeR)
    # print("RMSE:", rmse_timeR)

    metrics_dict = {
        "Metric": ["Accuracy", "F1-score", "MAE Time", "MAE TimeR"],
        "Value": [accuracy, f1, mae_time, mae_timeR]
    }

    # Create a DataFrame from the dictionary
    metrics_df = pd.DataFrame(metrics_dict)

    # Save the DataFrame to a CSV file
    metrics_df.to_csv(f"metrics_results_{model_used}_{main_eventlog}.csv", index=False)

    return accuracy


# def houci_function_list(config_list, num_epochs):
#     res = []
#     for cfg in config_list:
#         # num_layers, graph_hidden_size, graph_embedding_size, lstm_hidden_dim, learning_rate
#         cfg = list(map(int, cfg[:4])) + list(map(float, cfg[-2:]))
#         res.append(houci_function(num_epochs, *cfg))
#     return res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Hyperparameters
    lstm_hidden_dim = 100
    num_epochs = 20
    num_layers = 3
    graph_hidden_size = 1024
    graph_embedding_size = 1024
    learning_rate_graph = 0.0001
    learning_rate_lstm = 0.0001


    preprocess = True
    if preprocess:
        main_eventlog = "orders_complete"
        additional_objects = ["items_filtered", "packages_complete"]
        preprocess_and_prepare_graphs(main_eventlog, *additional_objects)

    acc = houci_function(num_epochs, num_layers, graph_hidden_size, graph_embedding_size, lstm_hidden_dim,
                         learning_rate_graph, learning_rate_lstm)
    print(acc)

    exit()

# if __name__ == '__main__':
#     num_layers_range = [3, 4, 5, 6, 7, 8]
#     graph_hidden_size_range = [32, 64, 128, 256, 512, 1024, 2048]
#     graph_embedding_size_range = [32, 64, 128, 256, 512, 1024, 2048]
#     lstm_hidden_dim_range = [50, 100]
#     learning_rate_graph_range = (np.arange(1, 10) * 1e-3).tolist() + [5e-4, 3e-2, 1e-4, 2e-4]
#     learning_rate_lstm_range = (np.arange(1,10) * 1e-3).tolist()
#     encodings = product(num_layers_range, graph_hidden_size_range,
#                         graph_embedding_size_range, lstm_hidden_dim_range,
#                         learning_rate_range)
#     encodings = torch.Tensor(list(encodings))
#     print(encodings.shape)
#
#     search_space = create_search_space(name='Exemple',
#                                        save_filename='nas/test_search_space.dill',
#                                        encodings=encodings,
#                                        encoding_to_net=None,
#                                        device='cpu')
#     search_space.preprocess_no_pretraining()
#
#     hi_fi_eval = lambda encodings_lst: houci_function_list(encodings_lst, 20)
#     hi_fi_cost = 20
#
#     lo_fi_eval = lambda encodings_lst: houci_function_list(encodings_lst, 5)
#     lo_fi_cost = 5
#
#     search_instance = SearchInstance(name='Exemple',
#                                      save_filename='nas/test_search_inst.dill',
#                                      search_space_filename='nas/test_search_space.dill',
#                                      hi_fi_eval=hi_fi_eval,
#                                      hi_fi_cost=hi_fi_cost,
#                                      lo_fi_eval=lo_fi_eval,
#                                      lo_fi_cost=lo_fi_cost,
#                                      device='cpu')
#
#     search_instance.run_search(eval_budget=int(1e6))
