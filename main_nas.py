import pickle
from itertools import product

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.optim as optim

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
    training_input = data_generator(G_simple, X_train_lstm, y_act_train, y_times_train)
    print("Test data generation...")
    test_input = data_generator(G_simple, X_test_lstm, y_act_test, y_times_test)

    config = [maxlen, time_target_means, char_indices]
    # Save training and test inputs to pickle files

    with open(f'./pickle_files/trainset_{main_eventlog}.pkl', 'wb') as train_file:
        pickle.dump(training_input, train_file)

    with open(f'./pickle_files/testset_{main_eventlog}.pkl', 'wb') as test_file:
        pickle.dump(test_input, test_file)

    with open(f'./pickle_files/config_{main_eventlog}.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)


def houci_function(num_epochs, num_layers, graph_hidden_dim, graph_embedding_dim, lstm_hidden_dim, learning_rate):
    main_eventlog = "orders_complete"

    with open(f'./pickle_files/trainset_{main_eventlog}.pkl', 'rb') as train_file:
        training_input = pickle.load(train_file)

    # Load test input from pickle file
    with open(f'./pickle_files/testset_{main_eventlog}.pkl', 'rb') as test_file:
        test_input = pickle.load(test_file)

    # Load test input from pickle file
    with open(f'./pickle_files/config_{main_eventlog}.pkl', 'rb') as config_file:
        Maxlen, time_target_means, Char_indices = pickle.load(config_file)

    train_data_loaded = DataLoader(training_input, batch_size=32, shuffle=True)
    test_data_loaded = DataLoader(test_input, batch_size=32, shuffle=False)

    # Number of features
    num_node_features = 1
    num_edge_features = 6
    num_lstm_features = 6

    using_graph = True
    input_dim = graph_embedding_dim * using_graph + num_lstm_features

    gat_model = GATModel(num_node_features, graph_hidden_dim, graph_embedding_dim, num_edge_features, num_layers)
    lstm_gat_model = LSTMGATModel(gat_model, input_dim, lstm_hidden_dim, len(Char_indices))

    lstm_gat_model = lstm_gat_model.to(device)
    # Define your optimizer
    optimizer = optim.Adam(lstm_gat_model.parameters(), lr=learning_rate)

    # Define your loss functions
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.L1Loss()

    # Training loop
    # print("Training...")

    for epoch in range(num_epochs):
        lstm_gat_model.train()  # Set the model to training mode
        running_classification_loss = 0.0
        running_regression_loss = 0.0
        for batch_idx, data in enumerate(train_data_loaded):
            optimizer.zero_grad()  # Zero the gradients
            data = data.to(device)
            # Forward pass
            act_output, time_output, timeR_output = lstm_gat_model(data, using_graph)
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

            # plot_grad_flow(lstm_gat_model.named_parameters())
            # Update weights
            optimizer.step()

            # Print statistics
            # running_classification_loss += classification_loss.item()
            # running_regression_loss += (regression_loss1.item() + regression_loss2.item())
            # if batch_idx % 100 == 99:  # Print every 100 mini-batches
            #     print('[%d, %5d] classification loss: %.3f, regression loss: %.3f' %
            #           (epoch + 1, batch_idx + 1, running_classification_loss / 100, running_regression_loss / 100))
            #     running_classification_loss = 0.0
            #     running_regression_loss = 0.0

    # print('Finished Training')

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
            act_output, time_output, timeR_output = lstm_gat_model(data, using_graph)

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

    return accuracy


def houci_function_list(config_list, num_epochs):
    res = []
    for cfg in config_list:
        # num_layers, graph_hidden_size, graph_embedding_size, lstm_hidden_dim, learning_rate
        cfg = list(map(int, cfg[:4])) + [float(cfg[-1])]
        res.append(houci_function(num_epochs, *cfg))
    return res


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
#     # Hyperparameters
#     lstm_hidden_dim = 100
#     num_epochs = 10
#     num_layers = 2
#     graph_hidden_size = 32
#     graph_embedding_size = 32
#     learning_rate = 0.005
#
#     main_eventlog = "orders_complete"
#     additional_objects = ["items_filtered", "packages_complete"]
#
#     preprocess = False
#     if preprocess:
#         preprocess_and_prepare_graphs(main_eventlog, *additional_objects)
#
#     acc = houci_function(num_epochs, num_layers, graph_hidden_size, graph_embedding_size, lstm_hidden_dim,
#                          learning_rate)
#     print(acc)
#
#     exit()

if __name__ == '__main__':
    num_layers_range = [3, 4, 5, 6, 7, 8]
    graph_hidden_size_range = [16, 32, 64, 128, 256, 512, 1024, 2048]
    graph_embedding_size_range = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    lstm_hidden_dim_range = [100]
    learning_rate_range = (np.arange(1, 10) * 1e-3).tolist() + [5e-4, 3e-2, 1e-4, 2e-4]
    encodings = product(num_layers_range, graph_hidden_size_range,
                        graph_embedding_size_range, lstm_hidden_dim_range,
                        learning_rate_range)
    encodings = torch.Tensor(list(encodings))
    print(encodings.shape)

    search_space = create_search_space(name='Exemple',
                                       save_filename='nas/test_search_space.dill',
                                       encodings=encodings,
                                       encoding_to_net=None,
                                       device='cpu')
    search_space.preprocess_no_pretraining()

    hi_fi_eval = lambda encodings_lst: houci_function_list(encodings_lst, 20)
    hi_fi_cost = 20

    lo_fi_eval = lambda encodings_lst: houci_function_list(encodings_lst, 5)
    lo_fi_cost = 5

    search_instance = SearchInstance(name='Exemple',
                                     save_filename='nas/test_search_inst.dill',
                                     search_space_filename='nas/test_search_space.dill',
                                     hi_fi_eval=hi_fi_eval,
                                     hi_fi_cost=hi_fi_cost,
                                     lo_fi_eval=lo_fi_eval,
                                     lo_fi_cost=lo_fi_cost,
                                     device='cpu')

    search_instance.run_search(eval_budget=int(1e6))
