import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTIVITY_KEY = 'ActivityID'
CASE_ID_KEY = 'CaseID'
TIMESTAMP_KEY = 'timestamp'


class GATModel(torch.nn.Module):
    def __init__(self, num_features, dropout, hidden_size, target_size, num_edge_features, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_edge_features = num_edge_features
        self.num_layers = num_layers
        self.add_self_loops = True
        self.dropout = dropout
        self.bn_gnn = nn.BatchNorm1d(self.target_size)

        # Create a list of GATConv layers
        self.convs = torch.nn.ModuleList(
            [GATConv(self.num_features, self.hidden_size, edge_dim=self.num_edge_features,
                     add_self_loops=self.add_self_loops)] + [
                GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features,
                        add_self_loops=self.add_self_loops)
                for _ in range(self.num_layers - 2)] +
            [GATConv(self.hidden_size, self.target_size, edge_dim=self.num_edge_features,
                     add_self_loops=self.add_self_loops)])

        # Linear layer for final output
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data, batch_size):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply multiple GATConv layers
        for conv in self.convs:
            x = F.gelu(conv(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply linear layer for final output
        # x = self.linear(x)

        # Reshape output
        x = torch.reshape(x, (batch_size, x.shape[0] // batch_size, self.target_size))
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_target_chars):
        super(LSTMModel, self).__init__()
        self.shared_lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                                   dropout=0.2, batch_first=True)
        self.bn_shared = nn.BatchNorm1d(lstm_hidden_dim)
        self.act_lstm = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                                dropout=0.2, batch_first=True)
        self.time_lstm = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                                 dropout=0.2, batch_first=True)
        self.timeR_lstm = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                                  dropout=0.2, batch_first=True)
        self.bn_act = nn.BatchNorm1d(lstm_hidden_dim)
        self.bn_time = nn.BatchNorm1d(lstm_hidden_dim)
        self.bn_timeR = nn.BatchNorm1d(lstm_hidden_dim)
        self.fc_act = nn.Linear(lstm_hidden_dim, num_target_chars)
        self.fc_time = nn.Linear(lstm_hidden_dim, 1)
        self.fc_timeR = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, lstm_input, task_type):
        # Pass the concatenated input through the shared LSTM layer
        shared_lstm_output, _ = self.shared_lstm(lstm_input)

        # Apply batch normalization after the shared LSTM layer
        shared_lstm_output = self.bn_shared(shared_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)

        act_output, time_output, timeR_output = None, None, None

        # For classification task
        if task_type == "classification" or task_type == "next" or task_type == "all":
            act_lstm_output, _ = self.act_lstm(shared_lstm_output)
            act_lstm_output = self.bn_act(act_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)
            act_output_last = act_lstm_output[:, -1, :]
            act_output = F.softmax(self.fc_act(act_output_last), dim=1)

        # For regression tasks
        if task_type == "regression" or task_type == "next" or task_type == "all":
            time_lstm_output, _ = self.time_lstm(shared_lstm_output)
            time_lstm_output = self.bn_time(time_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)
            time_output_last = time_lstm_output[:, -1, :]
            time_output = self.fc_time(time_output_last)

        if task_type == "regression" or task_type == "remaining" or task_type == "all":
            timeR_lstm_output, _ = self.timeR_lstm(shared_lstm_output)
            timeR_lstm_output = self.bn_timeR(timeR_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)
            timeR_output_last = timeR_lstm_output[:, -1, :]
            timeR_output = self.fc_timeR(timeR_output_last)

        return act_output, time_output, timeR_output


class LSTMGATModel(torch.nn.Module):
    def __init__(self, gat_model, lstm_model):
        super(LSTMGATModel, self).__init__()
        self.gat_model = gat_model
        self.lstm_model = lstm_model

    def forward(self, data, model_used, task_type):
        # Pass the graph through the GAT model
        data_input = data.lstm_input

        if model_used == "GRAPH":
            gat_output = self.gat_model(data, data_input.shape[0])
            gat_output = self.gat_model.bn_gnn(gat_output.permute(0, 2, 1)).permute(0, 2, 1)

            device = data_input.device

            # Get the first row from each batch (shape: [batch_size, sequence_length])
            first_row = data_input[:, :, 0]

            # Create a tensor to hold the concatenated output (shape:[batch_size,sequence_length,input_size +
            # gat_output_size])
            concatenated_rows = torch.zeros(data_input.size(0), data_input.size(1),
                                            data_input.size(2) + gat_output.size(2), device=device)

            # Vectorized search for matching indices
            for batch_idx in range(first_row.size(0)):
                # Create a mask for matching indices between first_row and data.x
                matching_indices = torch.zeros(first_row.size(1), dtype=torch.long, device=device)
                for i in range(first_row.size(1)):
                    value = first_row[batch_idx, i].item()
                    indices_in_data = (data.x[:, 0] == value).nonzero(as_tuple=True)[0]
                    if indices_in_data.size(0) > 0:
                        # If multiple matches are found, just take the first one
                        matching_indices[i] = indices_in_data[0]
                    else:
                        # If no match, set a flag for default behavior (using -1 as a flag here)
                        matching_indices[i] = -1

                # Gather the corresponding vectors from gat_output using the indices
                valid_indices_mask = matching_indices != -1
                matched_vectors = torch.zeros(first_row.size(1), gat_output.size(2), device=device)
                matched_vectors[valid_indices_mask] = gat_output[batch_idx, matching_indices[valid_indices_mask]]

                # Concatenate the vectors from gat_output to data_input
                concatenated_rows[batch_idx] = torch.cat((data_input[batch_idx], matched_vectors), dim=1)

            # Remove the first column (if needed) from the concatenated input
            lstm_input = concatenated_rows[:, :, 1:]
            # lstm_input = torch.cat((data_input, gat_output), dim=-1)
        elif model_used == "LSTM":
            lstm_input = data_input

        # Use the LSTM model to get the outputs
        act_output, time_output, timeR_output = self.lstm_model(lstm_input, task_type)

        return act_output, time_output, timeR_output


def initialize_models(num_node_features, dropout, graph_hidden_dim, graph_embedding_dim, num_edge_features,
                      num_layers, input_dim, lstm_hidden_dim, num_target_chars):
    # Initialize GAT model
    gat_model = GATModel(num_node_features, dropout, graph_hidden_dim, graph_embedding_dim, num_edge_features,
                         num_layers)

    # Initialize LSTM model
    lstm_model = LSTMModel(input_dim, lstm_hidden_dim, num_target_chars)

    # Initialize LSTMGATModel with the GAT model and LSTM model
    lstm_gat_model = LSTMGATModel(gat_model, lstm_model)

    # Move the model to the specified device
    lstm_gat_model = lstm_gat_model.to(device)
    return lstm_gat_model


def initialize_optimizers(lstm_gat_model, learning_rate_lstm, learning_rate_graph):
    # Extract parameters for LSTM and GAT models
    lstm_parameters = list(lstm_gat_model.lstm_model.shared_lstm.parameters()) + \
                      list(lstm_gat_model.lstm_model.act_lstm.parameters()) + \
                      list(lstm_gat_model.lstm_model.time_lstm.parameters()) + \
                      list(lstm_gat_model.lstm_model.timeR_lstm.parameters())
    gat_parameters = list(lstm_gat_model.gat_model.parameters())

    # Initialize optimizers
    optimizer_lstm = optim.NAdam(lstm_parameters, lr=learning_rate_lstm)
    optimizer_graph = optim.NAdam(gat_parameters, lr=learning_rate_graph)

    return optimizer_lstm, optimizer_graph


def train_model(lstm_gat_model, model_used, threshold, train_data_loaded, val_data_loader, num_epochs, optimizer_lstm,
                optimizer_graph,
                model_dir, patience=5, task_type='all'):
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.L1Loss()

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        lstm_gat_model.train()
        running_loss = 0.0

        for batch_idx, data in enumerate(train_data_loaded):
            optimizer_lstm.zero_grad()
            optimizer_graph.zero_grad()
            data = data.to(device)
            act_output, time_output, timeR_output = lstm_gat_model(data, model_used, task_type)
            loss = 0
            if task_type == "classification" or task_type == "next" or task_type == "all":
                classification_loss = classification_criterion(act_output, data.y_act)
                loss += classification_loss

            if task_type == "regression" or task_type == "next" or task_type == "all":
                regression_loss1 = regression_criterion(time_output, data.y_times[:, 0].unsqueeze(1))
                loss += regression_loss1
            if task_type == "regression" or task_type == "remaining" or task_type == "all":
                regression_loss2 = regression_criterion(timeR_output, data.y_times[:, 1].unsqueeze(1))
                loss += regression_loss2

            loss.backward()
            optimizer_lstm.step()
            optimizer_graph.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_data_loaded)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}', end='')

        with torch.no_grad():
            val_loss = 0.0
            for data in val_data_loader:
                data = data.to(device)
                act_output, time_output, timeR_output = lstm_gat_model(data, model_used, task_type)
                if task_type == "classification" or task_type == "next" or task_type == "all":
                    classification_loss = classification_criterion(act_output, data.y_act)
                    val_loss += classification_loss.item()
                if task_type == "regression" or task_type == "next" or task_type == "all":
                    regression_loss1 = regression_criterion(time_output, data.y_times[:, 0].unsqueeze(1))
                    val_loss += regression_loss1.item()
                if task_type == "regression" or task_type == "remaining" or task_type == "all":
                    regression_loss2 = regression_criterion(timeR_output, data.y_times[:, 1].unsqueeze(1))
                    val_loss += regression_loss2.item()

        epoch_val_loss = val_loss / len(val_data_loader)
        print(f', Val Loss: {epoch_val_loss:.4f}')

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the best model
            model_path = os.path.join(model_dir, f'best_model_{model_used}_{threshold}.pth')
            save_model(lstm_gat_model, model_path)
            print(f'Saved best model with validation loss: {best_val_loss:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    return lstm_gat_model


def save_model(model, model_path):
    # Extract GAT and LSTM models
    gat_model = model.gat_model
    lstm_model = model.lstm_model

    # Save the state dictionaries of both models
    torch.save({
        'gat_model_state_dict': gat_model.state_dict(),
        'lstm_model_state_dict': lstm_model.state_dict()
    }, model_path)


def load_model(model, model_path, device):
    # Extract GAT and LSTM models
    gat_model = model.gat_model
    lstm_model = model.lstm_model

    # Load the state dictionaries from the file
    checkpoint = torch.load(model_path)

    # Load state dicts into the respective models
    gat_model.load_state_dict(checkpoint['gat_model_state_dict'])
    lstm_model.load_state_dict(checkpoint['lstm_model_state_dict'])

    # Move models to the specified device
    gat_model.to(device)
    lstm_model.to(device)

    return model


def evaluate_model(lstm_gat_model, model_used, test_data_loaded, time_target_means, task_type):
    lstm_gat_model.eval()

    act_preds = []
    time_preds = []
    timeR_preds = []
    act_targets = []
    time_targets = []
    timeR_targets = []

    with torch.no_grad():
        for data in test_data_loaded:
            data = data.to(device)
            # Get outputs based on the task type
            act_output, time_output, timeR_output = lstm_gat_model(data, model_used, task_type)

            # Classification metrics for next activity prediction
            if task_type == "classification" or task_type == "next" or task_type == "all":
                act_preds.extend(torch.argmax(act_output, dim=-1).cpu().numpy())
                act_targets.extend(torch.argmax(data.y_act, dim=-1).cpu().numpy())

            # Regression metrics for event time prediction
            if task_type == "regression" or task_type == "next" or task_type == "all":
                time_preds.extend(time_output.cpu().numpy())
                time_targets.extend(data.y_times[:, 0].cpu().numpy())

            # Regression metrics for remaining time prediction
            if task_type == "regression" or task_type == "remaining" or task_type == "all":
                timeR_preds.extend(timeR_output.cpu().numpy())
                timeR_targets.extend(data.y_times[:, 1].cpu().numpy())
    metrics = {}
    # Compute classification metrics
    if task_type == "classification" or task_type == "next" or task_type == "all":
        accuracy = accuracy_score(act_targets, act_preds)
        precision = precision_score(act_targets, act_preds, average='weighted')
        recall = recall_score(act_targets, act_preds, average='weighted')
        f1 = f1_score(act_targets, act_preds, average='weighted')

        metrics.update({
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })
    # Compute regression metrics for event time prediction
    if task_type == "regression" or task_type == "next" or task_type == "all":
        mae_time = mean_absolute_error(time_targets, time_preds) * time_target_means[0]
        mse_time = mean_squared_error(time_targets, time_preds)
        rmse_time = np.sqrt(mse_time)

        metrics.update({
            "MAE Time": mae_time,
            "MSE Time": mse_time,
            "RMSE Time": rmse_time
        })

    # Compute regression metrics for remaining time prediction
    if task_type == "regression" or task_type == "remaining" or task_type == "all":
        mae_timeR = mean_absolute_error(timeR_targets, timeR_preds) * time_target_means[1]
        mse_timeR = mean_squared_error(timeR_targets, timeR_preds)
        rmse_timeR = np.sqrt(mse_timeR)

        metrics.update({
            "MAE TimeR": mae_timeR,
            "MSE TimeR": mse_timeR,
            "RMSE TimeR": rmse_timeR
        })

    return metrics


def save_results(metrics, results_dir, model_used, task_type, threshold):
    """
    Saves the evaluation metrics to a file.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
        results_dir (str): The path to the file where the results should be saved.
        model_used (str): Specifies the model used
        task_type (str): The type of task ('classification', 'regression', 'remaining', or 'all').
    """
    file_path = os.path.join(results_dir, f"{model_used}-{threshold}.csv")
    with open(file_path, 'a') as f:
        f.write(f"Results for task type: {task_type}\n")

        # Save classification metrics
        if task_type == "classification" or task_type == "next" or task_type == "all":
            f.write("Classification Metrics (Next Activity Prediction):\n")
            f.write(f"  Accuracy: {metrics.get('Accuracy', 'N/A')}\n")
            f.write(f"  Precision: {metrics.get('Precision', 'N/A')}\n")
            f.write(f"  Recall: {metrics.get('Recall', 'N/A')}\n")
            f.write(f"  F1-score: {metrics.get('F1-score', 'N/A')}\n")
            f.write("\n")

        # Save event time prediction metrics
        if task_type == "regression" or task_type == "next" or task_type == "all":
            f.write("Regression Metrics (Event Time Prediction):\n")
            f.write(f"  MAE Time: {metrics.get('MAE Time', 'N/A')}\n")
            f.write(f"  MSE Time: {metrics.get('MSE Time', 'N/A')}\n")
            f.write(f"  RMSE Time: {metrics.get('RMSE Time', 'N/A')}\n")
            f.write("\n")

        # Save remaining time prediction metrics
        if task_type == "remaining" or task_type == "all":
            f.write("Regression Metrics (Remaining Time Prediction):\n")
            f.write(f"  MAE TimeR: {metrics.get('MAE TimeR', 'N/A')}\n")
            f.write(f"  MSE TimeR: {metrics.get('MSE TimeR', 'N/A')}\n")
            f.write(f"  RMSE TimeR: {metrics.get('RMSE TimeR', 'N/A')}\n")
            f.write("\n")

        f.write("-" * 50 + "\n")
