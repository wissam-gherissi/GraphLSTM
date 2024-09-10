import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size, target_size, num_edge_features, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_edge_features = num_edge_features
        self.num_layers = num_layers
        self.add_self_loops = True

        # Create a list of GATConv layers
        self.convs = torch.nn.ModuleList(
            [GATConv(self.num_features, self.hidden_size, edge_dim=self.num_edge_features, add_self_loops=self.add_self_loops)] + [
                GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features, add_self_loops=self.add_self_loops)
                for _ in range(self.num_layers - 2)] +
            [GATConv(self.hidden_size, self.target_size, edge_dim=self.num_edge_features, add_self_loops=self.add_self_loops)])

        # Linear layer for final output
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data, batch_size):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply multiple GATConv layers
        for conv in self.convs:
            x = F.gelu(conv(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)

        # Apply linear layer for final output
        # x = self.linear(x)

        # Reshape output
        x = torch.reshape(x, (batch_size, x.shape[0] // batch_size, self.target_size))
        return x


class LSTMGATModel(torch.nn.Module):
    def __init__(self, gat_model, input_dim, lstm_hidden_dim, num_target_chars):
        super(LSTMGATModel, self).__init__()
        self.gat_model = gat_model
        self.bn_gnn = nn.BatchNorm1d(self.gat_model.target_size)
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

    def forward(self, data, model_used):
        # Pass the graph through the GAT model
        data_input = data.lstm_input

        if model_used == "graph":
            gat_output = self.gat_model(data, data_input.shape[0])
            gat_output = self.bn_gnn(gat_output.permute(0, 2, 1)).permute(0, 2, 1)

            device = data_input.device

            # Get the first row from each batch (shape: [batch_size, sequence_length])
            first_row = data_input[:, :, 0]

            # Create a tensor to hold the concatenated output (shape: [batch_size, sequence_length, input_size + gat_output_size])
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
            #lstm_input = torch.cat((data_input, gat_output), dim=-1)
        elif model_used == "LSTM":
            lstm_input = data_input
        # Pass the concatenated input through the shared LSTM layer
        shared_lstm_output, _ = self.shared_lstm(lstm_input)

        # Apply batch normalization after the shared LSTM layer
        shared_lstm_output = self.bn_shared(shared_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)

        # Pass the output of the shared LSTM layer through specialized LSTM layers
        act_lstm_output, _ = self.act_lstm(shared_lstm_output)
        time_lstm_output, _ = self.time_lstm(shared_lstm_output)
        timeR_lstm_output, _ = self.timeR_lstm(shared_lstm_output)

        # Apply batch normalization after each specialized LSTM layer
        act_lstm_output = self.bn_act(act_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)
        time_lstm_output = self.bn_time(time_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)
        timeR_lstm_output = self.bn_timeR(timeR_lstm_output.permute(0, 2, 1)).permute(0, 2, 1)

        # Get the output of the last timestep of each specialized LSTM layer
        act_output_last = act_lstm_output[:, -1, :]
        time_output_last = time_lstm_output[:, -1, :]
        timeR_output_last = timeR_lstm_output[:, -1, :]

        # Forward pass through output layers
        act_output = F.softmax(self.fc_act(act_output_last), dim=1)
        time_output = self.fc_time(time_output_last)
        timeR_output = self.fc_timeR(timeR_output_last)

        return act_output, time_output, timeR_output

