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
            # Concatenate GAT output with LSTM input
            first_row = data_input[0, :, 0]
            concatenated_rows = []
            for i in range(first_row.size(0)):
                value = first_row[i].item()

                indices_in_data = (data.x[:, 0] == value).nonzero(as_tuple=True)[0]

                if indices_in_data.size(0) > 0:
                    # Handle the case where multiple matches are found
                    if indices_in_data.size(0) > 1:
                        # If multiple matches are found, handle as needed.
                        # For now, let's just take the first match (you can change this behavior)
                        index_in_data = indices_in_data[0].item()
                    else:
                        index_in_data = indices_in_data.item()

                    # Get the corresponding vector from gat_output
                    vector_to_concat = gat_output[0, index_in_data, :]
                else:
                    # Default vector if value is not found
                    vector_to_concat = torch.zeros(gat_output.size(2))

                # Concatenate the vector from gat_output to the corresponding row of data_input
                vector_to_concat = vector_to_concat.to(device)
                concatenated_row = torch.cat((data_input[0, i, :], vector_to_concat), dim=0)
                concatenated_rows.append(concatenated_row)

            lstm_input = torch.stack(concatenated_rows).unsqueeze(0)
            lstm_input = lstm_input[:, 1:, :]
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

