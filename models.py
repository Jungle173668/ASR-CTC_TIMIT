import torch.nn as nn

# class BiLSTM(nn.Module):

#     def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
#         super().__init__()

#         self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)

#         # Define a Dropout layer after LSTM for regularization
#         self.dropout = nn.Dropout(dropout_prob)

#         self.proj = nn.Linear(hidden_dims * 2, out_dims)

#     def forward(self, feat):
#         hidden, _ = self.lstm(feat)

#         # # Apply dropout on LSTM outputs
#         # hidden = self.dropout(hidden)

#         output = self.proj(hidden)
#         return output


class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_prob):
        super().__init__()

        # Define LSTM with dropout between layers if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=in_dims,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0.0  # Apply dropout only between LSTM layers
        )

        # Define a Dropout layer for the output of the LSTM
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer (projection layer)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)


    def forward(self, feat):
        # Pass through LSTM
        hidden, _ = self.lstm(feat)

        # Apply dropout on LSTM outputs (only if training mode is enabled)
        hidden = self.dropout(hidden)

        # Pass through projection layer
        output = self.proj(hidden)

        return output



'''
with multi feed forward layers
'''
# class BiLSTM(nn.Module):

#     def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_prob):
#         super().__init__()

#         # Define LSTM with dropout between layers if num_layers > 1
#         self.lstm = nn.LSTM(
#             input_size=in_dims,
#             hidden_size=hidden_dims,
#             num_layers=num_layers,
#             bidirectional=True,
#             dropout=dropout_prob if num_layers > 1 else 0.0  # Apply dropout only between LSTM layers
#         )

#         # Define a Dropout layer for the output of the LSTM
#         self.dropout = nn.Dropout(dropout_prob)

#         # Add additional feed-forward layers
#         self.feedforward = nn.Sequential(
#             nn.Linear(hidden_dims * 2, hidden_dims),  # first layer processing output
#             nn.ReLU(), 
#             nn.Dropout(dropout_prob),  # Dropout
#             nn.Linear(hidden_dims, out_dims) # second layer
#         )


#     def forward(self, feat):
#         # Pass through LSTM
#         hidden, _ = self.lstm(feat)

#         # Apply dropout on LSTM outputs (only if training mode is enabled)
#         hidden = self.dropout(hidden)

#         output = self.feedforward(hidden)

#         return output

'''
Uni-directional LSTM
'''
# class BiLSTM(nn.Module):

#     def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_prob):
#         super().__init__()

#         # Define LSTM with dropout between layers if num_layers > 1
#         self.lstm = nn.LSTM(
#             input_size=in_dims,
#             hidden_size=hidden_dims,
#             num_layers=num_layers,
#             bidirectional=False,
#             dropout=dropout_prob if num_layers > 1 else 0.0  # Apply dropout only between LSTM layers
#         )

#         # Define a Dropout layer for the output of the LSTM
#         self.dropout = nn.Dropout(dropout_prob)

#         # Fully connected layer (projection layer)
#         self.proj = nn.Linear(hidden_dims, out_dims)


#     def forward(self, feat):
#         # Pass through LSTM
#         hidden, _ = self.lstm(feat)

#         # Apply dropout on LSTM outputs (only if training mode is enabled)
#         hidden = self.dropout(hidden)

#         # Pass through projection layer
#         output = self.proj(hidden)

#         return output
