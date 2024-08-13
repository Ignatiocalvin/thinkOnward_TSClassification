from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define custom dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, y_categorical):
        self.data = data
        self.y_categorical = y_categorical
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx], 
                self.y_categorical[idx])
    
# Define custom loss function
# class CustomLoss(nn.Module):
#     def __init__(self, column_groups):
#         super(CustomLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.column_groups = column_groups  # Dictionary mapping attribute prefixes to column indices
    
#     def forward(self, categorical_pred, categorical_true):
#         # Compute numerical loss (assuming the first few columns are numerical)
#         loss_numerical = self.mse_loss(categorical_pred[:, :5], categorical_true[:, :5])

#         # Initialize categorical loss
#         loss_categorical = 0.0
        
#         # For each attribute group, compute the cross-entropy loss
#         for attr, indices in self.column_groups.items():
#             # Extract logits for the current attribute
#             logits = categorical_pred[:, indices]
            
#             # Extract the true labels for the current attribute
#             # Convert one-hot encoding to class indices
#             true_labels = torch.argmax(categorical_true[:, indices], dim=1)
            
#             # Compute cross-entropy loss for the current attribute
#             loss_categorical += F.cross_entropy(logits, true_labels)

#         total_loss = loss_numerical + loss_categorical
#         return total_loss, loss_numerical, loss_categorical
# 
# 

class CustomLoss(nn.Module):
    def __init__(self, column_groups, valid_labels):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.column_groups = column_groups  # Dictionary mapping attribute prefixes to column indices
        self.valid_labels = {k: torch.tensor(v).float() for k, v in valid_labels.items()}  # Convert valid labels to tensor
        self.end_numerical = int(min([min(v) for v in self.column_groups.values()]))

    def custom_closest_loss(self, predicted, true, valid_labels):
        valid_labels = valid_labels.to(predicted.device)
        # Expand dimensions to allow broadcasting (batch_size, num_valid_labels)
        predicted_expanded = predicted.unsqueeze(1)  # (batch_size, 1)
        valid_labels_expanded = valid_labels.unsqueeze(0)  # (1, num_valid_labels)
        
        # Calculate the absolute differences
        distances = torch.abs(predicted_expanded - valid_labels_expanded)  # (batch_size, num_valid_labels)
        
        # Find the closest valid label (index of the smallest distance)
        min_distances, min_indices = torch.min(distances, dim=1)  # min_distances: (batch_size,), min_indices: (batch_size,)
        
        # Get the corresponding closest labels
        closest_labels = valid_labels[min_indices]  # (batch_size,)
        
        # Compute the loss only where the closest label is not equal to the true label
        mask = closest_labels != true
        loss = torch.abs(predicted[mask] - true[mask]).mean()  # Mean absolute error over all incorrect predictions
        
        return loss
    
    def forward(self, categorical_pred, categorical_true):

        loss_numerical = 0.0
        i = 0
        # Loop over each attribute group
        for attr, labels in self.valid_labels.items():# TODO: maybe speed it up by inserting all labels as a matrix and performing matrix operations
            # Calculate the loss for this group using the valid labels
            loss_numerical += self.custom_closest_loss(categorical_pred[:, i], categorical_true[:, i], labels)
            i += 1


        # Compute numerical loss (assuming the first few columns are numerical)
        # loss_numerical = self.custom_closest_loss(categorical_pred[:, :self.end_numerical], categorical_true[:, :self.end_numerical])
        # loss_numerical = self.mse_loss(categorical_pred[:, :self.end_numerical], categorical_true[:, :self.end_numerical])

        # Initialize categorical loss
        loss_categorical = 0.0
        
        # For each attribute group, compute the cross-entropy loss
        for attr, indices in self.column_groups.items():
            # Extract logits for the current attribute
            logits = categorical_pred[:, indices]
            
            # Extract the true labels for the current attribute
            # Convert one-hot encoding to class indices
            true_labels = torch.argmax(categorical_true[:, indices], dim=1)
            
            # Compute cross-entropy loss for the current attribute
            loss_categorical += F.cross_entropy(logits, true_labels)

        total_loss = loss_numerical + loss_categorical
        return total_loss, loss_numerical, loss_categorical    

# Define custom LSTM model
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_categorical, num_lstm_layers=2):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, bidirectional=True, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size * num_lstm_layers, hidden_size)
        self.categorical_head = nn.Linear(hidden_size, num_classes_categorical)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        lstm_out = lstm_out[:, -1] 
        lstm_out = F.relu(self.hidden_layer(lstm_out))
        categorical_pred = self.categorical_head(lstm_out)  
        return categorical_pred
    
    def predict(self, x, association_dict): # return One hot encoded predictions
        predictions = self.forward(x)
        for attr, indices in association_dict.items():
            logits = predictions[:, indices]
            argm = torch.argmax(logits, dim=1)
            num_classes = len(indices)
            predictions[:, indices] = torch.nn.functional.one_hot(argm, num_classes=num_classes).to(torch.float32)
        return predictions
    
    def predict_prob(self, x, association_dict): # return probability predictions
        predictions = self.forward(x)
        for attr, indices in association_dict.items():
            logits = predictions[:, indices]
            predictions[:, indices] = F.softmax(logits, dim=1)
        return predictions

