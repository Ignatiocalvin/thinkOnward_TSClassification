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
class CustomLoss(nn.Module):
    def __init__(self, column_groups):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.column_groups = column_groups  # Dictionary mapping attribute prefixes to column indices
    
    def forward(self, categorical_pred, categorical_true):
        # Compute numerical loss (assuming the first few columns are numerical)
        loss_numerical = self.mse_loss(categorical_pred[:, :5], categorical_true[:, :5])

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

