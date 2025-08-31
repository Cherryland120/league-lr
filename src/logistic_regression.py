# League of Legends Win Prediction using Logistic Regression
# This script implements a PyTorch-based logistic regression model to predict match outcomes
# in League of Legends based on game statistics

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

# Load the League of Legends dataset
data = r"../data/league_of_legends_data_large.csv"
df = pd.read_csv(data)

# Separate features (X) and target variable (y)
# X contains all game statistics, y contains win/loss outcomes
X = df.drop('win', axis=1)  # Features: all columns except 'win'
y = df['win']               # Target: win/loss binary outcome

# Initialize StandardScaler for feature normalization
# This ensures all features are on the same scale for better model performance
scaler = StandardScaler()

# Split data into training and testing sets (80/20 split)
# random_state=42 ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using fit_transform on training data and transform on test data
# This prevents data leakage by only using training data statistics
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled numpy arrays to PyTorch tensors for model training
# float32 is used for computational efficiency
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Convert target variables to PyTorch tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class LogisticRegressionModel(torch.nn.Module):
    """
    PyTorch implementation of Logistic Regression model.
    
    This model uses a single linear layer followed by sigmoid activation
    to perform binary classification for win/loss prediction.
    """
    
    def __init__(self, input_dim):
        """
        Initialize the logistic regression model.
        
        Args:
            input_dim (int): Number of input features
        """
        super(LogisticRegressionModel, self).__init__()
        # Single linear layer that maps input features to single output
        self.linear = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Sigmoid-activated output (probability between 0 and 1)
        """
        # Apply linear transformation followed by sigmoid activation
        return torch.sigmoid(self.linear(x))

# =============================================================================
# MODEL INITIALIZATION AND SETUP
# =============================================================================

# Get the number of input features from the training data
input_dim = X_train_tensor.shape[1]
print(f"Input dimension (number of features): {input_dim}")

# Initialize the model with the correct input dimension
model = LogisticRegressionModel(input_dim)
print("\nModel structure:")
print(model)

# Define loss function - Binary Cross Entropy Loss for binary classification
criterion = torch.nn.BCELoss()

# Define optimizer - Stochastic Gradient Descent with learning rate 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("\nLoss function initialized:", criterion)
print("Optimizer initialized:", optimizer)

# =============================================================================
# MODEL TRAINING
# =============================================================================

# Set the number of training epochs
num_epochs = 1000

print("\nStarting model training...")

# Reshape the target tensor to match the output shape [batch_size, 1]
y_train_tensor = y_train_tensor.view(-1, 1)

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    
    # Zero out gradients from previous iteration
    optimizer.zero_grad()
    
    # Forward pass: compute model predictions
    outputs = model(X_train_tensor)
    
    # Calculate loss between predictions and actual targets
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update model parameters using computed gradients
    optimizer.step()
    
    # Print loss every 100 epochs for monitoring training progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\nTraining complete.")

# =============================================================================
# MODEL EVALUATION - ACCURACY CALCULATION
# =============================================================================

# Evaluate model performance without computing gradients (inference mode)
with torch.no_grad():
    # Get predictions on the training data
    train_outputs = model(X_train_tensor)
    # Convert probabilities to binary predictions using 0.5 threshold
    train_predicted = (train_outputs >= 0.5).float()
    
    # Get predictions on the test data
    test_outputs = model(X_test_tensor)
    # Convert probabilities to binary predictions using 0.5 threshold
    test_predicted = (test_outputs >= 0.5).float()

    # Calculate training accuracy
    train_correct = (train_predicted == y_train_tensor).sum().item()
    train_total = y_train_tensor.size(0)
    train_accuracy = train_correct / train_total * 100

    # Calculate test accuracy
    test_correct = (test_predicted == y_test_tensor).sum().item()
    test_total = y_test_tensor.size(0)
    test_accuracy = test_correct / test_total * 100

    print(f'\nTraining Accuracy: {train_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# =============================================================================
# COMPREHENSIVE MODEL EVALUATION - METRICS AND VISUALIZATIONS
# =============================================================================

# Generate detailed evaluation metrics and visualizations
with torch.no_grad():
    # Get predictions on the training data
    train_outputs = model(X_train_tensor)
    train_predicted = (train_outputs >= 0.5).float()
    
    # Get predictions on the test data
    test_outputs = model(X_test_tensor)
    test_predicted = (test_outputs >= 0.5).float()

    # Convert PyTorch tensors to numpy arrays for scikit-learn compatibility
    y_test_np = y_test_tensor.numpy()
    test_predicted_np = test_predicted.numpy()
    test_outputs_np = test_outputs.numpy()

    # 1. Generate and print detailed classification report
    # Includes precision, recall, F1-score for each class
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test_np, test_predicted_np))

    # 2. Create and visualize confusion matrix
    # Shows true positives, false positives, true negatives, false negatives
    cm = confusion_matrix(y_test_np, test_predicted_np)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    # Save confusion matrix plot
    plt.savefig("../results/confusion_matrix.png")

    # 3. Generate and plot ROC (Receiver Operating Characteristic) curve
    # ROC curve shows the trade-off between true positive rate and false positive rate
    fpr, tpr, thresholds = roc_curve(y_test_np, test_outputs_np)
    roc_auc = auc(fpr, tpr)  # Calculate Area Under Curve (AUC)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    # Save ROC curve plot
    plt.savefig("../results/roc_curve.png")
    
    # Recalculate and display accuracy metrics
    train_correct = (train_predicted == y_train_tensor).sum().item()
    train_total = y_train_tensor.size(0)
    train_accuracy = train_correct / train_total * 100

    test_correct = (test_predicted == y_test_tensor).sum().item()
    test_total = y_test_tensor.size(0)
    test_accuracy = test_correct / test_total * 100

    print(f'\nTraining Accuracy: {train_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# =============================================================================
# MODEL PERSISTENCE - SAVE AND LOAD
# =============================================================================

# Save the trained model for future use
model_path = 'logistic_regression_model.pth'

# Save only the model's parameters (state dictionary) - more efficient than saving entire model
print(f"\nSaving model parameters to '{model_path}'...")
torch.save(model.state_dict(), model_path)
print("Model saved successfully.")

# Demonstrate model loading process
print("\nCreating a new model instance and loading saved parameters...")

# Create a new, untrained instance of the model with same architecture
new_model = LogisticRegressionModel(input_dim)

# Load the saved parameters into the new model instance
new_model.load_state_dict(torch.load(model_path))

# Set model to evaluation mode for inference
new_model.eval()

# Verify that the loaded model performs identically to the original
with torch.no_grad():
    # Get predictions using the loaded model
    loaded_model_test_outputs = new_model(X_test_tensor)
    
    # Convert probabilities to binary predictions
    loaded_model_test_predicted = (loaded_model_test_outputs >= 0.5).float()
    
    # Calculate accuracy of the loaded model
    loaded_model_test_correct = (loaded_model_test_predicted == y_test_tensor).sum().item()
    loaded_model_test_total = y_test_tensor.size(0)
    loaded_model_test_accuracy = loaded_model_test_correct / loaded_model_test_total * 100

    print(f"Loaded Model's Test Accuracy: {loaded_model_test_accuracy:.2f}%")
    
    # Generate classification report for loaded model to confirm identical performance
    print("\n--- Classification Report (Loaded Model Test Set) ---")
    print(classification_report(y_test_np, loaded_model_test_predicted.numpy()))

# =============================================================================
# HYPERPARAMETER TUNING - LEARNING RATE OPTIMIZATION
# =============================================================================

# Test different learning rates to find the optimal one
num_epochs = 100  # Reduced epochs for faster hyperparameter search
learning_rates = [0.01, 0.05, 0.1]  # Different learning rates to test
best_accuracy = 0
best_lr = 0
results = {}

print("\n=== HYPERPARAMETER TUNING: LEARNING RATE OPTIMIZATION ===")

# Test each learning rate
for lr in learning_rates:
    print(f"\n--- Training with Learning Rate: {lr} ---")
    
    # Reinitialize model and optimizer for each learning rate test
    # This ensures fair comparison by starting from scratch each time
    model = LogisticRegressionModel(input_dim)
    criterion = torch.nn.BCELoss()
    # Add L2 regularization (weight_decay) to prevent overfitting
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    # Training loop for current learning rate
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
    print(f'Training complete for lr={lr}.')

    # Evaluate the model trained with current learning rate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        # Convert probabilities to binary predictions
        test_predicted = (test_outputs >= 0.5).float()
        
        # Calculate test accuracy
        test_correct = (test_predicted == y_test_tensor).sum().item()
        test_total = y_test_tensor.size(0)
        test_accuracy = test_correct / test_total * 100    

    # Store results for this learning rate
    results[lr] = test_accuracy
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Track the best performing learning rate
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_lr = lr

# Display hyperparameter tuning results summary
print("\n--- Hyperparameter Tuning Results Summary ---")
for lr, acc in results.items():
    print(f"Learning Rate: {lr}, Test Accuracy: {acc:.2f}%")

print(f"\nThe best learning rate found is: {best_lr} with a test accuracy of {best_accuracy:.2f}%")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

# Extract model weights to understand feature importance
# The weights represent how much each feature contributes to the prediction

# Extract weights from the linear layer and flatten to 1D array
weights = model.linear.weight.data.numpy().flatten()

# Get feature names from the original dataset
features = X.columns

# Create a DataFrame to pair feature names with their corresponding weights
feature_importance = pd.DataFrame({'Feature': features, 'Importance': weights})

# Sort features by importance (weight magnitude) in descending order
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display feature importance table
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
print(feature_importance)

# Create visualization of feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance (Weight Value)')
plt.title('Feature Importance - League of Legends Win Prediction Model')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()

# Save feature importance plot
plt.savefig("../results/feature_importance.png")

print("\n=== ANALYSIS COMPLETE ===")
print("All results and visualizations have been saved to the '../results/' directory.")
