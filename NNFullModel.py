import torch

#-----------------------------------------------------------------------------------------------------------# 
# Class that can run a NN model for a single batch of data and compute the accuracy and loss.

class NNFullModel:
    
    def __init__(self, NN_model, loss_function, optimizer, accuracy_calculator):
        
        self.model = NN_model
        self.loss_function= loss_function
        self.optimizer = optimizer
        self.accuracy_calculator = accuracy_calculator
        
        self.runningaccuracy = 0
        self.runningloss = 0
    
    
    # Runs the model for a batch of data and trains it if Train = TRUE.
    def run_model(self, batch_of_data, labels, train):

        # Reset Gradient
        if train == True: self.optimizer.zero_grad()

        # Get Predictions
        self.outputs = self.model(batch_of_data)

        # Computing Loss
        loss = self.loss_function(input = self.outputs.squeeze(), target=labels.float())
        
        # Compute Gradients
        if train == True: loss.backward()
        
        # Gradient Step
        torch.save(self.model.state_dict(), './rcnt_model_backstep_param')
        if train == True: self.optimizer.step()
        
        # Record Accuracy and Loss
        self.runningloss += float(loss)
        self.runningaccuracy += self.accuracy_calculator(self.outputs, labels)
        
        return True
    
    # Resets the running accuracy and validation counts. Should be done after every batch of data.
    def reset_running_counts(self):
        self.runningaccuracy = 0
        self.runningloss = 0
        
        return True
    
    # Returns the tuple of [loss, accuracy] values for a batch of batch_size
    def get_results(self, number_of_batches):
        return self.runningloss / number_of_batches, self.runningaccuracy / number_of_batches
    
    # Returns the outputs of the last modelrun that was called.
    def get_last_outputs(self):
        return self.outputs
    
    # Steps back on the model's parameters (can only setback once)
    def backstep(self):
        (self.model).load_state_dict(torch.load('./rcnt_model_backstep_param'))
        
    # Change learning rate of all parameters
    def change_learning_rate(self, learning_rate):
        for i in self.optimizer.param_groups:
            i['lr'] = learning_rate

    
    
        
        