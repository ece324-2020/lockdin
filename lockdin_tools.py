import numpy as np
import matplotlib.pyplot as plt
import time

#-----------------------------------------------------------------------------------------------------------# 
# Different training loops, data representaions and more for the lockdin project.
class lockdin_tools:
    
#-----------------------------------------------------------------------------------------------------------#      
    def __init__(self, full_model):
        
        self.model = full_model
        
#-----------------------------------------------------------------------------------------------------------#
       
    def overfit_training(self):
        pass
    
#-----------------------------------------------------------------------------------------------------------# 
# A training loop that utilizes training, validaiton and test data sets.
    def regular_training(self, epochs, train_data, valid_data, test_data, sample_rate):
    
        # Create containers for data
        self.train_loss = []
        self.train_acc = []
        self.validation_loss = []
        self.validation_acc = []
        self.best_validation_acc = [0,0]
        
        start = time.time()        
        # Run the training
        for epoch in range(epochs):
            
            # Running Training
            self.batch_train_loop(train_data, True)

            # Record Training Data
            self.train_loss.append(self.loss)
            self.train_acc.append(self.acc)
            
            print('Epoch:', epoch , '||| Training Loss:', self.loss , '||| Training Accuracy:', self.acc)
            
            # Validation Loop 
            if epoch % sample_rate == 0:
                
                self.batch_train_loop(valid_data, False)

                # Record Training Data
                self.validation_loss.append(self.loss)
                self.validation_acc.append(self.acc)
                
                # Update best validation accuracy
                if (self.acc > self.best_validation_acc[0]): 
                    self.best_validation_acc = [self.acc, epoch]
                else:
                    self.model.backstep()
                
                print('Epoch:', epoch , '||| Validation Loss:', self.loss , '||| Validation Accuracy:', self.acc)
        end = time.time()
        
        # Time taken in seconds
        self.total_time = end - start
        
        # Find the test loss and accuracy
        self.batch_train_loop(test_data, False)
        self.test_loss = self.loss
        self.test_acc = self.acc
        
#-----------------------------------------------------------------------------------------------------------#      
# Training loop that fully loops over one batch of data 
    def batch_train_loop(self, data, train):
        
        (self.model).reset_running_counts()
        
        for (i, batch) in enumerate(data):
                images, labels = batch
                
                # Run the model and train
                (self.model).run_model(images, labels, train)
        
        # Get average batch results
        self.loss, self.acc = (self.model).get_results(i)
    
#-----------------------------------------------------------------------------------------------------------#    
# Displays the results of the last training loop that was run. 
    def display_results(self):
        
        # Creating the Epochs
        if ( len(self.train_acc) > 0 ): 
            
            X = np.arange(0,len(self.train_acc),1)
            
        if ( len(self.validation_acc) > 0 ): 
            
            Y = np.arange(0,len(self.train_acc),len(self.train_acc)/len(self.validation_acc))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout(pad=3.0)
    
        #Plot Loss
        if ( len(self.train_loss) > 0 ): 
            
            ax1.plot(X, self.train_loss, 'r', label='Training')
            
        if ( len(self.validation_loss) > 0 ): 
            
            ax1.plot(Y, self.validation_loss, 'b', label='Validation')
            
        ax1.set_title('Loss vs Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
    
        #Plot Accuracy
        if ( len(self.train_acc) > 0):
            
            ax2.plot(X, self.train_acc, 'r', label='Training')
        
        if ( len(self.validation_acc) > 0 ): 
            
            ax2.plot(Y, self.validation_acc , 'b', label='Validation')
        
        ax2.set_title('Accuracy vs Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        # Create legend
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.show()
        
        if ( len(self.train_acc) > 0):
            
            print('Final Training Accuracy      :', self.train_acc[len(self.train_acc)-1]*100)
            print('Final Training Loss          :', self.train_loss[len(self.train_loss)-1])
        
        if ( len(self.validation_acc) > 0 ): 
            print('Best Validation Accuracy     :', self.best_validation_acc[0]*100, 'achieved at epoch', self.best_validation_acc[1])
            print('Final Validation Accuracy    :', self.validation_acc[len(self.validation_acc)-1]*100)
            print('Final Validation Loss        :', self.validation_loss[len(self.validation_loss)-1])
        
        if (self.test_acc > 0):  
                                                                
            print('Final Test Accuracy          :', self.test_acc*100)
            print('Final Test Loss              :', self.test_loss)
            
        if (self.total_time > 0):
            
            print('Time to Train                :', self.total_time)
        
    