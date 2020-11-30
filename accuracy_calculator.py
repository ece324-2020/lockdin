import torch

#-----------------------------------------------------------------------------------------------------------#
def accuracy_calculator(outputs, labels):
        
    accuracy = 0

    for i in range(len(labels)):
        if ((outputs[i].item() >= 0.5) and (labels[i].item() == 1)) or ((outputs[i].item() < 0.5) and (labels[i].item() == 0)):
            accuracy += 1
        
                
    return accuracy/len(labels)

#-----------------------------------------------------------------------------------------------------------#
def decision_function(outputs, labels):
    
    predictions = []
    
    for i in range(len(labels)):
        if (outputs[i].item() >= 0.5):
            
            predictions.append(1)
            
        else:
            
            predictions.append(0)
        
    return predictions
