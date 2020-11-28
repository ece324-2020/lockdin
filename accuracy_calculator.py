import torch

#-----------------------------------------------------------------------------------------------------------#
def accuracy_calculator(ouputs, labels):
        
    predictions = torch.max(ouputs,1)
    
    accuracy = 0

    for i in range(len(predictions.indices)):
        if (predictions.indices[i] == (labels[i]==1).nonzero()):
            accuracy += 1/len(labels)
                
    return accuracy    