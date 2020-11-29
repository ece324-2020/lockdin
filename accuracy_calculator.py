import torch

#-----------------------------------------------------------------------------------------------------------#
def accuracy_calculator(outputs, labels):
        
    accuracy = 0

    for i in range(len(labels)):
        if ((outputs[i].item() >= 0.5) and (labels[i].item() == 1)) or ((outputs[i].item() < 0.5) and (labels[i].item() == 0)):
            accuracy += 1
        
                
    return accuracy/len(labels)  