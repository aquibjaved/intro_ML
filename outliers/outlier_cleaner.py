#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    #calculate squared errors and calculate how much data 
    #shoulb be excluded
    erros=(predictions - net_worths_train) ** 2
    i_exclude=int(len(net_worths_train)*0.1)
    #transform the array and find the 10 smallest errors
    #in the biggest ones
    test = erros.T[0]
    temp = numpy.argpartition(-test, i_exclude)
    result_args = temp[:i_exclude]
    #create the output, filtering out the outliers
    temp = numpy.partition(-test, i_exclude)
    minimum_biggest = min(-temp[:i_exclude])
    for e,a,n in zip(erros, ages, net_worths):
        if e< minimum_biggest:
            cleaned_data.append((a[0],n[0],e[0]))
    
    return cleaned_data

