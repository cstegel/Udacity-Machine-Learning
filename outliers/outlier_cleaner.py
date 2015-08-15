#!/usr/bin/python
from operator import itemgetter

def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error)
    """
    cleaned_data = []
    data = []


    ### your code goes here
    #calculate errors
    for i in range(len(ages)):
        # print(ages[i][0])
        error = net_worths[i][0] - predictions[i][0]
        data.append({'age': ages[i][0], 'net_worth': net_worths[i][0], 'error': error})

    data = sorted(data, key=itemgetter('error'), reverse=True)
    cutoff = int(round(len(data) * 0.9))
    for i in range(cutoff):
      cleaned_data.append((data[i]['age'], data[i]['net_worth'], data[i]['error']))
    return cleaned_data

