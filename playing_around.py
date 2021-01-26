import random

def reservoir_sample(arr,k):
    reservoir = []
    for i in range (len(arr)):
        if len(reservoir) < k:
            reservoir.append(arr[i])
        else:
            j = int(random.uniform(0,i))
            if j < k:
                reservoir[j] = arr[i]
    return reservoir

print (reservoir_sample([1,2,3,4,5,6,7,8,9,10,1,12,13,14,15],5))
