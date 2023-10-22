import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

import matplotlib as mpl
import matplotlib.pyplot as plt

path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/logs_final/final_with_2023_10_20-00_53_37/089/execution0/train/events.out.tfevents.1697834025.hybris.local.81912.268.v2"

# Loading too much data is slow...
tf_size_guidance = {
    'compressedHistograms': 10,
    'images': 0,
    'scalars': 10 * 10**6,
    'histograms': 1
}

event_acc = ea.EventAccumulator(path, tf_size_guidance)
event_acc.Reload()

# Show all tags in the log file
#print(event_acc.Tags())

import ipdb; ipdb.set_trace()
training_accuracies = event_acc.Scalars("execution0/train")
validation_accuracies =  event_acc.Scalars("test/Episode_Length_19")

steps = len(training_accuracies)
print(steps)
x = np.arange(steps)
y = np.zeros([steps, 2])

for i in range(steps):
    y[i, 0] = training_accuracies[i][2] # value
    y[i, 1] = validation_accuracies[i][2]

plt.plot(x, y[:,0], label='training accuracy')
plt.plot(x, y[:,1], label='validation accuracy')

plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Training Progress")
plt.legend(loc='upper right', frameon=True)
plt.show()
