import torch
import numpy as np

data = torch.load('kddcup99_processed_x_stoc_8.npy')
target = torch.load('kddcup99_processed_y_stoc_8.npy')

target = np.reshape(target, (-1, 1))

print(data.shape, target.shape)

data_target = np.hstack((data, target))

positive_samples = data_target[data_target[:, -1] == 1]
negative_samples = data_target[data_target[:, -1] == -1]

print(positive_samples.shape)
print(negative_samples.shape)

ret_data_target = np.zeros(data_target.shape)

positive_sample_size = len(positive_samples)
sample_size = len(target)
single_learner_size = int(sample_size / 8)

for i in range(positive_sample_size):
    selected_learner = int(i / single_learner_size)
    learner_sample = i - selected_learner * single_learner_size
    ret_index = learner_sample * 8 + selected_learner
    # print(ret_index)
    ret_data_target[ret_index, :] = positive_samples[i, :]

for i in range(positive_sample_size, sample_size):
    selected_learner = int(i / single_learner_size)
    learner_sample = i - selected_learner * single_learner_size
    ret_index = learner_sample * 8 + selected_learner
    ret_data_target[ret_index, :] = negative_samples[i-positive_sample_size, :]

ret_data = ret_data_target[:, :-1]
ret_target = ret_data_target[:, -1]

torch.save(ret_data, 'kddcup99_processed_x_8.npy')
torch.save(ret_target, 'kddcup99_processed_y_8.npy')
