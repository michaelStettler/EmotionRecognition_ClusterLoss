import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# prediction = np.load('../../models/RBF/predictions/predictions_Fear_1.0.npy')
prediction = np.load('../../models/RBF/predictions/predictions_.npy')
# prediction = np.load('../../models/RBF/predictions/test_predictions.npy')

prediction = np.zeros(100)
prediction[60] = 1
print("shape predictions", np.shape(prediction))
# squeeze third axis
# prediction = np.squeeze(prediction, axis=2)
# print("shape predictions", np.shape(prediction))
# for i in range(9):
#     print("i", i)
#     print(prediction[i].astype(int))
#     print()

# prediction = prediction[5]
print("prediction")
plt.figure()
plt.plot(prediction)

gaussian = np.arange(-10, 11)
print("shape gaussian", np.shape(gaussian))
print("gaussian", gaussian)
gaussian = np.exp((-0.1 * gaussian ** 2))
print("gaussian", gaussian)
plt.plot(gaussian)

res = convolve(prediction, gaussian, mode='same')
plt.plot(res)


plt.show()
