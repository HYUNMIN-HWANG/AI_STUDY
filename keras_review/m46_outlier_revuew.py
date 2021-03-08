# import numpy as np
# aaa = np.array([1,2,3,4,6,7,10, 12, 15, 16, 17, 18, 19, 20, 21,22,23,24,25,26,27,28,29,30,32,33,35,36,37,38,40,90,100])

# def outliers (data_out, column) :
#     q1, q2, q3 = np.percentile(data_out, [25, 50, 75])
#     iqr = q3 - q1
#     lower_bound = q1 - (iqr * 1.5)
#     upper_bound = q1 + (iqr * 1.5)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound))

# outlier_loc = outliers(aaa)
# print(outlier_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()

from sklearn.covariance import EllipticEnvelope
import numpy as np

aaa = np.array([[1,2,1000,3,4,5,7,8,90,100,5000],
                [1100,1200,3,1400,1500,1600,1700,8,1900,1100,1001]])

aaa = np.transpose(aaa)
print(aaa.shape)    # (11,2)

outlier = EllipticEnvelope(contamination=.2)
outlier.fit(aaa)

print(outlier.predict(aaa))

