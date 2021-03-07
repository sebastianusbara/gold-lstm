# Import modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

# Set plot size
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2001-01-05'

rmse_lstm = [
    1305470.42,
    476750.72,
    1260382.60,
    3529122.74,
    4528491.45,
    1012206.71,
    983174.32,
    2784692.49
]

rmse_gru = [
    628984.55,
    460679.94,
    1262250.99,
    1813811.23,
    3043471.50,
    2164846.99,
    3276504.50,
    3055371.13
]

rmse_time = [0, 1, 2, 3, 4, 5, 6, 7]

formatTime = pd.DataFrame(rmse_time, index=rmse_time)
formatTimeIndex = formatTime.index

plt.plot(formatTimeIndex, pd.DataFrame(rmse_gru)[0],
         color='#68c182', label='GRU')
plt.plot(
    formatTimeIndex,
    pd.DataFrame(rmse_lstm)[0],
    color='#ed6647', label='LSTM'
)

plt.grid(which='major', color='#cccccc', alpha=0.5)

xticksvalue = np.arange(0, 8, 1)
xtickslabel = [
    '3 Months',
    '4 Months',
    '6 Months',
    '1 Year',
    '3 Years',
    '5 Years',
    '10 Years',
    '20 Years'
]
yticksvalue = np.arange(0, 5000000, 500000)
ytickslabel = [
    "0", "500,000", "1,000,000", "1,500,000", "2,000,000", "2,500,000",
    "3,000,000", "3,500,000", "4,000,000", "4,500,000"
]
plt.yticks(yticksvalue, ytickslabel)
plt.xticks(xticksvalue, xtickslabel)
plt.legend(shadow=True)
plt.title('Root Mean Squared Error - LSTM vs GRU (Untuned Prediction Model)', family='Arial', fontsize=12)
plt.ylabel('RMSE Value', family='Arial', fontsize=10)
plt.xticks(fontsize=8)
plt.show()
