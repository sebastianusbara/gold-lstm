# Import modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

# Set plot size
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2001-01-05'

mse_gru = [
    0.189798,
    0.110262,
    0.291101,
    0.515070,
    0.357862,
    0.145600,
    0.047422,
    0.030286
]

mse_lstm = [
    0.215380,
    0.295917,
    0.587962,
    0.683635,
    0.451388,
    0.259811,
    0.028762,
    0.018706
]
mse_time = [0, 1, 2, 3, 4, 5, 6, 7]

formatTime = pd.DataFrame(mse_time, index=mse_time)
formatTimeIndex = formatTime.index

plt.plot(formatTimeIndex, pd.DataFrame(mse_gru)[0],
         color='#68c182', label='GRU')
plt.plot(
    formatTimeIndex,
    pd.DataFrame(mse_lstm)[0],
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
    '19 Years'
]
plt.xticks(xticksvalue, xtickslabel)
plt.legend(shadow=True)
plt.title('Mean Squared Error - LSTM vs GRU', family='Arial', fontsize=12)
plt.ylabel('MSE Value', family='Arial', fontsize=10)
plt.xticks(fontsize=8)
plt.show()
