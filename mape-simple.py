# Import modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

# Set plot size
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2001-01-05'

mape_lstm = [
    4.68,
    1.40,
    4.07,
    12.57,
    17.02,
    4.61,
    4.46,
    14.43
]

mape_gru = [
    1.96,
    1.30,
    4.07,
    6.43,
    11.61,
    9.40,
    16.89,
    15.90
]

mape_time = [0, 1, 2, 3, 4, 5, 6, 7]

formatTime = pd.DataFrame(mape_time, index=mape_time)
formatTimeIndex = formatTime.index

plt.plot(formatTimeIndex, pd.DataFrame(mape_gru)[0],
         color='#68c182', label='GRU')
plt.plot(
    formatTimeIndex,
    pd.DataFrame(mape_lstm)[0],
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
plt.xticks(xticksvalue, xtickslabel)
plt.legend(shadow=True)
plt.title('Mean Absolute Percentage Error - LSTM vs GRU (Untuned Prediction Model)', family='Arial', fontsize=12)
plt.ylabel('MAPE Value in %', family='Arial', fontsize=10)
plt.xticks(fontsize=8)
plt.show()
