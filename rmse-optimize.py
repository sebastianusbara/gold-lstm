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
    499991.96,
    464771.30,
    336763.85,
    384743.68,
    1435578.89,
    449659.67,
    229803.91,
    213036.85
]

rmse_gru = [
    482388.97,
    377447.84,
    515689.45,
    377166.67,
    345507.15,
    338228.05,
    249542.04,
    216012.48
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
yticksvalue = np.arange(100000, 2100000, 200000)
ytickslabel = [
    "100,000", "300,000", "500,000", "700,000", "900,000", "1,100,000",
    "1,300,000", "1,500,000", "1,700,000", "1,900,000"
]
plt.yticks(yticksvalue, ytickslabel)
plt.xticks(xticksvalue, xtickslabel)
plt.legend(shadow=True)
plt.title('Root Mean Squared Error - LSTM vs GRU (Tuned Prediction Model)', family='Arial', fontsize=12)
plt.ylabel('RMSE Value', family='Arial', fontsize=10)
plt.xticks(fontsize=8)
plt.show()
