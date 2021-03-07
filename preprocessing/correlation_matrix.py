import pandas
import matplotlib.pyplot as plt

# read data from data source
dataset = pandas.read_csv("data/aggregate.csv", index_col=0, delimiter=",", decimal=".")

# remove date column
withoutDate = dataset.drop(columns=['Date'])

# change data format to float
data = withoutDate.astype(float)

# calculate pearson correlation matrix
correlationMatrix = data.corr(method='pearson')

# visualize correlation matrix
plt.imshow(correlationMatrix, cmap=plt.cm.Reds, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(correlationMatrix.columns))]
plt.xticks(tick_marks, correlationMatrix.columns, rotation='vertical')
plt.yticks(tick_marks, correlationMatrix.columns)
plt.show()

# save to CSV
correlationMatrix.to_csv('data/correlation.csv')
