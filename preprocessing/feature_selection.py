import pandas
import numpy

# read data from data source
dataset = pandas.read_csv("data/aggregate.csv", index_col=0, delimiter=",", decimal=".")

# remove date column
withoutDate = dataset.drop(columns=['Date'])

# change data format to float
data = withoutDate.astype(float)

# calculate pearson correlation matrix
correlationMatrix = data.corr(method='pearson')

# remove column that has insignificant correlationMatrix
cor_target = abs(correlationMatrix['IDR'])
relevant_features = cor_target[cor_target > 0.75]

# save final train dataset
features = relevant_features.keys().values
features = numpy.append('Date', features)
train = dataset[features]
totalRow = train.shape[0]
validationData = round(totalRow / 10) * 2
testingData = round(totalRow / 10)
testing6Month = 18

# filter by time
train30 = train[totalRow - 30: totalRow].reset_index(drop=True)
train90 = train[totalRow - 90: totalRow].reset_index(drop=True)
train120 = train[totalRow - 120: totalRow].reset_index(drop=True)
train180 = train[totalRow - 180: totalRow].reset_index(drop=True)
train360 = train[totalRow - 360: totalRow].reset_index(drop=True)
train1080 = train[totalRow - 1080: totalRow].reset_index(drop=True)
train1800 = train[totalRow - 1800: totalRow].reset_index(drop=True)
train3600 = train[totalRow - 3600: totalRow].reset_index(drop=True)
validation = train[totalRow - validationData: totalRow - testingData].reset_index(drop=True)
testing = train[totalRow - testingData: totalRow].reset_index(drop=True)
testing180 = train[totalRow - testing6Month: totalRow].reset_index(drop=True)

# export file
train.to_csv('data/train.csv')
train30.to_csv('data/train-30.csv')
train90.to_csv('data/train-90.csv')
train120.to_csv('data/train-120.csv')
train180.to_csv('data/train-180.csv')
train360.to_csv('data/train-360.csv')
train1080.to_csv('data/train-1080.csv')
train1800.to_csv('data/train-1800.csv')
train3600.to_csv('data/train-3600.csv')
validation.to_csv('data/validation.csv')
testing.to_csv('data/testing.csv')
testing180.to_csv('data/testing-180.csv')
