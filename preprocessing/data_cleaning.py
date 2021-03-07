import pandas

# read data from data source
hangseng = pandas.read_csv("data/hangseng.csv", delimiter=";", decimal=",")
nasdaq = pandas.read_csv("data/nasdaq.csv", delimiter=";", decimal=",")
ihsg = pandas.read_csv("data/ihsg.csv", delimiter=";", decimal=",")
nikkei = pandas.read_csv("data/nikkei-225.csv", delimiter=";", decimal=",")
usd = pandas.read_csv("data/usd-idr.csv", delimiter=";", decimal=",")
gold = pandas.read_csv("data/gold.csv", delimiter=";", decimal=",")
silver = pandas.read_csv("data/gold-silver.csv", delimiter=";", decimal=",")
aud = pandas.read_csv("data/aud-usd.csv", delimiter=";", decimal=",")

# currency market
currency = pandas.merge(usd, aud)  # commodity market

# stock market index
inter = pandas.merge(hangseng, nasdaq)  # international stock market
asian = pandas.merge(nikkei, ihsg)  # asian stock market
stock = pandas.merge(asian, inter)  # merge all stock market

# currency + stock market
stockCurrency = pandas.merge(stock, currency)

# currency + stock market + silver
market = pandas.merge(silver, stockCurrency)

# aggregate all data
aggregate = pandas.merge(gold, market)

# export to csv file
aggregate.to_csv('data/aggregate.csv')