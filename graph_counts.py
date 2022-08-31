import csv
import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

datetimes = []
counts = []

with open('data/counts/scott_base-21-22.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines)
    for row in lines:
        datetimes.append(dt.datetime.strptime(row[0], "%Y-%m-%dT%H_%M_%S"))
        counts.append(int(row[1]))

plt.scatter(datetimes, counts, color='b')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Scott Base Seal Counts (2021-2022)', fontsize=20)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(datetimes, counts)
plt.gcf().autofmt_xdate()

plt.show()
