from csv import writer
import csv
import datetime
import pytz
from datetime import timedelta
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# note that time zone is a problem now the time starts at 8:00
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

with open('rawData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        abc = row
        print("hello")

with open('DataWithNormalTime.csv', 'a+',  newline='') as write_obj:
    for row in abc:
        row = row.strip()
        print(row)
        csv_writer = writer(write_obj)
        array = []
        array = row.split('=')
        newTime = datetime.datetime.fromtimestamp(int(array[0][0:10]))-timedelta(hours=8)
        csv_writer.writerow([newTime.strftime('%Y-%m-%d %H:%M:%S'), array[1]])
