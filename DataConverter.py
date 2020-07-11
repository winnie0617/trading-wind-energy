from csv import writer
import csv
import datetime

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#note that time zone is a problem now the time starts at 8:00
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

with open('/Users/issac/Documents/GitHub/trading-wind-energy/rawData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            abc = row

with open('DataWithNormalTime.csv','a+',  newline='') as write_obj:
    for row in abc:
        csv_writer = writer(write_obj)
        array = []
        array = row.split('=')
        newTime = datetime.datetime.fromtimestamp(int(array[0][0:11]))
        print(newTime)
        csv_writer.writerow([newTime.strftime('%Y-%m-%d %H:%M:%S'),array[1]])