import csv
import serial
from datetime import datetime, timedelta

ser = serial.Serial('COM4')
interval = 2  # Number of hours after which new file is made
time_now = datetime.now()
update_time = time_now + timedelta(hours=interval)
fileName = "{}.csv".format(time_now)

while True:
    if time_now > update_time:
        fileName = "{}.csv".format(datetime.now())
        update_time = time_now + timedelta(hours=interval)
    with open(fileName, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["DateTime", "Data 1", "Data 2", "Data 3", "Data 4", "Data 5", "Data 6", "Data 7", "Data 8"])
        if ser.in_waiting > 0:
            temp = ser.readline()
            data = temp.split()
            data.insert(0, datetime.now())
            writer.writerow(temp)

    time_now = datetime.now()
