import csv

array = [142, 22, 31, 4, 15, 66, 97]

# open the file in the write mode
f = open('files/csvFile.csv', 'w')

# create the csv writer
writer = csv.writer(f)


writer.writerow(array)
for i in range(0, 49):
    for i in range(0, len(array)):
        array[i] = array[i] + 10 - 2
    writer.writerow(array)

f.close()
