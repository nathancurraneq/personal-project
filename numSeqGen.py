import csv

array = [1, 1, 1, 1, 1, 1, 1]

# open the file in the write mode
f = open('csvFile.csv', 'w')

# create the csv writer
writer = csv.writer(f)


writer.writerow(array)
for i in range(0, 49):
    for i in range(0, len(array)):
        array[i] += 1
    writer.writerow(array)

f.close()
