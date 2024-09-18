import csv

count = 0

with open('../submissions/submission6.csv', 'r') as file:

    csv_reader = csv.reader(file)

    next(csv_reader)  # Skip the header row

    for row in csv_reader:

        if float(row[1]) > 0.5:

            count += 1

print(f"Number of items with class > 0.5: {count}")