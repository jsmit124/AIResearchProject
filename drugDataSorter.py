import csv
import queue

noDrugData = queue.Queue()
legalDrugNoIllegalDrugData = queue.Queue()
legalDrugAndIllegalDrugData = queue.Queue()
noLegalDrugButIllegalDrugData = queue.Queue()
classifiedDrugData = queue.Queue()

header = ["RecordNumber", "Age", "Gender", "Education", "Country", "Ethnicity", "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS", "Alcohol", "Cannabis", "Nicotine", "Amphet", "Amyl", "Benzos", "Cocaine", "Crack", "Ecstasy", "Heroin", "Ketamine", "LSD", "Meth", "Mushrooms", "Semeron", "VolatileSubstance"]
classifiedHeader = ["RecordNumber", "Age", "Gender", "Education", "Country", "Ethnicity", "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS", "Classifier"]

noDrugData.put(header)
legalDrugNoIllegalDrugData.put(header)
legalDrugAndIllegalDrugData.put(header)
noLegalDrugButIllegalDrugData.put(header)
classifiedDrugData.put(classifiedHeader)

comma = ", "

with open('drug_consumption.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    for row in csv_reader:
        if line_count == 0:
            print(type(row))
            line_count += 1
        else:
            recordNumber = str(row[0])
            age = str(row[1])
            gender = str(row[2])
            education = str(row[3])
            country = str(row[4])
            ethnicity = str(row[5])
            nscore = str(row[6])
            escore = str(row[7])
            oscore = str(row[8])
            ascore = str(row[9])
            cscore = str(row[10])
            impulsive = str(row[11])
            ss = str(row[12])
            
            alcohol = int(row[13])
            cannabis = int(row[14])
            nicotine = int(row[15])
            amphetamines = int(row[16])
            amyl = int(row[17])
            benzos = int(row[18])
            cocaine = int(row[19])
            crack = int(row[20])
            ecstacy = int(row[21])
            heroin = int(row[22])
            ketamine = int(row[23])
            lsd = int(row[24])
            meth = int(row[25])
            mushrooms = int(row[26])
            semeron = int(row[27])
            volatileSubstances = int(row[28])

            didLegalDrugs = alcohol + cannabis + nicotine
            didIllegalDrugs = amphetamines + amyl + benzos + cocaine + crack + ecstacy + heroin + ketamine + lsd + meth + mushrooms + semeron + volatileSubstances

            classifier = 0
            
            if didLegalDrugs == 0 and didIllegalDrugs == 0:
                noDrugData.put(row)

            if didLegalDrugs > 0 and didIllegalDrugs == 0:
                legalDrugNoIllegalDrugData.put(row)

            if didLegalDrugs > 0 and didIllegalDrugs > 0:
                classifier += 1
                legalDrugAndIllegalDrugData.put(row)

            #not including data who didn't do legal drugs but did do illegal drugs in classifier
            if didLegalDrugs == 0 and didIllegalDrugs > 0:
                noLegalDrugButIllegalDrugData.put(row)

            info = [recordNumber, age, gender, education, country, ethnicity, nscore, escore, oscore, ascore, cscore, impulsive, ss, str(classifier)]
            classifiedDrugData.put(info)

            line_count += 1
            
    print(f'Processed {line_count} lines.')


with open('sortedData/legalDrugNoIllegalDrugData.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',', )
    for data in range(0, legalDrugNoIllegalDrugData.qsize()):
        writer.writerow(legalDrugNoIllegalDrugData.get())

with open('sortedData/noDrugData.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for data in range(0, noDrugData.qsize()):
        writer.writerow(noDrugData.get())

with open('sortedData/LegalDrugAndIllegalDrugData.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for data in range(0, legalDrugAndIllegalDrugData.qsize()):
        writer.writerow(legalDrugAndIllegalDrugData.get())

with open('sortedData/noLegalButDidIllegalDrugData.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for data in range(0, noLegalDrugButIllegalDrugData.qsize()):
        writer.writerow(noLegalDrugButIllegalDrugData.get())

with open('sortedData/ClassifiedDrugData.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for data in range(0, classifiedDrugData.qsize()):
        writer.writerow(classifiedDrugData.get())

print("Files saved in the sortedData folder")
