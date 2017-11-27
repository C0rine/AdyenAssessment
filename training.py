# Corine Jacobs
# CJDJacobs@outlook.com

# Note: the code assumes the trainings data is ordered on time of transaction, starting with the oldest transactions.

import csv
import dateutil.parser as dparser
from sklearn import linear_model, svm, neighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

recordlist = []
groundtruth = []
emails = []
cardhashes = []
transtimes_email = []
transtimes_cardhash = []

# Open the trainingdata as dicts
with open("txs-50.csv") as f:
    records = csv.DictReader(f)
    for row in records:
        recordlist.append(row)

#Preprocess the data
for entry in recordlist:
    time = dparser.parse(entry['creation_date'], fuzzy=True, dayfirst=True)

    # Calculate value to represent the time between two transactions with the same EMAIL and add as feature
    if entry['shopper_email'] not in emails:
        emails.append(entry['shopper_email'])
        transtimes_email.append(time)
        entry['timediff_email'] = 0
    else:
        i = emails.index(entry['shopper_email'])
        time = dparser.parse(entry['creation_date'], fuzzy=True, dayfirst=True)
        timediff =(time - transtimes_email[i]).total_seconds()
        if timediff == 0:
            entry['timediff_email'] = 1.00 / 0.1
        else:
            entry['timediff_email'] = 1.00 / timediff
        transtimes_email[i] = time

    # Calculate value to represent the time between two transactions with the same CARD HASH and add as feature
    if entry['card_number_hash'] not in cardhashes:
        cardhashes.append(entry['card_number_hash'])
        transtimes_cardhash.append(time)
        entry['timediff_card'] = 0
    else:
        i = cardhashes.index(entry['card_number_hash'])
        time = dparser.parse(entry['creation_date'], fuzzy=True, dayfirst=True)
        timediff =(time - transtimes_cardhash[i]).total_seconds()
        if timediff == 0:
            entry['timediff_card'] = 1.00 / 0.1
        else:
            entry['timediff_card'] = 1.00 / timediff
        transtimes_cardhash[i] = time

    # FAILED ATTEMPTS TO IMPROVE ACCURACY
    # Tried to see if the domain name in the e-mail made any difference, but it turns out it actually lowers accuracy
    # entry['shopper_email'] = entry['shopper_email'][:-4].split('@')[1]

    # Add unconverted amount
    # entry['amount'] = int(entry['amount'])

    # Use converted amounts
    # if entry['currency'] == 'USD':
    #     entry['amount'] = float(entry['amount'])
    # elif entry['currency'] == 'AUD':
    #     entry['amount'] = float(entry['amount']) * 0.76
    # elif entry['currency'] == 'CAD':
    #     entry['amount'] = float(entry['amount']) * 0.75
    # elif entry['currency'] == 'CNY':
    #     entry['amount'] = float(entry['amount']) * 0.15
    # elif entry['currency'] == 'EUR':
    #     entry['amount'] = float(entry['amount']) * 1.11
    # elif entry['currency'] == 'GBP':
    #     entry['amount'] = float(entry['amount']) * 1.26
    # elif entry['currency'] == 'NOK':
    #     entry['amount'] = float(entry['amount']) * 0.12
    # elif entry['currency'] == 'SEK':
    #     entry['amount'] = float(entry['amount']) * 0.11
    # else:
    #     entry['amount'] = float(entry['amount'])


    # Save the ground truth in a separate list
    groundtruth.append(entry['fraud'])

    # Sanitize the data, remove any tuples we do not want to use during training
    entriesToRemove = ('txid', 'creation_date', 'amount', 'fraud')
    for k in entriesToRemove:
        entry.pop(k, None)
    print entry


# OneHot encode to get trainable data
vec = DictVectorizer()
data = vec.fit_transform(recordlist).toarray()

# split in train and test
train_data, test_data, train_truth, test_truth = train_test_split(data, groundtruth, test_size=0.3, random_state=42)

# Train model
# PICK A MODEL HERE:
# -----------------------------------
# classifier = linear_model.SGDClassifier(shuffle=True) # 86-94%
classifier = svm.SVC(kernel='linear') # 97%
# classifier = neighbors.KNeighborsClassifier(20, weights='distance') # 80%
# -----------------------------------
classifier.fit(train_data, train_truth)

# Get accuracy of the model, by classifying test data
score = 0
for i, item in enumerate(test_data):
    if classifier.predict([test_data[i]])==test_truth[i]:
        score += 1
accuracy = (100*score)/len(test_truth)


print '\n --> Accuracy: ' + str(accuracy) + '% <--'


