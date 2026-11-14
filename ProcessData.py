import csv


def searchForKeys(emailInfo):
    triggerWords = ["free", "win", "prize", "congratulations", "stop", "reply"]
    urlWords = [".com", "http", "www."]
    specialCharacters = ['@','#','$','%','^','&','*', '£']
    hasTrigger = any(trigger in emailInfo.lower() for trigger in triggerWords)
    hasUrl = any(url in emailInfo.lower() for url in urlWords)
    hasSpecialChar = any(char in emailInfo for char in specialCharacters)
    return (int(hasTrigger), int(hasUrl), int(hasSpecialChar))

def checkInARow(emailInfo):
    upperInRow = 0
    numberInRow = 0
    capsAchieved = 0
    numAchieved = 0
    for char in emailInfo:
        if char.isupper():
            numberInRow = 0
            upperInRow += 1
            if upperInRow == 3:
                capsAchieved = 1
        elif char.isnumeric():
            upperInRow = 0
            numberInRow += 1
            if numberInRow == 3:
                numAchieved = 1
        else:
            upperInRow = 0
            numberInRow = 0

    return (capsAchieved, numAchieved)

def createFeatureData(emailInfo):
    emailData = [None] * 7
    if emailInfo[0] == "spam":
        emailData[0] = 1
    elif emailInfo[0] == "ham":
        emailData[0] = 0
    searched = searchForKeys(emailInfo[1])
    inARow = checkInARow(emailInfo[1])
    emailData[1] = inARow[0]
    emailData[2] = searched[0]
    emailData[3] = searched[1]
    emailData[4] = searched[2]
    if len(emailInfo[1]) > 125:
        emailData[5] = 1
    else:
        emailData[5] = 0
    emailData[6] = inARow[1]
    return emailData


with open("emailDataMatrix.csv",  mode='w', newline='', encoding='utf-8') as dataMatrixFile:
    dataMatrixWriter= csv.writer(dataMatrixFile)
    dataMatrixWriter.writerow(["Classification", " Caps Lock", " Key Word", " Urls", " Special Character", " Long", " Number"])

    with open('email.csv', mode='r', newline='', encoding='utf-8') as emailTxt:
        emailReader = csv.reader(emailTxt)

        next(emailReader)

        for row in emailReader:
            featureData = createFeatureData(row)
            dataMatrixWriter.writerow(featureData)