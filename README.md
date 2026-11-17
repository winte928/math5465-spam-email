# math5465-spam-email

* Caps lock
* Key word (free, winner, prize, etc.)
* Urls (www.)
* Containing special characters
* Long (above a certain number of characters)
* Discrepencies between email and message

## What does this code do

### Outputs:
- Classification report showing precision, recall and accuracy
- Confusion matrix showing true/false positives and negatives
- Training and testing accuracy
- Total number of correct and incorrect classifications

### Creates:
- email_classified.csv — lists all emails with T/F at the start of each line, indicating whether the classification was correct
- decision_regions.png — a plot of the SVM decision boundary

## How to run this code (on a Mac)
### Download dependencies, in terminal enter:
```
pip install scikit-learn pandas numpy scipy matplotlib
```
### Then run the file, in terminal enter:
```
python classifier.py
```