# math5465-spam-email

* Caps lock
* Key word (free, win, prize, congratulations, stop, reply)
* Urls (www., http)
* Containing special characters
* Long (above a certain number of characters)
* 3 digit+ number

# Logic of Email to data Matrix
* Open given csv file that contains classification and email
* Check email for various features
* Create csv file based on these features
* final csv file is in emailDataMatrix.csv and has form: 
  * First row is header
  * classification, caps lock, has key word, has url, has special character, long, has 3digit+ number
  * All features are represented with 1 if present, or 0 otherwise