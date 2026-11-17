import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

# Load data
df = pd.read_csv("email.csv")

# Encode labels: spam = 1, ham = 0
df['target'] = (df['Category'] == 'spam').astype(int)

# Extract features
SPAM_KEYWORDS = [
    "free", "winner", "prize", "won", "win", "cash", "reward",
    "offer", "urgent", "congratulations", "claim", "guaranteed",
    "click here", "act now", "limited time", "risk free",
    "million", "billion", "credit", "loan", "discount", "deal"
]

def extract_custom_features(texts):
    """
    Returns a 2D array with one row per email:
      0: caps_ratio: fraction of letters that are uppercase
      1: exclamation_count: number of '!' characters
      2: keyword_count: number of spam keywords matched
      3: url_count: number of URLs (www. / http)
      4: special_char_ratio: fraction of chars that are $, €, %, #, *
      5: is_long: 1 if email is above 500 characters
      6: email_mismatch: 1 if multiple different domains found
    """
    features = []
    for text in texts:
        text = str(text)
        total_chars = max(len(text), 1)

        # Caps ratio
        letters = [c for c in text if c.isalpha()]
        caps_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)

        # Exclamation marks
        exclamation_count = text.count("!")

        # Spam keyword count
        text_lower = text.lower()
        keyword_count = sum(1 for kw in SPAM_KEYWORDS if kw in text_lower)

        # URL count
        url_count = len(re.findall(r'(www\.)\S+', text, re.IGNORECASE))

        # Special character ratio
        special_chars = sum(1 for c in text if c in "$€%#*")
        special_char_ratio = special_chars / total_chars

        # Is long (above 500 characters)
        is_long = int(len(text) > 500)

        # Email domain mismatch
        domains = re.findall(r'@([\w\.\-]+)', text_lower)
        email_mismatch = int(len(set(domains)) > 1)

        features.append([caps_ratio, exclamation_count, keyword_count, url_count, special_char_ratio, is_long, email_mismatch])
    return np.array(features, dtype=float)

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(df['Message'], df['target'], test_size=0.2, random_state=42)

# Feature matrix
tfidf = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))

# Catches obfuscated words like "fr€€"
tfidf_char = TfidfVectorizer(max_features=5000, ngram_range=(3, 5), analyzer="char_wb")

x_train_words = tfidf.fit_transform(x_train)
x_test_words = tfidf.transform(x_test)
x_train_chars = tfidf_char.fit_transform(x_train)
x_test_chars = tfidf_char.transform(x_test)
x_train_custom = csr_matrix(extract_custom_features(x_train))
x_test_custom = csr_matrix(extract_custom_features(x_test))

# Combine all features
x_train_combined = hstack([x_train_words, x_train_chars, x_train_custom])
x_test_combined = hstack([x_test_words, x_test_chars, x_test_custom])

# Train SVM
clf = SVC(kernel='linear', class_weight='balanced')
clf.fit(x_train_combined, y_train)

# Evaluate
y_pred = clf.predict(x_test_combined)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"True Negatives (Ham → Ham ): {cm[0,0]}")
print(f"False Positives (Ham → Spam): {cm[0,1]}")
print(f"False Negatives (Spam → Ham ): {cm[1,0]}")
print(f"True Positives (Spam → Spam): {cm[1,1]}")

# Training accuracy
y_train_pred = clf.predict(x_train_combined)
train_acc = np.mean(y_train_pred == y_train) * 100
test_acc = np.mean(y_pred == y_test) * 100
print('\nTraining Accuracy: %.2f%%' % train_acc)
print('Testing Accuracy: %.2f%%' % test_acc)

# Plot decision regions
def plot_region(X, L, clf, alpha=0.5, cmap='Paired',cp=np.array([0.5, 2.5, 6.5, 8.5, 10.5, 11.5]),
                markers=['o', 's', 'D', '^', 'v', 'p'], vmin=0,vmax=12,markersize=75,
                linewidths=1.25,markerlinewidths=1):
    x,y = X[:,0], X[:,1]
    f = 0.1 * np.maximum(np.max(np.abs(x)), np.max(np.abs(y)))
    xmin, xmax = np.min(x)-f,np.max(x)+f
    ymin, ymax = np.min(y)-f,np.max(y)+f
    plt.figure()
    for i, color in enumerate(np.unique(cp[L])):
        sub = cp[L] == color
        plt.scatter(x[sub],y[sub],zorder=2,c=cp[L][sub],cmap=cmap,edgecolors='black',
                    vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)
    X,Y = np.mgrid[xmin:xmax:0.01,ymin:ymax:0.01]
    z = clf.predict(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)
    plt.contourf(X, Y, cp[z],alpha=alpha,cmap=cmap,antialiased=True,vmin=vmin,vmax=vmax)
    plt.contour(X, Y, z, colors='black',linewidths=linewidths,antialiased=True)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('SVM Decision Regions')
    plt.tight_layout()
    plt.savefig('decision_regions.png', dpi=150)
    plt.show()

svd = TruncatedSVD(n_components=2, random_state=42)
x_2d = svd.fit_transform(x_test_combined)
clf_2d = SVC(kernel='linear', class_weight='balanced')
clf_2d.fit(x_2d, y_test)
plot_region(x_2d, y_test.values, clf_2d)

# Predict with new email
def predict_email(email_text):
    word_feat = tfidf.transform([email_text])
    char_feat = tfidf_char.transform([email_text])
    custom_feat = csr_matrix(extract_custom_features([email_text]))
    combined = hstack([word_feat, char_feat, custom_feat])
    prediction = clf.predict(combined)[0]
    return "SPAM" if prediction == 1 else "HAM (real email)"

# Export classifications to emails_classified.csv
# Run predictions on email.csv
x = df['Message']
x_all_words = tfidf.transform(x)
x_all_chars = tfidf_char.transform(x)
x_all_custom = csr_matrix(extract_custom_features(x))
x_all_combined = hstack([x_all_words, x_all_chars, x_all_custom])

all_predictions = clf.predict(x_all_combined)

df['Result'] = np.where(df['target'] == all_predictions, 'True', 'False - ' + np.where(all_predictions == 1, 'spam', 'ham'))
df[['Result', 'Message']].to_csv('email_classified.csv', index=False)

print(f"Total classified: {len(df)}")
print(f"Correct: {(df['Result'] == 'True').sum()}")
print(f"Incorrect: {(df['Result'] != 'True').sum()}")