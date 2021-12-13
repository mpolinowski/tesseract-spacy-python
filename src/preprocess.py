import spacy
from spacy.tokens import DocBin
import pickle

nlp = spacy.blank("en")

training_data = pickle.load(open('./data/TrainData.pickle', 'rb'))
testing_data = pickle.load(open('./data/TestData.pickle', 'rb'))

# Create training data file in spaCy format
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/train.spacy")

# Create testing data file in spaCy format
db_test = DocBin()
for text, annotations in testing_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db_test.to_disk("./data/test.spacy")