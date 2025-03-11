import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bertopic import BERTopic

def sentiment_analyzer(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05 :
        Sentiment = 'Positive'
    elif scores['compound'] <= -0.05 :
        Sentiment = 'Negative'
    else :
        Sentiment = 'Neutral'
    return Sentiment

def bertopic(data):
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(data['text1'])
    data['text_Topic'] = topics
    data['text_Probability'] = probabilities
    return topic_model, data

def spacy_ner(data):
    model = spacy.load("en_core_web_sm")

    def ner(text):
        doc = model(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        return entities
    
    data['entities'] = data['text1'].apply(ner)

    ner_data = []
    for index, row in data.iterrows():
        text = row['text1']
        for entity, entity_type in row['entities']:
            ner_data.append([text, entity, entity_type])

    ner_data = pd.DataFrame(ner_data, columns=['text', 'entity', 'entity_type'])
    return ner_data
