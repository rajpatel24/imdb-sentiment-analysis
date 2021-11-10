import re
import string
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [synset.name(), swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]


def sentiment_analysis():
    df = pd.read_csv("../Dataset/movie_review.csv")
    df['pos_tagged'], df['senti_score'], df['overall_sentiment'] = 0, 0, 0

    pos = neg = obj = count = 0
    negative_words = {"needn't", "hasn't", "no", "mightn't", "mustn't", "needn", "wasn't", "shouldn't", "couldn't", "don't",
                      "didn't", "weren't", "shouldn", "nor", "wouldn't", "didn", "hadn", "don", "isn", "won", "isn't",
                      "mightn", "weren", "haven", "couldn", "hadn't", "wasn", "mustn", "aren't", "hasn", "doesn't", "won't",
                      "doesn", "aren", "not", "wouldn", "shan't", "haven't", "ain", "shan"}

    stop_words = set(stopwords.words('english')) - negative_words

    for i in range(len(df)):
        # extracting the paragraph
        paragraph = df.text[i]
        paragraph = paragraph.replace("<br />", " ")

        # tokenizing paragraph into sentences
        sentences = sent_tokenize(paragraph)

        pos_tagged_words = []
        # lemmatized_words = []
        for sentence in sentences:
            # removing punctuations
            sentence = sentence.translate(str.maketrans('', '', string.punctuation.replace("'", "")))

            # splitting the sentence into words
            words = re.split(r"[\s+.]", sentence)

            # removing the empty strings
            words = filter(None, words)

            # removing stopwords
            words = [word for word in words if not word.lower() in stop_words]

            # separating numbers from a string
            for word in words:
                match = re.match(r"([a-z]+)([0-9]+)", word, re.I)
                if match:
                    words.remove(word)
                    items = match.groups()
                    words.extend(list(items))

            # labelling each word with appropriate PoS tag
            pos_tagged_words.append(nltk.pos_tag(words))

        df['pos_tagged'][i] = pos_tagged_words

        final_post_list = []
        for pos_tag in df.pos_tagged[i]:
            for item in pos_tag:
                final_post_list.append(item)

        senti_val = [get_sentiment(word, tag) for (word, tag) in final_post_list]
        for score in senti_val:
            try:
                pos = pos + score[1]  # positive score is stored at 2nd position
                neg = neg + score[2]  # negative score is stored at 3rd position
            except:
                continue
        df['senti_score'][i] = pos-neg
        pos = neg = 0

        if df['senti_score'][i] > 0.250:
            df['overall_sentiment'][i] = 1
        else:
            df['overall_sentiment'][i] = 0

    print("\n\n ------------------------ \n\n ", df)
    df.to_csv('sentiment_output.csv', columns=['text', 'senti_score', 'overall_sentiment'])


if __name__ == '__main__':
    sentiment_analysis()
