import os
import json
import time
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from gensim.models import ldamodel, CoherenceModel
import pandas as pd
from pprint import pprint

BASEPATH = os.path.abspath('')
DATADIR = os.path.join(BASEPATH, 'data/')

data_path = os.path.join(DATADIR, '2020-03-13', 'comm_use_subset', 'comm_use_subset')
list_of_files = os.listdir(data_path)


def get_dict(file_name):
    json_file = os.path.join(data_path, file_name)
    with open(json_file) as f:
        data = json.load(f)
        abstract = data['abstract']
        return abstract


all_abstracts = {}

for file_name in list_of_files:
    all_abstracts[file_name] = get_dict(file_name)


def removedigit(x):
    return ''.join(ch for ch in x if not ch.isdigit())

def removepunctuation(x):
    return ''.join(ch for ch in x if not ch in punctuation)

stop_words = set(stopwords.words("english"))

def removestopwords(x):
    return ' '.join(word for word in x if not word in stop_words)


def preprocess_before(x):
    x = x.replace("\s+", " ")
    x = x.lower()
    x = x.replace("'", "")
    x = removedigit(x)
    x = removepunctuation(x)
    x = word_tokenize(x)
    x = removestopwords(x)
    x = word_tokenize(x)
    return x

to_delete = []
for abstract in all_abstracts:
    if not all_abstracts[abstract]:
        to_delete.append(abstract)
    else:
        all_abstracts[abstract] = preprocess_before(all_abstracts[abstract][0]['text'])

for a in to_delete:
    all_abstracts.pop(a)

usable_text = []
for abstract, value in all_abstracts.items():
    usable_text.append(value)

bigram = gensim.models.Phrases(usable_text)


frequency = defaultdict(int)
for text in usable_text:
    for token in text:
        frequency[token] += 1



lemmatizer = WordNetLemmatizer()

def lemming(x):
    lemlist = [lemmatizer.lemmatize(word) for word in x]
    return lemlist


lemming(usable_text[1])
usable_text = pd.Series(usable_text)
lemmed_text = usable_text.apply(lemming)

dictionary_train = corpora.Dictionary(lemmed_text)

dictionary_train.save('dictionary_train.dict')


corpus_train = [dictionary_train.doc2bow(text) for text in lemmed_text]

lda_start = time.time()
lda_mod = ldamodel.LdaModel(corpus_train, id2word=dictionary_train, num_topics=20, passes=10,per_word_topics=True)
lda_end = time.time()

print("time taken: ", (lda_end-lda_start)/60)

pprint(lda_mod.print_topics())


print(lda_mod.log_perplexity(corpus_train))

coherence_model_lda = CoherenceModel(model=lda_mod, texts=lemmed_text, dictionary=dictionary_train, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(coherence_lda)


lda_mod.get_document_topics(corpus_train[1])

lda_start = time.time()


def lda_mod_get(x, passes=10):
    newmod = ldamodel.LdaModel(corpus_train, id2word=dictionary_train, num_topics=x, passes=passes, per_word_topics=True)
    return (newmod)


lda_models = []
for x in [20,40,80,82]:
    lda_models.append(lda_mod_get(x, passes=20))
    print(x)


def perplexity_get(y):
    newperpl = lda_models[y].log_perplexity(corpus_train)
    return (newperpl)


perplexities = []
for y in range(4):
    perplexities.append(perplexity_get(y))


def coherence_get(t):
    coherence_model_lda = CoherenceModel(model=lda_models[t], texts=lemmed_text, dictionary=dictionary_train,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return (coherence_lda)


coherences = []
for t in range(4):
    coherences.append(coherence_get(t))

lda_end = time.time()

print("time taken: ", (lda_end - lda_start) / 60)

print(pd.DataFrame([coherences, perplexities]).T)

main_model = lda_mod_get(80, 10)


abstracts_keys = []
for abstract in all_abstracts:
    abstracts_keys.append(abstract)


def find_primary_topic(topic_list):
    topic_scores = []
    for topic in topic_list:
        topic_scores.append(topic[1])
    main_topic_value = max(topic_scores)
    for topic in topic_list:
        if topic[1] == main_topic_value:
            return topic[0]


def find_secondary_topic(topic_list, primary_topic):
    list_of_topics = [topic for topic in topic_list if topic[0] != primary_topic]
    topic_scores = []
    for topic in list_of_topics:
        topic_scores.append(topic[1])
    if not topic_scores:
        return None
    else:
        main_topic_value = max(topic_scores)
        for topic in list_of_topics:
            if topic[1] == main_topic_value:
                return topic[0]

total_abstracts = {}
for x in range(len(abstracts_keys)):
    total_abstracts[abstracts_keys[x]] = corpus_train[x]


topic_abstracts = {}

for key in total_abstracts:
    topic_abstracts[key] = {
        'all_topics': main_model.get_document_topics(total_abstracts[key])
    }


for key in topic_abstracts:
    topic_abstracts[key]['primary_topic'] = find_primary_topic(topic_abstracts[key]['all_topics'])


for key in topic_abstracts:
    topic_abstracts[key]['secondary_topic'] = find_secondary_topic(topic_abstracts[key]['all_topics'], topic_abstracts[key]['primary_topic'])

