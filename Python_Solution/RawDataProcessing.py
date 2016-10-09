
import numpy
import random
import re
import nltk
# from nltk.corpus import stopwords
import Contractions
import json
import os

print('loading raw data...')

with open("Coursera-SwiftKey/final/en_US/en_US.twitter.txt", encoding="utf8") as f:
    twitters = f.readlines()
with open("Coursera-SwiftKey/final/en_US/en_US.blogs.txt", encoding="utf8") as f:
    blogs = f.readlines()
with open("Coursera-SwiftKey/final/en_US/en_US.news.txt", encoding="utf8") as f:
    news = f.readlines()
with open('profanity.txt') as f:
    profanity = f.read().splitlines()

print('sampling data...')
sample_percent = 0.15
random.seed(2335)


def sample_data(data, percent):
    data_size = len(data)
    sample_size = int(data_size * percent)
    idx = random.sample(range(data_size), sample_size)
    # idx.sort()
    sub_data = [data[i] for i in idx]
    return sub_data

sub_twitters = sample_data(twitters, sample_percent)
sub_blogs = sample_data(blogs, sample_percent)
sub_news = sample_data(news, sample_percent)

print('combine sample data...')
total = sub_twitters
total.extend(sub_blogs)
total.extend(sub_news)

del twitters, sub_twitters
del blogs, sub_blogs
del news, sub_news


# stopwords_str = r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*'
# stopwords_pattern = re.compile(stopwords_str)

profanity_words = r'\b(' + r'|'.join(profanity) + r')\b\s*'
profanity_pattern = re.compile(profanity_words)

non_ascii_pattern = re.compile('[^\x00-\x7F]')
non_english_pattern = re.compile('[^a-zA-Z_]')
# url twitter hashtags
twitteruser_url_pattern = re.compile('(@[A-Za-z0-9]+)|(\#\S+)|(http\S+)|(www.\S+)|(\w+:\/\/\S+)|(\/\/\S+)|(\w+\.com\S+)')

# email
email_pattern = re.compile(r'[\w.-]+@[\w.-]+')

contractions_re = re.compile('(%s)' % '|'.join(Contractions.contractions.keys()))


def expand_contractions(s, contractions_dict=Contractions.contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

print('cleaning data...')

total_str = ''
for line in total:
    # str = stopwords_pattern.sub('', str)

    # remove no ascii code
    sub_str = non_ascii_pattern.sub(' ', line)

    # set to lower case
    sub_str = sub_str.lower()

    # remove lines with profanity
    if profanity_pattern.search(sub_str) is not None:
        continue

    # remove url twitter hashtags
    sub_str = twitteruser_url_pattern.sub(' ', sub_str)

    # remove email
    sub_str = email_pattern.sub(' ', sub_str)

    # expand contractions
    sub_str = expand_contractions(sub_str)

    # remove non-english characters: punctuation and numbers
    sub_str = non_english_pattern.sub(' ', sub_str)

    # strip white spaces
    sub_str = re.sub(' +', ' ', sub_str)

    total_str += ' ' + sub_str

print('tokenization...')
# token
tokens = nltk.word_tokenize(total_str)
del total, total_str

# unigram and frequency
print('create unigram...')
fdist_unigram = nltk.FreqDist(tokens)

# bigram and frequency
print('create bigram...')
bigram = nltk.bigrams(tokens)
bigram_list = list(bigram)
fdist_bigram = nltk.FreqDist(bigram_list)

# trigram and frequency
print('create trigram...')
trigram = nltk.trigrams(tokens)
trigram_list = list(trigram)
fdist_trigram = nltk.FreqDist(trigram_list)

# four-gram and frequency
print('create four-grame...')
fourgram = nltk.ngrams(tokens, 4)
fourgram_list = list(fourgram)
fdist_fourgram = nltk.FreqDist(fourgram_list)

del tokens

# remove low frequent words
# step 1: extract low frequent words from unigram
threshold = 10
low_freq_words = dict()
unigram_frequency_dictionary = dict()
for key, value in fdist_unigram.items():
    if value <= threshold:
        low_freq_words[key] = value
    else:
        unigram_frequency_dictionary[key] = value


def match_low_freq_words(words):
    for word in words:
        if low_freq_words.get(word) is not None:
            return True
    return False


def remove_low_frequent_words(fdist, thresh_hold):
    temp_dict = dict()
    for k, v in fdist.items():
        if v <= thresh_hold:
            continue
        if match_low_freq_words(k):
            continue
        else:
            temp_dict[' '.join(k)] = v
    return temp_dict

print('remove_low_frequent_words: bigram...')
bigram_frequency_dictionary = remove_low_frequent_words(fdist_bigram, 1)
del fdist_unigram, bigram, bigram_list, fdist_bigram

print('remove_low_frequent_words: trigram...')
trigram_frequency_dictionary = remove_low_frequent_words(fdist_trigram, 2)
del trigram, trigram_list, fdist_trigram

print('remove_low_frequent_words: fourgram...')
fourgram_frequency_dictionary = remove_low_frequent_words(fdist_fourgram, 3)
del fourgram, fourgram_list, fdist_fourgram


def save_frequency_as_json(frequency_dictionary, file):
    output = open(file, 'w')
    json.dump(frequency_dictionary, output)
    output.close()


# write frequency data to local file
def write_freq_to_file(freq_dist, file):
    with open(file, 'a') as output:
        for word in freq_dist.keys():
            key_value = '"{}" {}'.format(word, freq_dist.get(word))
            print(key_value, file=output)

print('save frequency data...')

# os.remove('low_freq_words_json.txt')
# os.remove('freq_1_final_json.txt')
# os.remove('freq_2_final_json.txt')
# os.remove('freq_3_final_json.txt')
# os.remove('freq_4_final_json.txt')
os.remove('low_freq_words.txt')
os.remove('freq_1_final.txt')
os.remove('freq_2_final.txt')
os.remove('freq_3_final.txt')
os.remove('freq_4_final.txt')

# save_frequency_as_json(low_freq_words, 'low_freq_words_json.txt')
write_freq_to_file(low_freq_words, 'low_freq_words.txt')
del low_freq_words

# save_frequency_as_json(unigram_frequency_dictionary, 'freq_1_final_json.txt')
write_freq_to_file(unigram_frequency_dictionary, 'freq_1_final.txt')
del unigram_frequency_dictionary

# save_frequency_as_json(bigram_frequency_dictionary, 'freq_2_final_json.txt')
write_freq_to_file(bigram_frequency_dictionary, 'freq_2_final.txt')
del bigram_frequency_dictionary

# save_frequency_as_json(trigram_frequency_dictionary, 'freq_3_final_json.txt')
write_freq_to_file(trigram_frequency_dictionary, 'freq_3_final.txt')
del trigram_frequency_dictionary

# save_frequency_as_json(fourgram_frequency_dictionary, 'freq_4_final_json.txt')
write_freq_to_file(fourgram_frequency_dictionary, 'freq_4_final.txt')
del fourgram_frequency_dictionary

print('done!!!')

'''
# tag and backoff n-grams
trains = []
for str in total:
    # remove non-english characters and numbers
    sub_str = pattern.sub(' ', str).lower()
    # strip extra whitespaces
    #tem_str = ' '.join(sub_str.split())
    #token = nltk.word_tokenize(tem_str)
    #tem_str = nltk.pos_tag(token)
    tem_str = nltk.pos_tag(sub_str.split())
    trains.append(tem_str)

tagger = None
for n in range(1, 5):
    tagger = nltk.NgramTagger(n, trains, backoff=tagger)
'''
