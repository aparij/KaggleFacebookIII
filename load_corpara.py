__author__ = 'alex parij'

from gensim import corpora, models, similarities
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
import nltk
import re, string
import logging
import numpy as np
import mkl
from collections import Counter


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus(object):
    table = string.maketrans("","")
    LIMIT = 200000
    stopwords = nltk.corpus.stopwords.words('english')
    s = nltk.stem.SnowballStemmer('english')

    def __init__(self, dictionary, train_path):
        self.dictionary = dictionary
        self.train_path = train_path

    def __iter__(self):
        complete_line = ''
        i = 0

        with open(self.train_path) as f:
            next(f)
            for line in f:

                if line == '\n':
                    continue
                line = line.lower()
                code_start = line.find("<code>")
                code_end = line.find("</code>")
                if code_start != -1:
                    complete_line += line[:code_start]
                elif code_end != -1:
                    complete_line += line[code_end:]
                else:
                    complete_line += line

                if "\r" not in line:
                    continue
                else:
                    complete_line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', complete_line)
                    split_line = complete_line.split('","')
                    #remove <p> in the beginning and end of the body
                    split_line[2] = split_line[2][3:-4]
                    split_line[2] = split_line[2][3:-4]
                    complete_line = ''
                    i += 1
                processed_line = []
                processed_line.extend([self.s.stem(word) for word in split_line[1].translate(self.table, string.punctuation).split() if word not in self.stopwords and not word.isdigit()])
                processed_line.extend([self.s.stem(word) for word in split_line[2].translate(self.table, string.punctuation).split() if word not in self.stopwords and not word.isdigit()])
                if i == self.LIMIT:
                    break

                yield self.dictionary.doc2bow(processed_line)

class MyDict(object):
    table = string.maketrans("","")
    LIMIT = 200000
    stopwords = nltk.corpus.stopwords.words('english')
    s = nltk.stem.SnowballStemmer('english')
    DISCARD_POS = ['CC', 'DT', 'EX', 'IN','JJ','JJR','JJS', 'MD','PRP','PRP$','RB','RBR','RBS','WDT','WP','WRB','WP$']


    def __init__(self, train_path):
        self.train_path = train_path


    def __iter__(self):
        complete_line = ''
        i = 0
        with open(self.train_path) as f:
            next(f)
            for line in f:
                # assume there's one document per line, tokens separated by whitespace
                if line == '\n':
                    continue
                line = line.lower()
                #remove code parts , they are not that important
                code_start = line.find("<code>")
                code_end = line.find("</code>")
                if code_start != -1:
                    complete_line += line[:code_start]
                elif code_end != -1:
                    complete_line += line[code_end:]
                else:
                    complete_line += line
                if "\r" not in line:
                    continue
                else:
                    complete_line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', complete_line)
                    split_line = complete_line.split('","')
                    #remove <p> in the beginning and end of the body
                    try:
                        split_line[2] = split_line[2][3:-4]
                    except:

                        print i, split_line
                    complete_line = ''
                    i += 1
                processed_line = []
                ws = [self.s.stem(word) for word in split_line[1].translate(self.table, string.punctuation).split() if word not in self.stopwords and not word.isdigit()]
                #[w[0] for w in nltk.pos_tag(ws) if w[1] not in self.DISCARD_POS]
                processed_line.extend(ws)
                ws = [self.s.stem(word) for word in split_line[2].translate(self.table, string.punctuation).split() if word not in self.stopwords and not word.isdigit()]
                #[w[0] for w in nltk.pos_tag(ws) if w[1] not in self.DISCARD_POS]
                processed_line.extend(ws)

                if i == self.LIMIT:
                    break

                yield processed_line


def build_dict(text_file, file_name):
    '''

        Dictionary building

    '''

    fast_dict = MyDict(text_file)
    #one_percent = fast_dict.LIMIT/100
    dictionary = corpora.Dictionary(fast_dict)
    bad_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq ==1]
    dictionary.filter_tokens(bad_ids) # remove stop words and words that appear only once
    dictionary.compactify() # remove gaps in id sequence after words that were removed
    dictionary.save(file_name) # store the dictionary, for future reference
    return dictionary

def build_corpora(train_path, dict_path,  corpora_path):
    '''
        Load dict, train csv and build-store the corpora

    '''

    print "building corpora with training file %s and dictionary %s , output corpora %s" % (train_path, dict_path, corpora_path)
    d = load_dict(dict_path)
    corpus_memory_friendly = MyCorpus(d, train_path)
    corpora.MmCorpus.serialize(corpora_path, corpus_memory_friendly)
    return corpora

def load_dict(dict_path):
    dictionary = corpora.Dictionary.load(dict_path)
    return dictionary

def load_corpus(corpus_path):
    mm_corpus = MmCorpus(corpus_path)  # Revive a corpus
    return mm_corpus

def load_similarity(index_path):
    return similarities.MatrixSimilarity.load(index_path)

def load_model(model_path):
    lda_model = models.ldamodel.LdaModel.load(model_path)
    return lda_model

def build_model(corpora_path, dict_path, model_path):
    print "building model %s " % model_path
    corpus = load_corpus(corpora_path)
    dictionary = load_dict(dict_path)
    lda_model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, update_every=1, chunksize=10000, passes=1)
    lda_model.save(model_path)

    return lda_model

def build_similarity(model_path, corpus_path, dict_path, index_out_path):
    lda_model = load_model(model_path)
    corpus = load_corpus(corpus_path)
    dictionary = load_dict(dict_path)
    index = similarities.MatrixSimilarity(lda_model[corpus])
    index.save(index_out_path)
    return index


def build_test(train_path, test_path, dict_path, model_path, index_path):
    LIMIT = 1000
    table = string.maketrans("","")
    s = nltk.stem.SnowballStemmer('english')
    stopwords = nltk.corpus.stopwords.words('english')
    sim_index = load_similarity(index_path)
    #sim_index.num_best = 1
    dictionary = load_dict(dict_path)
    lda_model = load_model(model_path)
    complete_line = ''
    test_to_train_map = {}
    test_ids_list = []
    i = 0
    print "loading test file %s" % test_path
    with open(test_path) as f:
        next(f)
        for line in f:
            if line == '\n':
                continue
            complete_line += line
            if "\r" not in line:
                continue
            else:
                complete_line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', complete_line)
                split_line = complete_line.split('","')
                try:
                    split_line[2] = split_line[2][3:-4]
                except:
                    continue
                complete_line = ''
                i += 1
            processed_line = []
            processed_line.extend([s.stem(word) for word in split_line[1].translate(table, string.punctuation).lower().split() if word not in stopwords])# and not word.isdigit()])
            processed_line.extend([s.stem(word) for word in split_line[2].translate(table, string.punctuation).lower().split() if word not in stopwords])# and not word.isdigit()])
            if (i % 10000) == 0:
                print i

            #if i == LIMIT:
            #    break

            vec_bow = dictionary.doc2bow(processed_line)
            vec_lda = lda_model[vec_bow]
            sims = sim_index[vec_lda]
            train_index = np.argmax(sims)
            #train_index = sims[0][0]
            #print train_index
            #best_score = max(sims)
            #train_index = sims.tolist().index(best_score)
            #sims = sorted(enumerate(sims), key=lambda item: -item[1])
            test_id = split_line[0][1:]
            test_to_train_map[test_id] = train_index
            test_ids_list.append(test_id)


    complete_line = ''
    i = 0
    LIMIT = 200000
    tags_list = []
    print "opening train file %s" %s
    counters = Counter()
    with open(train_path) as f_train:
        next(f_train)
        for line in f_train:
            # assume there's one document per line, tokens separated by whitespace
            if line == '\n':
                continue
            complete_line += line
            if "\r" not in line:
                continue
            else:
                split_line = complete_line.split('","')
                complete_line = ''
                i += 1
            tags = '"' + split_line[3][:-1]
            if (i % 10000) == 0:
                print i
            if i == LIMIT:
                break
            for t in tags[1:-2].split(' '):
                counters[t] += 1
            tags_list.append(tags)
    one_percent_tags = len(counters)/100
    most_common_90_percent = set([tag for (tag, freq) in counters.most_common(len(counters) - one_percent_tags)])
    f = open('test_result.csv',"w+")
    f.write("Id,Tags\n")
    for test_id in test_ids_list:
        tags = (tags_list[test_to_train_map[test_id]])[1:-2].split(" ")
        common_tags = []
        for t in tags:
            if t in most_common_90_percent:
                common_tags.append(t)
        if len(common_tags)==1 and common_tags[0]=='"':
            common_tags = ['java']
        f.write('%s,"%s"\n' %(test_id, ' '.join(common_tags)))
        #f.write('%s,"%s"\n' %(test_id, tags_list[test_to_train_map[test_id]]))

    print "finished writing"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dict", nargs="+", help="build dictionary train_path dict_out_path")
    parser.add_argument("--build-corpora", nargs="+",  help="build corpora : train_path dict_path corpora_out_path")
    parser.add_argument("--build-model", nargs="+",  help="build a model : corpora_path dict_path model_out_path")
    parser.add_argument("--build-similarity", nargs="+",  help="")
    parser.add_argument("--build-test", nargs="+",  help="")
    args = parser.parse_args()
    print args
    if args.build_dict:
        traincsv = args.build_dict[0]
        dict_file_name = args.build_dict[1]
        build_dict(traincsv, dict_file_name)
    elif args.build_corpora:
        traincsv = args.build_corpora[0]
        dict_file_name = args.build_corpora[1]
        corpora_path = args.build_corpora[2]
        build_corpora(traincsv, dict_file_name, corpora_path)
    elif args.build_model:
        corpora_path = args.build_model[0]
        dict_path = args.build_model[1]
        model_path = args.build_model[2]
        build_model(corpora_path, dict_path, model_path)
    elif args.build_similarity:
        model_path = args.build_similarity[0]
        corpus_path = args.build_similarity[1]
        dict_path = args.build_similarity[2]
        index_out_path = args.build_similarity[3]
        build_similarity(model_path, corpus_path, dict_path, index_out_path)
    elif args.build_test:
        traincsv = args.build_test[0]
        testcsv = args.build_test[1]
        dict_path = args.build_test[2]
        model_path = args.build_test[3]
        index_path = args.build_test[4]

        build_test(traincsv, testcsv, dict_path, model_path, index_path)



