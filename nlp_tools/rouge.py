from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from collections import Counter
import re

from nltk.stem.porter import PorterStemmer
import numpy as np, scipy.stats as st
import itertools
import sys

stemmer = PorterStemmer()


class Rouge(object):
    def __init__(self, stem=True, use_ngram_buf=False):
        self.N = 2
        self.stem = stem
        self.use_ngram_buf = use_ngram_buf
        self.ngram_buf = {}

    @staticmethod
    def _format_sentence(sentence):
        s = sentence.lower()
        s = re.sub(r"[^0-9a-z]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        return s

    def _create_n_gram(self, raw_sentence, n, stem):
        if self.use_ngram_buf:
            if raw_sentence in self.ngram_buf:
                return self.ngram_buf[raw_sentence]
        res = {}
        sentence = Rouge._format_sentence(raw_sentence)
        tokens = sentence.split(' ')
        if stem:
            # try:  # TODO older NLTK has a bug in Porter Stemmer
            tokens = [stemmer.stem(t) for t in tokens]
            # except:
            #     pass
        sent_len = len(tokens)
        for _n in range(n):
            buf = Counter()
            for idx, token in enumerate(tokens):
                if idx + _n >= sent_len:
                    break
                ngram = ' '.join(tokens[idx: idx + _n + 1])
                buf[ngram] += 1
            res[_n] = buf
        if self.use_ngram_buf:
            self.ngram_buf[raw_sentence] = res
        return res

    def get_ngram(self, sents, N, stem=False):
        if isinstance(sents, list):
            res = {}
            for _n in range(N):
                res[_n] = Counter()
            for sent in sents:
                ngrams = self._create_n_gram(sent, N, stem)
                for this_n, counter in ngrams.items():
                    # res[this_n] = res[this_n] + counter
                    self_counter = res[this_n]
                    for elem, count in counter.items():
                        if elem not in self_counter:
                            self_counter[elem] = count
                        else:
                            self_counter[elem] += count
            return res
        elif isinstance(sents, str):
            return self._create_n_gram(sents, N, stem)
        else:
            raise ValueError

    def find_lcseque(self,s1, s2):
        m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
        d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

        for p1 in range(len(s1)):
            for p2 in range(len(s2)):
                if s1[p1] == s2[p2]:
                    m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                    d[p1 + 1][p2 + 1] = 'ok'
                elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                    m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                    d[p1 + 1][p2 + 1] = 'left'
                else:
                    m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                    d[p1 + 1][p2 + 1] = 'up'
        (p1, p2) = (len(s1), len(s2))
        s = []
        while m[p1][p2]:
            c = d[p1][p2]
            if c == 'ok':
                s.append(s1[p1 - 1])
                p1 -= 1
                p2 -= 1
            if c == 'left':
                p2 -= 1
            if c == 'up':
                p1 -= 1
        s.reverse()
        return ' '.join(s)

    def get_mean_sd_internal(self, x):
        mean = np.mean(x)
        sd = st.sem(x)
        res = st.t.interval(0.95, len(x) - 1, loc=mean, scale=sd)
        return (mean, sd, res)

    def compute_rouge(self, references, systems):
        assert (len(references) == len(systems))

        peer_count = len(references)


        result_buf = {}
        for n in range(self.N):
            result_buf[n] = {'p': [], 'r': [], 'f': []}
        result_buf['L'] = {'p': [], 'r': [], 'f': []}

        for ref_sent, sys_sent in zip(references, systems):
            ref_ngrams = self.get_ngram(ref_sent, self.N, self.stem)
            sys_ngrams = self.get_ngram(sys_sent, self.N, self.stem)
            for n in range(self.N):
                ref_ngram = ref_ngrams[n]
                sys_ngram = sys_ngrams[n]
                ref_count = sum(ref_ngram.values())
                sys_count = sum(sys_ngram.values())
                match_count = 0
                for k, v in sys_ngram.items():
                    if k in ref_ngram:
                        match_count += min(v, ref_ngram[k])
                p = match_count / sys_count if sys_count != 0 else 0
                r = match_count / ref_count if ref_count != 0 else 0
                f = 0 if (p == 0 or r == 0) else 2 * p * r / (p + r)
                result_buf[n]['p'].append(p)
                result_buf[n]['r'].append(r)
                result_buf[n]['f'].append(f)

        res = {}
        for n in range(self.N):
            n_key = 'rouge-{0}'.format(n + 1)
            res[n_key] = {}
            if len(result_buf[n]['p']) >= 50:
                res[n_key]['p'] = self.get_mean_sd_internal(result_buf[n]['p'])
                res[n_key]['r'] = self.get_mean_sd_internal(result_buf[n]['r'])
                res[n_key]['f'] = self.get_mean_sd_internal(result_buf[n]['f'])
            else:
                # not enough samples to calculate confidence interval
                res[n_key]['p'] = (np.mean(np.array(result_buf[n]['p'])), 0, (0, 0))
                res[n_key]['r'] = (np.mean(np.array(result_buf[n]['r'])), 0, (0, 0))
                res[n_key]['f'] = (np.mean(np.array(result_buf[n]['f'])), 0, (0, 0))

        
        for ref_sent, sys_sent in zip(references, systems):    
            alllcs = 0
            alls = 0
            allr = 0
            
            for ref_sents in ref_sent:
                sys_sent = sys_sent.replace('<unknown>', 'unk')
                ref_sents = ref_sents.replace('<unknown>', 'unk')
                ref_sent_token = Rouge._format_sentence(ref_sents).split()
                sys_sent_token = Rouge._format_sentence(sys_sent).split()
                if self.stem:
                    ref_sent_token = [stemmer.stem(t) for t in ref_sent_token]
                    sys_sent_token = [stemmer.stem(t) for t in sys_sent_token]
                lcs=self.find_lcseque(ref_sent_token,sys_sent_token)
                alllcs += len(lcs.split())
                alls += len(sys_sent_token)
                allr += len(ref_sent_token)
                # print(alllcs, alls, allr)
            p = alllcs / alls if alls != 0 else 0
            r = alllcs / allr if allr != 0 else 0
            f = 0 if (p == 0 or r == 0) else 2 * p * r / (p + r)

            result_buf['L']['p'].append(p)
            result_buf['L']['r'].append(r)
            result_buf['L']['f'].append(f)
            
        n_key = 'rouge-L'
        res[n_key] = {}
        if len(result_buf['L']['f']) >= 50:
            res[n_key]['p'] = self.get_mean_sd_internal(result_buf['L']['p'])
            res[n_key]['r'] = self.get_mean_sd_internal(result_buf['L']['r'])
            res[n_key]['f'] = self.get_mean_sd_internal(result_buf['L']['f'])
        else:
            # not enough samples to calculate confidence interval
            res[n_key]['p'] = (np.mean(np.array(result_buf['L']['p'])), 0, (0, 0))
            res[n_key]['r'] = (np.mean(np.array(result_buf['L']['r'])), 0, (0, 0))
            res[n_key]['f'] = (np.mean(np.array(result_buf['L']['f'])), 0, (0, 0))  

        return res

    def print_score(self, references, systems):

        gen = open(systems, 'r', encoding='utf-8')
        ref = open(references, 'r', encoding='utf-8')
    
        gen_corpus = []
        ref_corpus = []

        for g, r in zip(gen, ref):
            gen_corpus.append(g.strip())
            ref_corpus.append([r.strip()])    

        scores = self.compute_rouge(ref_corpus, gen_corpus)
        print("Samples: %4d" %len(gen_corpus))
        print("rouge-1 F1(R/P): %02.2f (%02.2f/%02.2f)" \
            %(scores['rouge-1']['f'][0]*100,\
             scores['rouge-1']['r'][0]*100,\
              scores['rouge-1']['p'][0]*100))
        print("rouge-2 F1(R/P): %02.2f (%02.2f/%02.2f)" \
            %(scores['rouge-2']['f'][0]*100,\
             scores['rouge-2']['r'][0]*100,\
              scores['rouge-2']['p'][0]*100))
        print("rouge-L F1(R/P): %02.2f (%02.2f/%02.2f)" \
            %(scores['rouge-L']['f'][0]*100,\
             scores['rouge-L']['r'][0]*100,\
              scores['rouge-L']['p'][0]*100))
        gen.close()
        ref.close()

    def print_all(self, references, systems):
        gen = open(systems, 'r', encoding='utf-8')
        ref = open(references, 'r', encoding='utf-8')
    
        gen_corpus = []
        ref_corpus = []

        for g, r in zip(gen, ref):
            gen_corpus.append(g.strip())
            ref_corpus.append([r.strip()])    

        scores = self.compute_rouge(ref_corpus, gen_corpus)
        print("Sample #: %4d" %len(gen_corpus))
        print("------------------------------------------")
        print("Rouge-1 Average_F: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-1']['f'][0]*100,\
             scores['rouge-1']['f'][2][0]*100,\
              scores['rouge-1']['f'][2][1]*100))
        print("Rouge-1 Average_R: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-1']['r'][0]*100,\
             scores['rouge-1']['r'][2][0]*100,\
              scores['rouge-1']['r'][2][1]*100))
        print("Rouge-1 Average_P: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-1']['p'][0]*100,\
             scores['rouge-1']['p'][2][0]*100,\
              scores['rouge-1']['p'][2][1]*100))
        print("------------------------------------------")
        print("Rouge-2 Average_F: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-2']['f'][0]*100,\
             scores['rouge-2']['f'][2][0]*100,\
              scores['rouge-2']['f'][2][1]*100))
        print("Rouge-2 Average_R: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-2']['r'][0]*100,\
             scores['rouge-2']['r'][2][0]*100,\
              scores['rouge-2']['r'][2][1]*100))
        print("Rouge-2 Average_P: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-2']['p'][0]*100,\
             scores['rouge-2']['p'][2][0]*100,\
              scores['rouge-2']['p'][2][1]*100))
        print("------------------------------------------")
        print("Rouge-L Average_F: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-L']['f'][0]*100,\
             scores['rouge-L']['f'][2][0]*100,\
              scores['rouge-L']['f'][2][1]*100))
        print("Rouge-L Average_R: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-L']['r'][0]*100,\
             scores['rouge-L']['r'][2][0]*100,\
              scores['rouge-L']['r'][2][1]*100))
        print("Rouge-L Average_P: %02.3f (95-conf.int. %02.3f - %02.3f)" \
            %(scores['rouge-L']['p'][0]*100,\
             scores['rouge-L']['p'][2][0]*100,\
              scores['rouge-L']['p'][2][1]*100))
        print("------------------------------------------")
        gen.close()
        ref.close()

if __name__ == "__main__":
    rouge = Rouge()
    generated_file = sys.argv[1]
    reference_file = sys.argv[2]
    
    gen_corpus = []
    ref_corpus = []
    
    gen = open(generated_file, 'r', encoding='utf-8')
    ref = open(reference_file, 'r', encoding='utf-8')
    for g, r in zip(gen, ref):
        gen_corpus.append(g.strip())
        ref_corpus.append([r.strip()])


    print("Samples: %4d" %len(gen_corpus))
    scores = rouge.compute_rouge(ref_corpus, gen_corpus)
    print("rouge-1 F1(R/P): %02.2f (%02.2f/%02.2f)" %(scores['rouge-1']['f'][0]*100, scores['rouge-1']['r'][0]*100, scores['rouge-1']['p'][0]*100))
    print("rouge-2 F1(R/P): %02.2f (%02.2f/%02.2f)" %(scores['rouge-2']['f'][0]*100, scores['rouge-2']['r'][0]*100, scores['rouge-2']['p'][0]*100))
    print("rouge-L F1(R/P): %02.2f (%02.2f/%02.2f)" %(scores['rouge-L']['f'][0]*100, scores['rouge-L']['r'][0]*100, scores['rouge-L']['p'][0]*100))
    gen.close()
    ref.close()


