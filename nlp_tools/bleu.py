from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from statistics import mean
import sys


global sm_func
sm_func = {}
sm_func["None"] = None
sm_func["sm1"] = SmoothingFunction().method1
sm_func["sm2"] = SmoothingFunction().method2
sm_func["sm3"] = SmoothingFunction().method3
sm_func["sm4"] = SmoothingFunction().method4
sm_func["sm5"] = SmoothingFunction().method5
sm_func["sm6"] = SmoothingFunction().method6
sm_func["sm7"] = SmoothingFunction().method7


class BLEU(object):
    

    def compute_bleu(self, refs, systems, sm):
        bleu_1 = []
        bleu_2 = []
        bleu_3 = []
        bleu_4 = []
        bleu_all = []
        for i in range(len(systems)):
            B1 = sentence_bleu(refs[i].split(), systems[i].split(), weights = (1, 0, 0, 0), smoothing_function=sm)
            bleu_1.append(float(B1))
            B2 = sentence_bleu(refs[i].split(), systems[i].split(), weights = (0, 1, 0, 0), smoothing_function=sm)
            bleu_2.append(float(B2))
            B3 = sentence_bleu(refs[i].split(), systems[i].split(), weights = (0, 0, 1, 0), smoothing_function=sm)
            bleu_3.append(float(B3))
            B4 = sentence_bleu(refs[i].split(), systems[i].split(), weights = (0, 0, 0, 1), smoothing_function=sm)
            bleu_4.append(float(B4))    
            BA = sentence_bleu(refs[i].split(), systems[i].split(), smoothing_function=sm)
            bleu_all.append(float(BA))   
        return mean(bleu_1), mean(bleu_2), mean(bleu_3), mean(bleu_4), mean(bleu_all)

    def print_score(self, references, systems, sm):

        gen = open(systems, 'r', encoding='utf-8')
        ref = open(references, 'r', encoding='utf-8')
    
        gen_corpus = []
        ref_corpus = []

        for g, r in zip(gen, ref):
            gen_corpus.append(g.strip())
            ref_corpus.append(r.strip())   
        

        b1, b2, b3, b4, ba = self.compute_bleu(ref_corpus, gen_corpus, sm_func[sm])
        print("------------------------------------------")
        print("S_FUNC: %s, BLEU-ALL: %02.3f, BLEU-1: %02.3f, BLEU-2: %02.3f, BLEU-3: %02.3f, BLEU-4: %02.3f" \
            %(sm, ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))
        gen.close()
        ref.close()

if __name__ == "__main__":
    bleu = BLEU()
    generated_file = sys.argv[1]
    reference_file = sys.argv[2]
    sm = sys.argv[3]

    gen = open(generated_file, 'r', encoding='utf-8')
    ref = open(reference_file, 'r', encoding='utf-8')

    gen_corpus = []
    ref_corpus = []

    for g, r in zip(gen, ref):
        gen_corpus.append(g.strip())
        ref_corpus.append(r.strip())   
    

    b1, b2, b3, b4, ba = bleu.compute_bleu(ref_corpus, gen_corpus, sm_func[sm])
    print("------------------------------------------")
    print("S_FUNC: %s, BLEU-ALL: %02.3f, BLEU-1: %02.3f, BLEU-2: %02.3f, BLEU-3: %02.3f, BLEU-4: %02.3f" \
        %(sm, ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))
    gen.close()
    ref.close()