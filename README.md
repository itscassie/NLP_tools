# NLP_tools
Compute BLEU and ROUGE scores

(The original rouge tool doesn't support evaluation for chinese corpus) 

## Example
```
import rouge
import bleu
bleu = bleu.BLEU()
rouge = rouge.Rouge()

ref_file = "path_to_reference_file"
system_file = "path_to_predict_file"

# for rouge score (simpified version)
rouge.print_score(ref_file, system_file)

# for complete version of rouge score
rouge.print_all(ref_file, system_file)

# for bleu score
# None, sm1~sm7 denotes smoothing function type
bleu.print_score(ref_file, system_file, "sm3")
bleu.print_score(ref_file, system_file, "sm5")
```

## Sample output

### ROUGE score (simplified version) 
```
Samples: 1761
rouge-1 F1(R/P): 55.59 (55.71/56.02)
rouge-2 F1(R/P): 2.62 (2.61/2.80)
rouge-L F1(R/P): 5.67 (5.78/6.10)
```

### ROUGE score (complete version)
```
Sample #: 1761
------------------------------------------
Rouge-1 Average_F: 55.595 (95-conf.int. 53.321 - 57.868)
Rouge-1 Average_R: 55.709 (95-conf.int. 53.429 - 57.990)
Rouge-1 Average_P: 56.025 (95-conf.int. 53.740 - 58.309)
------------------------------------------
Rouge-2 Average_F: 2.618 (95-conf.int. 1.923 - 3.312)
Rouge-2 Average_R: 2.606 (95-conf.int. 1.906 - 3.307)
Rouge-2 Average_P: 2.797 (95-conf.int. 2.061 - 3.532)
------------------------------------------
Rouge-L Average_F: 5.666 (95-conf.int. 4.696 - 6.635)
Rouge-L Average_R: 5.780 (95-conf.int. 4.783 - 6.778)
Rouge-L Average_P: 6.096 (95-conf.int. 5.055 - 7.137)
------------------------------------------
```

### BLEU score
```
S_FUNC: sm3, BLEU-ALL: 3.819, BLEU-1: 19.825, BLEU-2: 4.018, BLEU-3: 2.308, BLEU-4: 1.379
```
