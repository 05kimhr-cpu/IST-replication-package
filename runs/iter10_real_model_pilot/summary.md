# Iter 10 — Real-model realism pilot (CodeLlama-7B-Instruct on MCMD)

Model: <model_path>
Samples: 80 (10 per language × 8 langs)
Gold token floor: 8

### Overall  (n=80)
- bleu            mean=+0.053  median=+0.031
- rougeL          mean=+0.153  median=+0.123
- chrf            mean=+0.239  median=+0.218
- meteor          mean=+0.082  median=+0.049
- bertscore_f1    mean=+0.860  median=+0.859
- nli_signed      mean=+0.038  median=+0.001
- nli_entail      mean=+0.127  median=+0.026
- NLI signed >= +0.56 (paper balanced op point):   6/80  (7.5%)
- NLI signed >= +0.88 (paper conservative op pt):  4/80  (5.0%)

## Per-language

### cpp  (n=10)
- bleu            mean=+0.027  median=+0.016
- rougeL          mean=+0.153  median=+0.133
- chrf            mean=+0.230  median=+0.220
- meteor          mean=+0.092  median=+0.062
- bertscore_f1    mean=+0.850  median=+0.853
- nli_signed      mean=+0.122  median=+0.000
- nli_entail      mean=+0.189  median=+0.013
- NLI signed >= +0.56 (paper balanced op point):   2/10  (20.0%)
- NLI signed >= +0.88 (paper conservative op pt):  1/10  (10.0%)

### cs  (n=10)
- bleu            mean=+0.029  median=+0.029
- rougeL          mean=+0.079  median=+0.073
- chrf            mean=+0.196  median=+0.194
- meteor          mean=+0.037  median=+0.035
- bertscore_f1    mean=+0.854  median=+0.849
- nli_signed      mean=-0.079  median=+0.001
- nli_entail      mean=+0.049  median=+0.034
- NLI signed >= +0.56 (paper balanced op point):   0/10  (0.0%)
- NLI signed >= +0.88 (paper conservative op pt):  0/10  (0.0%)

### go  (n=10)
- bleu            mean=+0.028  median=+0.018
- rougeL          mean=+0.116  median=+0.103
- chrf            mean=+0.178  median=+0.115
- meteor          mean=+0.077  median=+0.044
- bertscore_f1    mean=+0.844  median=+0.843
- nli_signed      mean=+0.004  median=-0.002
- nli_entail      mean=+0.033  median=+0.014
- NLI signed >= +0.56 (paper balanced op point):   0/10  (0.0%)
- NLI signed >= +0.88 (paper conservative op pt):  0/10  (0.0%)

### java  (n=10)
- bleu            mean=+0.028  median=+0.031
- rougeL          mean=+0.112  median=+0.101
- chrf            mean=+0.271  median=+0.269
- meteor          mean=+0.057  median=+0.034
- bertscore_f1    mean=+0.856  median=+0.861
- nli_signed      mean=-0.079  median=-0.024
- nli_entail      mean=+0.077  median=+0.016
- NLI signed >= +0.56 (paper balanced op point):   0/10  (0.0%)
- NLI signed >= +0.88 (paper conservative op pt):  0/10  (0.0%)

### js  (n=10)
- bleu            mean=+0.032  median=+0.016
- rougeL          mean=+0.093  median=+0.093
- chrf            mean=+0.202  median=+0.180
- meteor          mean=+0.063  median=+0.040
- bertscore_f1    mean=+0.849  median=+0.846
- nli_signed      mean=-0.031  median=+0.018
- nli_entail      mean=+0.066  median=+0.049
- NLI signed >= +0.56 (paper balanced op point):   0/10  (0.0%)
- NLI signed >= +0.88 (paper conservative op pt):  0/10  (0.0%)

### php  (n=10)
- bleu            mean=+0.110  median=+0.043
- rougeL          mean=+0.281  median=+0.226
- chrf            mean=+0.319  median=+0.247
- meteor          mean=+0.155  median=+0.087
- bertscore_f1    mean=+0.884  median=+0.881
- nli_signed      mean=+0.199  median=+0.059
- nli_entail      mean=+0.251  median=+0.066
- NLI signed >= +0.56 (paper balanced op point):   2/10  (20.0%)
- NLI signed >= +0.88 (paper conservative op pt):  1/10  (10.0%)

### py  (n=10)
- bleu            mean=+0.147  median=+0.046
- rougeL          mean=+0.241  median=+0.200
- chrf            mean=+0.318  median=+0.295
- meteor          mean=+0.095  median=+0.079
- bertscore_f1    mean=+0.875  median=+0.870
- nli_signed      mean=+0.040  median=+0.035
- nli_entail      mean=+0.172  median=+0.053
- NLI signed >= +0.56 (paper balanced op point):   1/10  (10.0%)
- NLI signed >= +0.88 (paper conservative op pt):  1/10  (10.0%)

### rust  (n=10)
- bleu            mean=+0.024  median=+0.025
- rougeL          mean=+0.149  median=+0.125
- chrf            mean=+0.198  median=+0.212
- meteor          mean=+0.076  median=+0.066
- bertscore_f1    mean=+0.865  median=+0.862
- nli_signed      mean=+0.129  median=+0.014
- nli_entail      mean=+0.181  median=+0.021
- NLI signed >= +0.56 (paper balanced op point):   1/10  (10.0%)
- NLI signed >= +0.88 (paper conservative op pt):  1/10  (10.0%)

