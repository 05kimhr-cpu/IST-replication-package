# Iteration 4 — BERTScore paired perturbation

- model: roberta-large, device: cuda
- total pairs: 1630

### Overall (all languages)
- n=1630
- mean=-0.0000 median=+0.0000 stdev=0.0086
- mean|x|=0.0051
- exact_zero=0/1630  near_zero(|x|<0.01)=1357/1630
- sign: syn>ant: 825/1630   syn<ant: 805/1630

## Per language

### cpp
- n=162
- mean=-0.0003 median=-0.0001 stdev=0.0086
- mean|x|=0.0056
- exact_zero=0/162  near_zero(|x|<0.01)=133/162
- sign: syn>ant: 74/162   syn<ant: 88/162

### cs
- n=201
- mean=-0.0016 median=-0.0002 stdev=0.0094
- mean|x|=0.0061
- exact_zero=0/201  near_zero(|x|<0.01)=155/201
- sign: syn>ant: 84/201   syn<ant: 117/201

### go
- n=217
- mean=+0.0009 median=+0.0003 stdev=0.0071
- mean|x|=0.0038
- exact_zero=0/217  near_zero(|x|<0.01)=195/217
- sign: syn>ant: 128/217   syn<ant: 89/217

### java
- n=190
- mean=+0.0003 median=+0.0001 stdev=0.0080
- mean|x|=0.0052
- exact_zero=0/190  near_zero(|x|<0.01)=161/190
- sign: syn>ant: 102/190   syn<ant: 88/190

### js
- n=211
- mean=-0.0009 median=+0.0000 stdev=0.0090
- mean|x|=0.0060
- exact_zero=0/211  near_zero(|x|<0.01)=165/211
- sign: syn>ant: 106/211   syn<ant: 105/211

### php
- n=224
- mean=+0.0004 median=+0.0000 stdev=0.0078
- mean|x|=0.0049
- exact_zero=0/224  near_zero(|x|<0.01)=183/224
- sign: syn>ant: 115/224   syn<ant: 109/224

### py
- n=207
- mean=+0.0001 median=-0.0001 stdev=0.0110
- mean|x|=0.0055
- exact_zero=0/207  near_zero(|x|<0.01)=174/207
- sign: syn>ant: 101/207   syn<ant: 106/207

### rust
- n=218
- mean=+0.0009 median=+0.0001 stdev=0.0070
- mean|x|=0.0039
- exact_zero=0/218  near_zero(|x|<0.01)=191/218
- sign: syn>ant: 115/218   syn<ant: 103/218

