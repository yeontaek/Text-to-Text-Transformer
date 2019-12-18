# Text-to-Text Transformer

본 repository에서는 Google의 [T5(T5: Text-To-Text Transfer Transformer)](https://arxiv.org/abs/1910.10683)의 text-to-text 형태로 한국어 QA Task를 해결하고자 한다.

* Text-to-Text Transformer-Base, Korean Model: 12-layer, 768-hidden, 12-heads(비공개)
* Text-to-Text Transformer-Small, Korean Model: 6-layer, 512-hidden, 8-heads(비공개)

## Pre-training

### 1.Objective 



<img src = "https://yhdosu.github.io/assets/images/T5/T5_6.png" width=80%>

### 2.문장 구성

```
Input 문장 :


Target 문장 : 
```


### 3.Use TPU


### 4.Parameter

> Base This is our baseline model, whose hyperparameters are described in Section 3.1.1. It has roughly 220million parameters.
> Small. We consider a smaller model, which scales the baseline down by using dmodel= 512, dff= 2,048, 8-headed attention, and only 6layers each in the encoder and decoder. This varianthas about 60million parameters.


## Fine-Tuning(QA Task)




## 성능 비교




## Reference

* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)<br>
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>
* [Chatbot using Tensorflow (Model is transformer) ko](https://github.com/changwookjun/Transformer)<br>
* [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)
