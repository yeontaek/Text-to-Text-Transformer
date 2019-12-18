# Text-to-Text Transformer

본 repository에서는 Google의 [T5(T5: Text-To-Text Transfer Transformer)](https://arxiv.org/abs/1910.10683)의 text-to-text 형태로 한국어 QA Task를 해결하고자 한다. 전체 모델의 아키텍처는 기존 Transformer 모델을 사용했다. 

* Text-to-Text Transformer-Base, Korean Model: 12-layer, 768-hidden, 12-heads(비공개)
* Text-to-Text Transformer-Small, Korean Model: 6-layer, 512-hidden, 8-heads(비공개)

<img src = "https://github.com/changwookjun/Transformer/raw/master/images/1.png" width=70%>


## Pre-training

### 1.Unsupervised objective

T5 논문에서 가장 성능이 잘 나온다고 서술된 BERT Style Objective로 문장을 구성하여 학습을 진행했다. BERT와 동일하게 입력 문장의 15%를 Random 하게 마스킹 처리했다. 마스킹 대상의 80%는 <MASK>토큰으로 대체하며, 10%는 사전 내 임의의 토큰으로 나머지 10%는 원래의 단어를 그대로 사용했다.

<img src = "https://yhdosu.github.io/assets/images/T5/T5_6.png" width=80%>
<img src = "https://yhdosu.github.io/assets/images/T5/T5_7.png" width=80%>


### 2.문장 구성

```
Input 문장 : 


Target 문장 : 
```


### 3.Unlabeld dataset
학습 데이터는 **한국어 위키데이터(2019.01 dump file 기준, 약 350만 문장)** 을 사용하여 학습을 진행했다. 
 
~~~
라 토스카(La Tosca)는 1887년에 프랑스 극작가 사르두가 배우 사라 베르나르를 위해 만든 작품이다.
1887년 파리에서 처음 상연되었다.
1990년 베르나르를 주인공으로 미국 뉴욕에서 재상연되었다.
1800년 6월 중순의 이탈리아 로마를 배경으로 하며, 당시의 시대적 상황 하에서 이야기가 전개된다.
1900년, 사르두의 연극은 푸치니의 오페라 토스카로 새롭게 각색되었다.
베르디는 사드루의 각본에서 "갑작스런 종결" 부분을 수정할 것을 권하지만, 사르루는 이를 거절한다.
후에, 푸치니 또한 사르두의 각본에서 "갑작스런 종결부분"을 수정할 것을 제안하지만 끝내 사르두를 설득하지 못했다.

2008년 하계 올림픽의 복싱 남자 라이트급 종목은 8월 11일일부터 8월 24일까지 중화인민공화국의 베이징에 있는 베이징 노동자 체육관에서 열렸다.
27개국에서 27명의 선수가 참가하였다.
2008년 하계 올림픽 복싱 남자 라이트급 경기는 개최 도시인 베이징에 있는 베이징 노동자 체육관에서 경기가 열렸다.

도리데 시는 일본 이바라키현의 남부에 있는 시이다.
간토 평야에 위치하고 도네 강과 고카이 강에 접하고 있다.
이 때문인지 일찍이 수해가 많았다.
현재에도 시 남서부의 대지를 제외하면 시역이 많은 부분이 침수의 위험성이 있다.
그러나 최근에 도네 강, 고카이 강 등의 제방의 고기능화에 의해 하천의 범람에 의한 침수 피해는 거의 없어졌다.
한편 집중호우에 의해 시내의 저지 등에서는 도로가 일부 침수하는 등의 피해가 일어난다.
~~~


### 4.Parameter

> Base This is our baseline model, whose hyperparameters are described in Section 3.1.1. It has roughly 220million parameters.
> Small. We consider a smaller model, which scales the baseline down by using dmodel= 512, dff= 2,048, 8-headed attention, and only 6layers each in the encoder and decoder. This varianthas about 60million parameters.


## Fine-Tuning

### 1.QA Task


## Requirement




## Reference

* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)<br>
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>
* [Chatbot using Tensorflow (Model is transformer) ko](https://github.com/changwookjun/Transformer)<br>
* [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)

