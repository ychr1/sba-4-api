'''
문제 1
'''

import collections 
import matplotlib.pyplot as plt

num_firends = [100,40,30,30,30,30,30,30,30,30,54,54,
54,54,54,54,54,54,25,3,100,100,100,3,3]

friend_counts = collections.Counter(num_firends)

print('friends:', friend_counts)

xs = range(101)

ys = [friend_counts[x] for x in xs]

plt.bar(xs,ys)
plt.axis([0,101,0,25])
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()


'''
문제 2 컴퓨터 시스템은 텍스트 데이터를 어떻게 수집합니까?

언어는 텍스트 (또는 컴퓨터가 이해할 수있는 문자열)로 공식화됩니다. 
한편 머신 러닝 모델은 실수로 작동합니다. 
텍스트를 수집하려는 방식에 따라 각 관측치를 문서로 유지하거나 
더 작은 토큰으로 나눌 수 있습니다. 토큰의 세분성은 우리의 재량에 달려 있습니다. 
토큰은 단어, 문구 또는 문자 수준에서 생성 될 수 있습니다.

그 후, 임베딩 기술 (예 : 문서 임베딩을위한 tf-idf 또는 토큰 임베딩을위한 GloVe / BERT)을
 활용하여 구조화되지 않은 텍스트를 실수 벡터 (또는 벡터)로 변환 할 수 있습니다.

 언어 데이터를 모델링 할 때주의해야 할 또 하나의주의 사항은 
 모든 이전 및 향후 관측치의 입력 크기가 동일해야한다는 것입니다. 
 텍스트를 토큰으로 나누면 긴 텍스트에 다른 토큰보다 많은 토큰이 포함되는 문제가 발생합니다. 
 해결책은 지정된 입력 크기에 따라 입력을 자르거나 채우는 것입니다.

문제 3 : 텍스트 입력을 사전 처리 할 수있는 방법은 무엇입니까?
대소 문자 표준화 : 텍스트를보다 표준적인 형식으로 줄이는 방법으로 
모든 입력을 같은 대소 문자 (소문자 또는 대문자)로 변환 할 수 있습니다 .

구두점 / 중지 단어 / 공백 / 특수 문자 제거 : 
- 이러한 단어 나 문자가 적절하지 않다고 생각되면 제거하여 
- 피처 공간을 줄일 수 있습니다.

lemmatizing / stemming : 
- 어휘를 더 다듬기 위해 단어를 변형 형태 (예 : 걷기 → 걷기)로 줄일 수도 있습니다.
관련없는 정보 일반화 : 
- 모든 숫자를 토큰으로 바꾸거나 모든 이름을 토큰으로 바꿀 수 있습니다 

문제 4 : 언어 모델링을 위해 인코더-디코더 구조는 어떻게 작동합니까?
인코더-디코더 구조는 기계 번역을 포함한 여러 최신 솔루션을 담당하는 
딥 러닝 모델 아키텍처입니다.

입력 시퀀스는 인코더 로 전달되어 신경망을 사용하여 고정 차원 벡터 표현으로 변환됩니다. 
그런 다음 변환 된 입력은 다른 신경망을 사용하여 디코딩 됩니다. 
그런 다음 이러한 출력은 다른 변환과 softmax 레이어를 거칩니다. 
최종 결과는 어휘에 대한 확률의 벡터입니다. 
이러한 확률에 따라 의미있는 정보가 추출됩니다.

문제 5 :주의 메커니즘은 무엇이며 왜 사용합니까?
이것은 인코더 디코더 질문에 대한 후속 조치였습니다. 
마지막 시간 단계의 출력 만 디코더로 전달되어 이전 시간 단계에서 
학습 한 정보가 손실됩니다. 
이 정보 손실은 더 많은 시간 간격으로 더 긴 텍스트 시퀀스에 대해 복잡해집니다.

주의 메커니즘은 각 시간 단계에서 숨겨진 가중치 의 기능 입니다. 
인코더-디코더 네트워크에서주의를 기울일 때, 
디코더로 전달 된 고정 차원 벡터 는 중간 단계에서 출력 된 모든 벡터 의 함수 가됩니다 .
 
일반적으로 사용되는 두 가지주의 메커니즘은 추가주의 와 곱셈주의입니다. 
이름에서 알 수 있듯이 가산주의 는 가중 합계이며 곱셈주의는 숨겨진 가중치의 가중 곱셈입니다.
훈련 과정동안, 모델은 또한 각 시간 단계의 상대적 중요성을 인식하기 위해 주의 메커니즘에 대한 
가중치를 학습합니다.

문제 6 : One Hot 인코딩에 대해 설명해주세요
원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 
표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다. 이렇게 표현된 벡터를 원-핫 벡터(One-Hot vector)라고 합니다.

원-핫 인코딩을 두 가지 과정으로 정리해보겠습니다.
(1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
(2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 
다른 단어의 인덱스의 위치에는 0을 부여합니다.

문제 7 : TF-IDF 점수는 무엇이며 어떤 경우 유용한가요?

TF(단어 빈도, term frequency)는 특정한 단어가 문서 내에 얼마나 자주 등장하는지를 나타내는 값. 
이 값이 높을수록 문서에서 중요하다고 생각할 수 있다.
하지만 하나의 문서에서 많이 나오지 않고 다른 문서에서 자주 등장하면 
단어의 중요도는 낮아진다.
DF(문서 빈도, document frequency)라고 하며, 
이 값의 역수를 IDF(역문서 빈도, inverse document frequency)라고 한다.
TF-IDF는 TF와 IDF를 곱한 값으로 점수가 높은 단어일수록
 다른 문서에는 많지 않고 해당 문서에서 자주 등장하는 단어를 의미한다.

문제 8 :
카카오는 2018년 말부터 딥러닝 기반 형태소(形態素, morpheme) 분석기 '카이(khaiii)'를 
오픈소스로 제공하고 있다. 딥러닝을 통해 학습한 데이터를 활용해 형태소를 분석하는 모델이다. 
딥러닝 기술 중 하나인 콘볼루션 신경망(CNN, Convolutional Neural Network)을 이용해
음절기반으로 형태소를 분석하는 방법을 채택했다.

세종 코퍼스를 기반으로 데이터의 오류를 수정하고 카카오에서 
자체 구축한 데이터를 추가해 85만 문장, 1003만 어절의 데이터를 학습하여 정확도를 높였다. 
또 딥러닝 과정에서 C++ 언어를 적용해 일반적으로 딥러닝에 쓰이는 GPU(그래픽처리장치)를 
사용하지 않고도 빠른 분석 속도를 구현했다.

[  ?  ] 은 2개 이상의 글자로 이루어진 단어 혹은 문장을 입력 시, 
의미를 가진 언어 단위 중 가장 작은 단위인 형태소 단위로 자동으로 분리하는 기술이다. 
예를 들면, '학교에 간다'라고 입력하면 '학교/명사 + 에/조사 + 가/동사 + ㄴ다/어미' 로 
형태소 단위와 품사를 파악해 분류해내는 기술이다.

답 : 형태소 분석 기술

문제 9 : 전처리에서 사용되는 용어
자연어처리의 전처리는 머신러닝에서 피처를 1차로 정하는 것이다.
전처리 과정에서 등장하는 다음 용어들은 모두 피처를 설명하기 위해 만들어진 것이다.
다만, 각자가 사용되는 맥락에 따라서 비슷한 개념을 다르게 표현하였다.
단어 word : 인간이 데이터를 인지할 때 사용하는 피처 혹은 개념이다.
형태소 morpheme : 단어에 대한 상대적인 개념으로 , 합성어와 가티이 여러 형태소로 
이루어진 단어를 설명하고자 만들어졌다.
서브워드: 형태소와 비슷한 개념으로 어던 워드가 다수의 서브워드(피처)로 이뤄진 것을 표현하고자 사용한다.
토큰: 단어나 형태소를 컴퓨터에 입력하고자 컴퓨터 내부 공간에 할당한 개념
엔티티(독립체): 지극히 서양적인 사고에서 나온 것으로 더 이상 쪼갤 수 없다는 원자적 관점에서
파생된 개념이다. 사건이 아니라 사물 중심으로 만들어진 것으로, 미래엔 사라질 개념이다.
어절(띄워쓰기 단위): 한국어와 영어는 띄어쓰기 개념이 다르다. 한국어는 단어의 문법적 기능을 표현하고자
조사를 사용하며 이를 단어에 붙이지만, 영어는 전치사를 따로 떼어 사용한다.

전처리란, 자연어를 token 단위로 쪼개서 입력 처리하는 것이다.
가령 RNN 으로 문장을 처리한다고 할 때, RNN 의 매 time step 마다 입력되는
단위가 token 이다. 
token 이 될 수 있는 것은 word, subword, morpheme, character 등이다.
즉, 정하기 나름이다.
자연에 처리에서 token을 무엇으로 규정하느냐는 곧 이미지처리에서 feature 를 어떤 것으로
규정하느냐와 같은 문제이다.


문제에 따라 피처가 상대적으로 규정되는 것을 맥락에 따른 규정이라고 한다.
머신러닝은 피처 중심(과거) 에서 문제 중심(미래)로 변화고 있다.
딥러닝 방법론으로 자연어 처리를 자동화하는 것은 좀 더 맥락적이라 할 수 있다.

문제 10: 	
한글을 이용한 데이터마이닝및 word2vec이용한 유사도 분석

1.  prototyping에 python만한게 없는 것 같다.파이선설치 확인



2. python용 버전관리 소프트 pip설치(Mac기반, 다른 OS라도 python만 돌아간다면야.)

sudo easy_install pip



3. gensim 설치 ( Gensim 이라는 파이선 기반의 text mining library를 다운받는다. 토픽 모델링및 ,word2vec도 지원한다.)

sudo pip install -U gensim



4. NLTK설치(자연어 처리를 위한 광범위하게 쓰이는 python lib)

sudo pip install nltk



5. KoNLPy설치 (한글처리를 위해)

sudo pip install konlpy



6. twython 설치 (twitter api쉽게 사용하기위해)

sudo pip install twython
1. 읽기

$ python

from konlpy.corpus import kobill    # Docs from pokr.kr/bill

files_ko = kobill.fileids()         # Get file ids

doc_ko = kobill.open('news.txt').read() 

# news.txt는 http://boilerpipe-web.appspot.com/ 를 통해 포탈뉴스 부분에서 긁어왔다.

# news.txt 는  konlpy의 corpus아래에 있는 kobill directory에 미리 저장되어있어야 한다. 

# /Library/Python/2.7/site-packages/konlpy/data/corpus/kobill



2.Tokenize (의미단어 검출)

from konlpy.tag import Twitter; t = Twitter()

tokens_ko = t.morphs(doc_ko)



3. Token Wapper 클래스 만들기(token에대해 이런 저런 처리를 하기 위해)

import nltk

ko = nltk.Text(tokens_ko, name='뉴스')   # 이름은 아무거나



4. 토근 정보및 단일 토큰 정보 알아내기

print(len(ko.tokens))       # returns number of tokens (document length)

print(len(set(ko.tokens)))  # returns number of unique tokens

ko.vocab()                  # returns frequency distribution



5. 챠트로 보기

ko.plot(50) #상위 50개의 unique token에 대해서 plot한결과를 아래와 같이 보여준다.
분석은 굉장히 쉽다. 왼쪽부터 잦은 빈도로 출현한 단어들이다. ...과 같은 의미없는 특수문자들을 제외하면 요즘 유행하는 메르스가 역시 가장 높은 빈도수(약 30건) 정도를 보여주고 있다. 적절한 토큰나이저로 특수문자들을 걸러준다면 의미있는 결과를 얻어내기 용이할 것이라고 본다.



6. 특정 단어에 대해서 빈도수 확인하기

print(ko.count(str('메르스'))) 



결과)

18



위처럼 하면 메르스에 대한 숫자를 준다. 만약에 제대로 글자를 인식 못한다면, 파일의 앞에 아래처럼 디폴트 인코딩을 세팅해준다.

#!/usr/bin/env python

# -*- coding: utf-8 -*-

import sys

reload(sys)

sys.setdefaultencoding('utf-8')



7. 분산 차트 보기 (dispersion plot)

ko.dispersion_plot(['메르스','학교','병원'])
https://blog.naver.com/2feelus/220384206922

https://blog.naver.com/2feelus/220384206922

Word2Vec 은 2013년 구글 논문으로 
nn 기반으로 대량의 문서 데이터 셋을 벡터 공간에 고수준의 
벡터를 가지도록 효율적으로 Word 의 벡터 값을 추정하는 기계학습 모델이다.



문제 11 : 
 텍스트를 읽고 긍정, 부정 예측하기
 https://anpigon.github.io/blog/kr/@anpigon/4/

실습해 볼 과제는 영화를 보고 남긴 리뷰를 딥러닝 모델로 학습해서, 
각 리뷰가 긍정적인지 부정적인지를 예측하는 것입니다.

먼저 짧은 리뷰 10개를 불러와 각각 긍정이면 1이라는 클래스를, 
부정적이면 0이라는 클래스로 지정합니다.



'''
# 텍스트 리뷰 자료 지정
from konlpy.tag import Mecab
pos_tagger = Mecab()
from textblob.classifiers import NaiveBayesClassifier
train = [
    ('나는 이 샌드위치를 정말 좋아해.', '긍정'),
    ('정말 멋진 곳이에요!', '긍정'),
    ('나는 이 맥주들이 아주 좋다고 생각해요.', '긍정'),
    ('이것은 나의 최고의 작품입니다.', '긍정'),
    ("정말 멋진 광경이다", "긍정"),
    ('난 이 식당 싫어', '부정'),
    ('난 이게 지겨워.', '부정'),
    ("이 문제는 처리할 수 없습니다.", "부정"),
    ('그는 나의 불구대천의 원수이다.', '부정'),
    ('내 상사는 끔찍해.', '부정')
]

test = [
    ('맥주가 좋았습니다.', '긍정'),
    ('난 내 일을 즐기지 않는다', '부정'),
    ('오늘은 기분이 안 좋아요.', '부정'),
    ('놀라워요!', '긍정'),
    ('네드는 나의 친구입니다.', '긍정'),
    ('제가 이렇게 하고 있다니 믿을 수가 없어요.', '부정')
]

# 긍정 리뷰는 1, 부정 리뷰는 0으로 클래스 지정
train_data = [(['/'.join(token) for token in pos_tagger.pos(sentence)], result) for [sentence, result] in train]
train_data
# 토큰화
test_data = [(['/'.join(token) for token in pos_tagger.pos(sentence)], result) for [sentence, result] in test]
cl = NaiveBayesClassifier(train)
cl.show_informative_features()
print('============== Accuracy ==============')
print(cl.accuracy(test_data))
cl.show_informative_features()