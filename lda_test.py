import codecs as cd
import gensim
from janome.tokenizer import Tokenizer
from gensim import corpora, models, similarities

#dataは1文書の文字が１行に収められているテキストファイル
#A文書の内容が1行目、B文書の内容は２行目というようなかんじになっている
filename = 'data.txt'
file = cd.open(filename,  'r', 'utf-8')
lines = file.readlines()

t = Tokenizer()
wvs = []

for i, line in enumerate(lines):
  # 一つのページのワードのベクトル
  word_vector = []

  # 短すぎる場合は無視
  # if len(line)<30:
  #     continue
  # 記号以外はベクトル作成
  # else:
  tokens = t.tokenize(line)

  for token in tokens:
      if token.part_of_speech[:2] == '名詞':
          word_vector += [token.base_form]

  # データを連結
  wvs += [word_vector]

#各文書の抜き出した文言のリストのリスト
# print(wvs)
# 辞書作成
dictionary = corpora.Dictionary(wvs)
dictionary.filter_extremes(no_below=1, no_above=0.3)
dictionary.save_as_text('dict.txt')

# コーパスを作成
corpus = [dictionary.doc2bow(text) for text in wvs]
corpora.MmCorpus.serialize('cop.mm', corpus)

dictionary = gensim.corpora.Dictionary.load_from_text('dict.txt')
corpus = corpora.MmCorpus('cop.mm')

#トピック数を制御
topic_N = 10
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=topic_N, id2word=dictionary)

for i in range(topic_N):
    print('TOPIC:', i, '__', lda.print_topic(i))










# filename = 'test.txt'
# file = cd.open(filename,  'r', 'utf-8')
# lines = file.readlines()

# t = Tokenizer()
# wvs = []

# for i, line in enumerate(lines):
#   # 一つのページのワードのベクトル
#   word_vector = []

#   # 短すぎる場合は無視
#   # if len(line)<30:
#   #     continue
#   # 記号以外はベクトル作成
#   # else:
#   tokens = t.tokenize(line)

#   for token in tokens:
#       if token.part_of_speech[:2] == '名詞':
#           word_vector += [token.base_form]

#   # データを連結
#   wvs += [word_vector]

# #各文書の抜き出した文言のリストのリスト
# # print(wvs)
# # 辞書作成
# dictionary = corpora.Dictionary(wvs)
# dictionary.filter_extremes(no_below=1, no_above=0.3)
# dictionary.save_as_text('dict.txt')

# # コーパスを作成
# corpus = [dictionary.doc2bow(text) for text in wvs]
# corpora.MmCorpus.serialize('cop.mm', corpus)

# dictionary = gensim.corpora.Dictionary.load_from_text('dict.txt')
# corpus = corpora.MmCorpus('cop.mm')
# print(lda.log_perplexity(corpus, total_docs = None))

#ここで出力されるものは
#文章中のトピックの値をしゅつりょくしている
#つまり出たトピック数をそのまま特徴ベクトルとすればよい

#slideshereに参考元がある
# LDA におけるトピック数の決定法 
# 1.  データを学習⽤用、テスト⽤用に分ける 
# 2.  特定のトピック数を⽤用いて LDA を学習し、 テストデータで  Perplexity を求める 
# 3.  LDA 学習時に必要な初期値を変えて学習を 繰り返し、Perplexity の平均を求める 
# 4.  トピック数で⽐比較し、最も良いものを選ぶ 

#コード自体の参考サイト
# http://paper.hatenadiary.jp/entry/2016/11/06/212149