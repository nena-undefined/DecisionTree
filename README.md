# DecisionTree

## Description

こちらのサイトの決定木のコードを参考に，簡易化+コメント追加+回帰問題に対応したものです．
http://darden.hatenablog.com/entry/2016/12/15/222447

研究室の後輩向けに実装しました．
何か問題や不明点などありましたらご連絡ください．

## Requirement

開発環境はanaconda3-5.3.1を使用しています． anaconda3-5.3.1において，今回の実装で使用しているパッケージなどは 以下の通りです．


* Python3.7
* numpy 1.15.1
* scipy 1.1.0
* scikit-learn 0.19.2

## Usage

今回実装したモデルは回帰木のDecisionTreeRと分類木のDecisionTreeCです．

DecisionTree.pyと同じディレクトリで

```
from DecisionTree import DecisionTreeR, DecisionTreeC
```

とするとモデルをimportすることができます．
sklearn準拠モデルとして実装しましたので，GridSearchなども行えます．

簡単な使い方はexample.ipynbを見てください．

## 高速化について

この実装では，分割点を評価（情報利得を計算）する際に毎回計算し直しています．
実は毎回計算する必要はなく，うまく実装すると定数時間で評価を行うことができます．
ぜひ実装して見てください．
DecisionTree_fast.pyに実装したコードがあります（わからなかったら私に聞いてください）
