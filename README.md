研究
====

```
.
├── conv.py      : 学習用スクリプト
├── images       : 画像用ディレクトリ
├── kis.js       : KIS Download Client (要 secret.json)
├── package.json : for kis.js
├── resize.py    : 画像リサイズ (要 OpenCV)
├── samples.py   : 学習画像の選択
├── secret.json  : for kis.js
└── test.py      : 判定用スクリプト
```

流れ
----
1. KIS Server から画像をダウンロード (kis.js)
1. 64x64px に縮小 (resize.py)
1. 学習 (conv.py)
1. 作成された model.ckpt を用いて判定 (test.py)
