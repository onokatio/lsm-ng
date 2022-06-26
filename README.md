最小二乗法 プログラム
===


`data/lsmCompe_test.csv`のデータ(x,yをペアとしたcsvデータ)を読み込み、最小二乗法で学習し、最終的に多項式のパラメータを出力します。

# 詳細

1. まず、入力されたデータから外れ値を取り除きます。xを100のブロックごと分割し、平均±標準偏差*2より外の範囲に居る値を外れ値としています。

2. その後leave-one-out交差検証（k=データ数のk交差検証）で学習データと検証データに分割します。

3. その後、学習データで正則化最小二乗法（リッジ回帰）を行い、そのパラメータで検証データを予測しRMSEを計算します。

4. これで計算したk個のRMSEの標準偏差と平均を求め、また最も小さいRMSEを出力したパラメータを保管します。

5. 以上 1. - 4. の操作を複数回繰り返し、最もRMSEが小さく標準偏差も小さいパラメータを選択し、最終的なパラメータとします。

6. 最終的なパラメータを、入力データに対して適用し、保存します。

# その他

ライブラリは、numpy, cupy, sklearnを用いました。

ハイパーパラメータ（次元やkや正則化項）は、5.の平均RMSEが最も小さくなるように、一定の範囲の元総当りをし、グラフ可視化をし、選択しました。
具体的には、次元=20、k=データの数(leave-one-out)、λ=0です。