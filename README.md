## 岡本達史（おかもとさとし）の公開フォルダです。
私のスキルの参考としてこのフォルダを作っております。
すべて自作のコードですので、ご自由にご利用ください。

# ファイル一覧
### 詳細は下記各ファイルの段落をご覧ください。
- ディープラーニングによる株価予測コード（Notebook）  
　　"stock_price_prediction.ipynb" [詳細](stock_price_prediction.ipynb)  
　　　[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazakamibeer/public/blob/main/stock_price_prediction.ipynb)　[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kazakamibeer/public/blob/main/stock_price_prediction.ipynb)
- ディープラーニングによる株価予測コード（Python）  
　　"stock_price_prediction.py" [詳細](stock_price_prediction.py)
- Kaggleデータ分析処理コード（Notebook）  
　　"titanic-a-set-of-fundamental-analyses.ipynb" [詳細](titanic-a-set-of-fundamental-analyses.ipynb)  
　　　[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazakamibeer/public/blob/main/titanic-a-set-of-fundamental-analyses.ipynb)　[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kazakamibeer/public/blob/main/titanic-a-set-of-fundamental-analyses.ipynb)
- Kaggleデータ分析処理コード（Python）  
　　"titanic_code_only.py" [詳細](titanic_code_only.py)
- LINE・スプレッドシート連携アプリコード（JavaScript, GAS）  
　　"LINEから見積書.txt"[詳細](LINEから見積書.txt)  
- ExcelVBAビッグデータ処理コード（VBA）  
　　"vba_code_only.txt"[詳細](vba_code_only.txt)
- OutlookVBAメール自動返信コード（VBA）  
　　"問い合わせメール自動返信vba.txt"[詳細](問い合わせメール自動返信vba.txt) 
- Java掲示板アプリのコード（Java）  
　　"掲示板アプリプロジェクト"[詳細](掲示板アプリプロジェクト)
  
# ディープラーニングによる株価予測コード（Notebook）
未完成（40%）  
このNotebookでは株価など時系列データの未来予測をしています。  
いずれは自動売買の予想モデルとして使用したいと考えております。  
#### 論文を参考にした以下の５つのモデルでそれぞれ予測を行う
- LSTM
- Bidirectional LSTM + Attention
- Transformer
- Informer
- Autoformer
#### ５つのモデルの精度を考慮しつつ、結果をアンサンブルする
上記５モデルが未完成のため、現状は机上の空論のみで簡易なコードを作成。  
５モデルの出力に加えて、相関・逆相関のある金融の指数を特徴量として加えることを予定。  
アンサンブルのメタモデルには、学習性能を弱めたRandomForestを使用予定。  
| 内容 | ファイル名 | Colab | Kaggle |
| :-- | :-- | :-- | :-- | 
| ディープラーニングによる株価予測Notebook | [詳細](stock_price_prediction.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazakamibeer/public/blob/main/stock_price_prediction.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kazakamibeer/public/blob/main/stock_price_prediction.ipynb) |
| ディープラーニングによる株価予測PythonCode | [詳細](stock_price_prediction.py) | | |
  
# Kaggleデータ分析処理コード（Python3）
国際的なデータ分析コンペティションのプラットフォームであるKaggleの入門コンペ【Titanic - Machine Learning from Disaster】での分析です。   
簡単にではありますが、データ分析と機械学習・ディープラーニングの処理をNotebookに記してあります。    
NotebookではなくPython3のコードのみのファイルも公開しております。  
| 内容 | ファイル名 | Colab | Kaggle |
| :-- | :-- | :-- | :-- | 
| Kaggleデータ分析Notebook | [詳細](titanic-a-set-of-fundamental-analyses.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazakamibeer/public/blob/main/titanic-a-set-of-fundamental-analyses.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kazakamibeer/public/blob/main/titanic-a-set-of-fundamental-analyses.ipynb) |
| Kaggleデータ分析PythonCode | [詳細](titanic_code_only.py) | | |
  
# LINE・スプレッドシート連携アプリコード（JavaScript, GAS）
前職で作ったLINE・スプレッドシート連携アプリのコードです。  
　　"LINEから見積書.txt"[詳細](LINEから見積書.txt)
LINEでの入力により現場画像付きの見積書をすばやく作成し、PDFファイルとして取引先に送付することができます。  
アプリのために作成した業務用LINEアカウント（公式アカウントと呼ばれています）とユーザーが友達になり、対話形式で求められる情報を入力していきます。  
以下がスマートフォンでの情報入力デモ画面です。  
<img src="img/LINE操作画面.jpg" width="600px">  
以下が作成される見積書の見本のリンクです。画像は商店街の写真などを適当に当てはめただけのもので、本来現場の写真が入ります。  
　　"見積書の見本"[詳細](img/ダミー_見積書見本.pdf)

# ExcelVBAビッグデータ処理コード（VBA）  
前職のコールセンターでは１ケ月で10万件の問い合わせを処理していました。その応答状況を専用ソフトウェアのエクスポートファイルから集計し、ダッシュボードに表示するアプリの一部分です。  
　　"vba_code_only.txt"[詳細](vba_code_only.txt)  
機密性の高い情報ですので、一部のコードのみの公開とさせていただきます。  
興味がおありでしたら、お会いしたときに詳細をお話しさせてください。  
  
# OutlookVBAメール自動返信のコード（VBA）  
前職の不動産賃貸部門で使っていた、Outlookに設定する問い合わせメールへの自動返信のVBAコードです。  
　　"問い合わせメール自動返信vba.txt"[詳細](問い合わせメール自動返信vba.txt)  
大手の不動産賃貸サイトSUUMOで自社の管理する物件に問い合わせのクリックが発生すると、問い合わせの詳細メールが届きます。  
そのメールの内容を読み取り、瞬時に問い合わせ相手のお客様に問い合わせ結果のメールを送信します。  
お客様は数社・数物件に同時に問い合わせを入れることが多く、秒を争う返信合戦になりますが、そのための自動返信コードです。  
  
# Java掲示板アプリコード（Java）  
Javaの現時点での知識のまとめとして、サーブレットによる掲示板アプリを作成しました。  
Tomcatのサーバーにメッセージの情報を入れ、ログイン、メッセージを表示・削除などの機能を付けました。  
Eclipseでのフォルダ構成に合わせています。以下のリンクでプロジェクトフォルダに移動します。  
　　"掲示板アプリプロジェクト"[詳細](掲示板アプリプロジェクト)  
掲示板アプリプロジェクト/src/main/java などにjavaファイルが入っています。
デモ画面を以下に示します。  
<img src="img/掲示板アプリ画面デモ.jpg" width="400px">  
Javaはまだ学び始めて数ヶ月です。
