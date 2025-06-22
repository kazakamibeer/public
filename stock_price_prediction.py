# 0) 概論
# このNotebookでは、大まかに以下の方法で株価など時系列データの未来予測を行う
# 一年ほどを掛けて精度を上げていく予定。いずれは自動売買の予想モデルとして使用したい。  
# 2026年夏頃の完成を目指す。  

# 論文を参考にした以下の５つのモデルでそれぞれ予測を行う  
# - LSTM
# - Bidirectional LSTM + Attention
# - Transformer
# - Informer
# - Autoformer  

# ５つのモデルの精度を考慮しつつ、結果をアンサンブルする
# 上記５モデルが未完成のため、現状は机上の空論のみで簡易なコードを作成。  
# ５モデルの出力に加えて、相関・逆相関のある金融の指数を特徴量として加える。  
# アンサンブルのメタモデルには、学習性能を弱めたRandomForestを使おうと考えている。  

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas_datareader import data as pdr
import pandas_datareader.data as pdr
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
%matplotlib inline

ticker_symbole = 'GOOG.US' 
start = '2014-01-01'
end   = datetime.now()
data = pdr.DataReader(ticker_symbole, 'stooq', start=start, end=end)

# dataには新しい順に数値が入っている 時系列にしたいので、逆順にソート
data = data.iloc[::-1]  
data.head()

#-- 移動平均 --#
ma_day = [20, 50, 100]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    data[column_name] = data['Close'].rolling(ma).mean()

#-- MACD --#
short_period = 12
long_period = 26
signal_period = 9
# 短期と長期のEMA（指数平滑移動平均）を計算
data['EMA_short'] = data['Close'].ewm(span=short_period, adjust=False).mean()
data['EMA_long'] = data['Close'].ewm(span=long_period, adjust=False).mean()
# MACDラインを計算
data['MACD'] = data['EMA_short'] - data['EMA_long']
# シグナルラインを計算
data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
# MACDヒストグラムを計算
data['MACD_Histogram'] = data['MACD'] - data['Signal']

#-- RSI --#
def calc_rsi(period, delta):
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rsi = 100 *(avg_gain/(avg_gain+avg_loss))
    return rsi
delta = data['Close'].diff()
data['RSI9'] = calc_rsi(9, delta)
data['RSI14'] = calc_rsi(14, delta)

plt.title(ticker_symbole + ' Close Price MA History')
plt.plot(data['Close'][-500:])           # 最新500日のみ表示
plt.plot(data['MA for 20 days'][-500:])
plt.plot(data['MA for 50 days'][-500:])
plt.plot(data['MA for 100 days'][-500:])
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.legend(['Close', 'MA for 20 days', 'MA for 50 days', 'MA for 100 days'], loc='upper right')
plt.show()

# datasetとして利用するClose(終値)
dataset = data[['Close', 'Volume', 'MACD_Histogram', 'Signal', 'RSI9', 'RSI14']]
#-- リターン（単純変化率） --#
dataset['returns'] = dataset['Close'].pct_change()

dataset = dataset.iloc[20:]


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 特徴量列の選択
features = ['returns', 'Volume', 'MACD_Histogram', 'Signal', 'RSI9', 'RSI14']
feature_data = dataset[features].copy()

# スケーリング
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# StandardScaler：returns, MACD_Histogram, Signal（正負ありデータ）
standard_cols = ['returns', 'MACD_Histogram', 'Signal']
feature_data[standard_cols] = standard_scaler.fit_transform(feature_data[standard_cols])

# MinMaxScaler：Volume, RSI9, RSI14（非負データ）
minmax_cols = ['Volume', 'RSI9', 'RSI14']
feature_data[minmax_cols] = minmax_scaler.fit_transform(feature_data[minmax_cols])

#---------- トレーニングデータの作成 ----------# 
# 前から80%をトレーニングデータとして扱う
train_data_len = int(np.ceil(len(feature_data) * 0.8))
# どれくらいの期間をもとに予測するか
window_size = 60  # 60days

# 特徴量すべて（6列）をスライスし、x_train, y_trainを作成
x_train, y_train = [], []  # type: list 変換の必要あり

for i in range(window_size, train_data_len):
    x_train.append(feature_data.iloc[i - window_size:i].to_numpy())
    y_train.append(feature_data.iloc[i]['returns'])  # 目的変数は returns（正規化済）

# ndarrayに変換
x_train, y_train = np.array(x_train), np.array(y_train)

# LSTM入力用の shape に変形
# (samples, timesteps, features)
print("x_train shape:", x_train.shape)  # → (train_samples, 60, 6)
print("y_train shape:", y_train.shape)  # → (train_samples,)

#---------- テストデータの作成 ----------#
test_data = feature_data.iloc[train_data_len - window_size:].copy()  # 60日分前から必要

x_test, y_test = [], []  # type: list 変換の必要あり

for i in range(window_size, len(test_data)):
    x_test.append(test_data.iloc[i - window_size:i].to_numpy())
    y_test.append(test_data.iloc[i]['returns'])  # 予測対象は returns（正規化済）

# ndarrayに変換
x_test, y_test = np.array(x_test), np.array(y_test)

# shape 確認（samples, timesteps, features）
print("x_test shape:", x_test.shape)   # (テストサンプル数, 60, 6)
print("y_test shape:", y_test.shape)   # (テストサンプル数,)


# 後の逆変換用に returns 専用スケーラー保存
returns_scaler = StandardScaler()
returns_scaler.fit(dataset[['returns']])  # 正規化前の 'returns' に対してフィット


model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(x_train, y_train, batch_size=32, epochs=120)

# 予測を実行
predictions = model.predict(x_test)
predicted = returns_scaler.inverse_transform(predictions)
actual = returns_scaler.inverse_transform(y_test.reshape(-1, 1))

# 評価指標の計算
m_pred = np.mean(predicted) * 100  # %表示
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

# 結果表示
print(f"平均  : {m_pred:.3f}[%]")
# print(f"MAE  : {mae:.4f}")  # 平均絶対誤差: 0に近いほど良い
print(f"RMSE : {rmse:.4f}")  # 二乗平均平方根誤差: 0に近いほど良い
print(f"R²   : {r2:.4f}")    # 決定係数: 1に近いほど良い

# 的中率（独自定義）を求める
# 閾値リスト（パーセンテージ表記）
thresholds = [0.001, 0.002, 0.005]

print("的中率（予測誤差が閾値以内）:")
for threshold in thresholds:
    errors = np.abs(predicted - actual)  # 絶対誤差を計算
    hit_mask = errors <= threshold  # 的中条件
    hit_rate = np.mean(hit_mask) * 100  # 的中率（%）を計算
    print(f"  ±{threshold*100:.1f}%以内: {hit_rate:.2f}%")

# 予測と実際の符号が一致しているか
same_direction = np.sign(predicted) == np.sign(actual)
directional_accuracy = np.mean(same_direction) * 100

print(f"方向性一致率: {directional_accuracy:.2f}%")


def predict_future_sequence(model, recent_data, window_size=60, future_days=30):
    """
    model        : 学習済みのLSTMモデル（Keras）
    recent_data  : 正規化された系列データ（scaled）で、shape=(window_size, 1)
    window_size  : モデルの入力シーケンス長（通常は60）
    future_days  : 何日先まで予測するか（通常は30）
    
    戻り値: 予測値のリスト（正規化済）
    """
    predictions = []

    # 予測に使うデータをコピー（書き換えないため）
    input_seq = recent_data.copy()

    for _ in range(future_days):
        # モデルに入力するためshape変換: (1, window_size, x_train.shape[2])
        x_input = np.reshape(input_seq, (1, window_size, x_train.shape[2]))

        # 予測
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0][0])  # スカラー値として保存

        # input_seqを1つシフトして、予測値を末尾に追加
        input_seq = np.append(input_seq[1:], [[pred[0][0]]], axis=0)

    return predictions


# 30日前からさかのぼって60日分（正規化済）のデータを準備
last_60_scaled = feature_data[["returns", "Volume", "MACD_Histogram", "Signal", "RSI9", "RSI14"]][-90:-30].to_numpy().reshape(-1, 1)  # shape=(60, 1)

# 最後の30日をマルチステップ予測
future_returns_scaled = predict_future_sequence(model, last_60_scaled, window_size=60, future_days=30)

# 逆正規化
future_returns = returns_scaler.inverse_transform(np.array(future_returns_scaled).reshape(-1, 1))

# 最後の実際の価格
last_price = dataset.iloc[-31]['Close']

# リターンから株価へ変換
future_preds = [last_price]
for ret in future_returns:
    next_price = future_preds[-1] * (1 + ret[0])
    future_preds.append(next_price)

# 先頭のlast_priceだけは実際の値であることに注意

print(f"起点価格: {last_price:.2f}")
print(f"予測された価格（最初の5日）: {future_preds[:5]}")

train = data[:-30]
valid = data[-31:].copy()  # コピーを作成 グラフをつなげるために31日前からのデータ
valid['Predictions'] = future_preds

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.plot(train['Close'][-30:], label='Train')
plt.plot(valid['Close'], label='Val')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.show()



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Layer, BatchNormalization
import tensorflow.keras.backend as K

# Attentionレイヤー（重み出力用）
class AttentionLayer(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)  # Kerasカスタムレイヤーの内部的な初期化処理を実行
        self.return_attention = return_attention  # Attentionの重み出力をするか否か

    def build(self, input_shape):  # レイヤー内部の学習可能なパラメータ（重みなど）を定義
                                   # 入力を与えると自動で実行
                                   # W（重み行列）, b（バイアス）
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),    # 特徴量の次元数（LSTMの出力次元数）
                                 initializer='glorot_uniform',  # Xavier初期化
                                 trainable=True)                # 学習対象
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),  # 時系列の長さ（タイムステップ数）
                                 initializer='zeros',        # 初期値 0
                                 trainable=True)             # 学習対象
        super().build(input_shape)

    def call(self, x):  # Kerasのカスタムレイヤーで定義するメインの処理関数
                        # __call()__に入れられ、自動で実行
        e = K.tanh(K.dot(x, self.W) + self.b)        # スコア
        a = K.softmax(e, axis=1)                     # 重み（合計1）
        output = x * a                               # 各時刻の出力に重みを乗算
        context_vector = K.sum(output, axis=1)       # 重み付き和
        if self.return_attention:
            return [context_vector, a]
        else:
            return context_vector


# モデル定義関数
def create_model_with_attention(seq_len=60, num_features=6):
    # 入力の形状（例：60タイムステップ × 特徴量数）
    inputs = Input(shape=(seq_len, num_features))
    
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    context_vector, attention_weights = AttentionLayer(return_attention=True)(x)

    x = Dense(128, activation='relu')(context_vector)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    output = Dense(1)(x)  # 予測値（1日後の変化率）

    model = Model(inputs=inputs, outputs=output)
    att_model = Model(inputs=inputs, outputs=attention_weights)  # Attention可視化用モデル

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model, att_model

# モデルの作成
model, att_model = create_model_with_attention()

# 学習
history = model.fit(x_train, y_train, batch_size=32, epochs=100)

# 予測を実行
predictions = model.predict(x_test)
predicted = returns_scaler.inverse_transform(predictions)
actual = returns_scaler.inverse_transform(y_test.reshape(-1, 1))

# 評価指標の計算
m_pred = np.mean(predicted) * 100  # %表示
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

# 結果表示
print(f"平均  : {m_pred:.3f}[%]")
# print(f"MAE  : {mae:.4f}")  # 平均絶対誤差: 0に近いほど良い
print(f"RMSE : {rmse:.4f}")  # 二乗平均平方根誤差: 0に近いほど良い
print(f"R²   : {r2:.4f}")    # 決定係数: 1に近いほど良い

# 的中率（独自定義）を求める
# 閾値リスト（パーセンテージ表記）
thresholds = [0.001, 0.002, 0.005]

print("的中率（予測誤差が閾値以内）:")
for threshold in thresholds:
    errors = np.abs(predicted - actual)  # 絶対誤差を計算
    hit_mask = errors <= threshold  # 的中条件
    hit_rate = np.mean(hit_mask) * 100  # 的中率（%）を計算
    print(f"  ±{threshold*100:.1f}%以内: {hit_rate:.2f}%")

# 予測と実際の符号が一致しているか
same_direction = np.sign(predicted) == np.sign(actual)
directional_accuracy = np.mean(same_direction) * 100

print(f"方向性一致率: {directional_accuracy:.2f}%")


# X_sample: (1, 60, num_features) の1つのサンプルを渡す
day_sample = []
day_sample.append(x_test[300])
X_sample = np.array(day_sample)

attention_weights = att_model.predict(X_sample)

# 可視化
plt.figure(figsize=(10, 4))
plt.plot(attention_weights[0])
plt.title("Attention Weights by Time Step")
plt.xlabel("Time Step")
plt.ylabel("Attention Weight")
plt.show()

def predict_future_sequence(model, recent_data, window_size=60, future_days=30):
    """
    model        : 学習済みのLSTMモデル（Keras）
    recent_data  : 正規化された系列データ（scaled）で、shape=(window_size, 1)
    window_size  : モデルの入力シーケンス長（通常は60）
    future_days  : 何日先まで予測するか（通常は30）
    
    戻り値: 予測値のリスト（正規化済）
    """
    predictions = []

    # 予測に使うデータをコピー（書き換えないため）
    input_seq = recent_data.copy()

    for _ in range(future_days):
        # モデルに入力するためshape変換: (1, window_size, x_train.shape[2])
        x_input = np.reshape(input_seq, (1, window_size, x_train.shape[2]))

        # 予測
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0][0])  # スカラー値として保存

        # input_seqを1つシフトして、予測値を末尾に追加
        input_seq = np.append(input_seq[1:], [[pred[0][0]]], axis=0)

    return predictions


# 30日前からさかのぼって60日分（正規化済）のデータを準備
last_60_scaled = feature_data[["returns", "Volume", "MACD_Histogram", "Signal", "RSI9", "RSI14"]][-90:-30].to_numpy().reshape(-1, 1)  # shape=(60, 1)

# 最後の30日をマルチステップ予測
future_returns_scaled = predict_future_sequence(model, last_60_scaled, window_size=60, future_days=30)

# 逆正規化
future_returns = returns_scaler.inverse_transform(np.array(future_returns_scaled).reshape(-1, 1))

# 最後の実際の価格
last_price = dataset.iloc[-31]['Close']

# リターンから株価へ変換
future_preds = [last_price]
for ret in future_returns:
    next_price = future_preds[-1] * (1 + ret[0])
    future_preds.append(next_price)

# 先頭のlast_priceだけは実際の値であることに注意

print(f"起点価格: {last_price:.2f}")
print(f"予測された価格（最初の5日）: {future_preds[:5]}")

# %% [markdown]
# ### 実際の株価とマルチステップ予測値をグラフで比較
# 実際の株価＝Val  
# 予測値＝Predictions

# %% [code]
train = data[:-30]
valid = data[-31:].copy()  # コピーを作成 グラフをつなげるために31日前からのデータ
valid['Predictions'] = future_preds

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.plot(train['Close'][-30:], label='Train')
plt.plot(valid['Close'], label='Val')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.show()


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Input, Model
# from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
# from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D

# def build_transformer_model(seq_len=60, num_features=5, d_model=64, num_heads=4, ff_dim=128, dropout=0.1):
#     inputs = Input(shape=(seq_len, num_features))
#     x = Dense(d_model)(inputs)  # 埋め込み次元へ変換
    
#     # Encoder層
#     attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
#     attn_out = Dropout(dropout)(attn_out)
#     x = LayerNormalization()(x + attn_out)  # 残差接続 + 正規化
    
#     ff = Dense(ff_dim, activation="relu")(x)
#     ff = Dense(d_model)(ff)
#     ff = Dropout(dropout)(ff)
#     x = LayerNormalization()(x + ff)
    
#     x = GlobalAveragePooling1D()(x)  # 時系列平均
#     x = Dropout(dropout)(x)
#     outputs = Dense(1)(x)  # 次日株価予測
    
#     model = Model(inputs, outputs)
#     model.compile(optimizer="adam", loss="mse")
#     return model

# model = build_transformer_model(seq_len=60, num_features=5)
# model.summary()
