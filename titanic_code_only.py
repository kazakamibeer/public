###########################################
# Titanic - A set of fundamental analyses #
###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# 学習用データとテスト用データの結合
data = pd.concat([train, test], sort=False)

# dataの情報
data.info()


# -------------------------------------------------------------------------
# 0) 大まかなデータの分析と欠損値の補完
# -------------------------------------------------------------------------

# Sexと生存率の関係 
sns.barplot(x='Sex', y='Survived', data=data, palette='Set3')
plt.show()


# ------------ Name --------------
# 敬称（Title）を Name 列から抽出して新しい列として追加
data['Title'] = data['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
# Title の生存率につながる役割を考え、カテゴリにまとめる
data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
data['Title'].replace(['Mlle'], 'Miss', inplace=True)
data['Title'].replace(['Jonkheer'], 'Master', inplace=True)


# ------------ Age補完 --------------
# Age を Pclass, Sex, Parch, SibSp, Title からランダムフォレストで推定
from sklearn.ensemble import RandomForestRegressor


# 推定に使用する項目を指定
age_data = data[['Age', 'Pclass', 'Sex', 'Parch', 'SibSp', 'Title']]

# ラベル特徴量をワンホットエンコーディング
age_data=pd.get_dummies(age_data)

# 学習データとテストデータに分離し、numpyに変換
known_age = age_data[age_data.Age.notnull()].to_numpy()
unknown_age = age_data[age_data.Age.isnull()].to_numpy()

# 学習データをX, yに分離
X = known_age[:, 1:]   # 列0（Age）を除くと特徴量になる
y = known_age[:, 0]    # 列0が目的変数

# ランダムフォレストで推定モデルを構築
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)

# 推定モデルを使って、テストデータのAgeを予測し、補完
predictedAges = rfr.predict(unknown_age[:, 1:]) 
data.loc[(data.Age.isnull()), 'Age'] = predictedAges
if sum(data['Age'].isnull()) == 0:
    print('--Age補完完了--')


# 年齢別生存曲線と死亡曲線
facet = sns.FacetGrid(data[0:890], hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, data.loc[:890,'Age'].max()))
facet.add_legend()
plt.show()

# 敬称と生存率の関係
sns.barplot(x='Title', y='Survived', data=data, palette='Set3')


# ----------- Fare -------------
# 欠損値を Embarked='S', Pclass=3 の平均値で補完
fare = data.loc[(data['Embarked'] == 'S') & (data['Pclass'] == 3), 'Fare'].median()
data['Fare'] = data['Fare'].fillna(fare)

# ------------- Cabin ----------------
# Cabinの先頭文字を特徴量とする(欠損値は U )
data['Cabin'] = data['Cabin'].fillna('Unknown')
data['Cabin_label'] = data['Cabin'].str.get(0)
sns.barplot(x='Cabin_label', y='Survived', data=data, palette='Set3')
plt.show()

# ---------- Embarked ---------------
# 欠損値をSで補完
data['Embarked'] = data['Embarked'].fillna('S')


data.describe()


# -------------------------------------------------------------------------
# 1) 詳細分析により特徴量を作成する
# -------------------------------------------------------------------------

# ------------ FamilySize ------------
# 家族人数の列を作成
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
# 家族人数と生存率の関係
sns.barplot(x='FamilySize', y='Survived', data=data, palette='Set3')

# ------------ Surname ------------
# NameからSurname(苗字)を抽出
data['Surname'] = data['Name'].str.split(',', 1).str[0].str.strip()

# 同じSurname(苗字)の出現頻度をカウント(出現回数が2以上なら家族)
data['FamilyCount'] = data['Surname'].map(data['Surname'].value_counts())
# 同じSurname(苗字)ごとの生存率の分布を表示
print('苗字（２人以上）ごとの生存率平均値\n生存率平均値　　家族数\n',data.loc[(data['FamilyCount']>=2)].groupby('Surname')['Survived'].mean().value_counts())


# ----------- FamilySize -------------
# FamilySize を生存率でグルーピング
data.loc[(data['FamilySize'] >= 2) & (data['FamilySize'] <= 4), 'Family_label'] = 2
data.loc[(data['FamilySize'] >= 5) & (data['FamilySize'] <= 7) | (data['FamilySize'] == 1), 'Family_label'] = 1
data.loc[(data['FamilySize'] >= 8), 'Family_label'] = 0


# ----------- Same_Surname -------------
# 家族で16才以下または女性の生存率
Female_Child_Group = data.loc[(data['FamilyCount'] >= 2) & ((data['Age'] <= 16) | (data['Sex'] == 'female'))]
Female_Child_Survived_List = Female_Child_Group.groupby('Surname')['Survived'].mean()
print('Female_Child_Group生存率\n', Female_Child_Survived_List.value_counts(), '\n')

# 家族で16才超えかつ男性の生存率
Male_Adult_Group = data.loc[(data['FamilyCount'] >= 2) & (data['Age'] > 16) & (data['Sex'] == 'male')]
Male_Adult_Survived_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
print('Female_Child_Group生存率\n', Male_Adult_Survived_List.value_counts(), '\n')


# ----------- Dead_List, Survived_List -------------
# デッドリストとサバイブリストの作成
Dead_List = set(Female_Child_Survived_List[Female_Child_Survived_List == 0].index)
Survived_List = set(Male_Adult_Survived_List[Male_Adult_Survived_List == 1].index)

# デッドリストとサバイブリストをSex, Age, Title, Surname に反映させる
# ただし変更するのはtestデータのみ
data.loc[(data['Survived'].isnull()) & (data['Surname'].isin(Dead_List)), ['Sex', 'Age', 'Title',  'Surname']] = ['male', 28.0, 'Mr', 'DiCaprio']
data.loc[(data['Survived'].isnull()) & (data['Surname'].isin(Survived_List)), ['Sex', 'Age', 'Title', 'Surname']] = ['female', 5.0, 'Mrs', 'Winslet']


print(data.loc[data['Surname']=='DiCaprio'].head())


# ----------- Ticket ----------------
# 同一Ticketナンバーの人が何人いるかを特徴量として抽出
Ticket_Count = dict(data['Ticket'].value_counts())
data['TicketGroup'] = data['Ticket'].map(Ticket_Count)
sns.barplot(x='TicketGroup', y='Survived', data=data, palette='Set3')
plt.show()


# 生存率で3つにグルーピング
data.loc[(data['TicketGroup'] >= 2) & (data['TicketGroup'] <= 4), 'Ticket_label'] = 2
data.loc[(data['TicketGroup'] >= 5) & (data['TicketGroup'] <= 8) | (data['TicketGroup'] == 1), 'Ticket_label'] = 1  
data.loc[(data['TicketGroup'] >= 11), 'Ticket_label'] = 0


# -------------------------------------------------------------------------
# 2) 推定に使用すべき特徴量を選択する
# -------------------------------------------------------------------------

# 推定に使用する項目を指定
useful_list = ['Survived','Pclass','Sex','Age','Fare','Embarked','Title','Family_label','Cabin_label','Ticket_label']
feature = data[useful_list]

# ラベル特徴量をワンホットエンコーディング
feature = pd.get_dummies(feature)  # このとき特徴量は25個
print(feature.head(3))


# データセットを trainとtestに分割
train = feature[:len(train)]
test = feature[len(train):]

# さらにXとyに分割
X_train = train.drop('Survived', axis=1)  # 学習用入力データ
y_train = train['Survived']               # 学習用正解ラベル
X_test = test.drop('Survived', axis=1)    # テスト用入力データ

# 特徴量を取捨選択する前に、全特徴量の相関を見る
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(train.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()


# -------------------------------------------
# SelectKBestで有用な特徴量を選択し、３セットの入力を作る

from sklearn.feature_selection import SelectKBest, f_classif

# 特徴量をk個選択する関数
def select_k_best_features(X_train, y_train, X_test, k=20, score_func=f_classif):
    # 特徴量選択
    selector = SelectKBest(k=k)
    selector.fit(X_train, y_train)

    # データの変換
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # 採用の可否と列名取得
    mask = selector.get_support()
    original_columns = list(X_train.columns)
    selected_columns = [col for col, keep in zip(original_columns, mask) if keep]

    # データフレームに変換
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_columns)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_columns)

    # 採用可否の出力（オプション：ログとして表示）
    print('\n===== 採用された特徴量一覧 =====')
    for i, col in enumerate(original_columns):
        print(f"No{i+1:2d}: {col:20} => {'〇' if mask[i] else '×'}")

    return X_train_df, X_test_df

X_train_6, X_test_6 = select_k_best_features(X_train, y_train, X_test, k=6)
X_train_12, X_test_12 = select_k_best_features(X_train, y_train, X_test, k=12)
X_train_20, X_test_20 = select_k_best_features(X_train, y_train, X_test, k=20)


# 特徴量の標準化関数
from sklearn.preprocessing import StandardScaler

def standard_scaler(X_train_df, X_test_df):
    list_col = list(X_train_df.columns)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_df)
    X_test_sc = sc.fit_transform(X_test_df)

    # pandasに変換
    X_train_sc = pd.DataFrame(X_train_sc, columns=list_col)
    X_test_sc = pd.DataFrame(X_test_sc, columns=list_col)

    return X_train_sc, X_test_sc


X_train_6, X_test_6 = standard_scaler(X_train_6, X_test_6)
X_train_12, X_test_12 = standard_scaler(X_train_12, X_test_12)
X_train_20, X_test_20 = standard_scaler(X_train_20, X_test_20)


# -------------------------------------------------------------------------
# 3) 推定のためのモデルを関数として組み立てる
# -------------------------------------------------------------------------

# -------------------------------------------
# 3-1 RandomForest

# RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


rf_params = {
    'random_state': 10,
    'n_estimators': 26,
    'max_depth': 6,
    'max_features': 'sqrt',
    'random_state': 0
}
# アンサンブル学習用の弱いパラメータ
rf_weak_params = {
    'n_estimators': 10,      # 木の本数を少なく（小規模なアンサンブル）
    'max_depth': 2,          # 各木を浅く（過学習を防ぐ）
    'max_leaf_nodes': 4,     # 各木の最大葉ノード数を4つに制限（複雑化の防止）
    'max_features': 1,       # 各分割時に1つの特徴量のみ検討（ランダム性↑＋単純化）
    'min_samples_leaf': 10,  # 各葉に最低10サンプル必要（安定したノード）
    'n_jobs': -1,            # 並列処理（速度向上のため）
    'random_state': 0
}

# RandomForestClassifierをcvで実行する関数
def rf_predict(X_train_df, y_train, X_test_df, params=None):
    # デフォルトのパラメータ（指定されなかった場合に使用）
    default_params = {
        'random_state': 10,
        'n_estimators': 26,
        'max_depth': 6,
        'max_features': 'sqrt',
        'random_state': 0
    }
    # 渡されたパラメータで上書き
    if params is not None:
        default_params.update(params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    oof_rf = np.zeros(len(X_train_df))  # X_trainへの正値予測確率を入れる
    y_preds = []  # 各分割ごとの正値予測確率を入れる
    feature_importances = []  # 各特徴量の重要度（寄与度）を表すスコア（和１）を入れる

    
    for train_idx, valid_idx in cv.split(X_train_df, y_train):
        X_tr, X_va = X_train_df.iloc[train_idx], X_train_df.iloc[valid_idx]
        y_tr = y_train[train_idx]

        model_rf = RandomForestClassifier(
            random_state=10,
            n_estimators=26,
            max_depth=6,
            max_features='sqrt'
        )
        model_rf.fit(X_tr, y_tr)
    
        # クラス1（＝正例）の確率を取得
        y_preds.append(model_rf.predict_proba(X_test_df)[:, 1])
        oof_rf[valid_idx] = model_rf.predict_proba(X_va)[:, 1]

        # 各foldのfeature_importances_ を保存
        feature_importances.append(model_rf.feature_importances_)
    
    
    #-- テストデータ予測（CV平均） --#
    proba_rf = np.mean(y_preds, axis=0)

    # OOF精度確認
    y_pred_oof = (oof_rf > 0.5).astype(int)
    print("OOF Accuracy (rf):", accuracy_score(y_train, y_pred_oof))

    # 平均feature_importanceを計算
    avg_importance = np.mean(feature_importances, axis=0)

    return oof_rf, proba_rf, avg_importance


# -------------------------------------------
# 3-2 LightGBM

# LightGBM
import lightgbm as lgb
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


lgbm_params = {
        'objective': 'binary',       # 2クラス分類タスク
        'metric': 'binary_logloss',  # 学習中の評価指標（default）
        'learning_rate': 0.05,       # 学習率
        'num_leaves': 40,            # 1本の木の最大リーフ数（非線形性の強さ）
        'max_bin': 300,              # ヒストグラムのビン数 default250
        'feature_fraction': 0.8,     # 特徴量をランダムにサンプリングして使う（過学習防止）
        'bagging_fraction': 0.8,     # データの一部で学習することで汎化性能アップ（行のサブサンプリング）
        'bagging_freq': 5,           # データの一部で学習することで汎化性能アップ（行のサブサンプリング）
        'verbose': -1,               # ログ出力抑制
        'random_state': 0            # 
    }
# アンサンブル学習用の弱いパラメータ
lgbm_weak_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.3,        # 学習率を大きく（早く収束して木が浅くなる）
    'num_leaves': 4,             # 葉の数を極端に少なくして弱く
    'max_depth': 2,              # 木の深さを制限（浅い木に）
    'min_data_in_leaf': 20,      # 各リーフに必要な最小データ数を増やす
    'feature_fraction': 0.5,     # 特徴量サブサンプリング
    'bagging_fraction': 0.5,     # データサブサンプリング
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 0
}

# lightgbmをcvで実行する関数
def lgbm_predict(X_train_df, y_train, X_test_df, params):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    categorical_features = []

    oof_lgbm = np.zeros(len(X_train_df))  # X_trainへの正値予測確率を入れる
    y_preds = []  # 各分割ごとの正値予測確率を入れる
    models = []   # 各分割ごとの学習したモデルを入れる
    scores = []   # 各分割ごとのbinary_loglossを入れる
    feature_importances = []  # 各foldの特徴量重要度を格納

    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train_df, y_train)):
        X_tr, X_va = X_train_df.iloc[train_idx], X_train_df.iloc[valid_idx]
        y_tr, y_va = y_train[train_idx], y_train[valid_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_valid = lgb.Dataset(X_va, y_va, reference=lgb_train)

        model_lgbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(10)
            ]
        )

        oof_lgbm[valid_idx] = model_lgbm.predict(
            X_va, num_iteration=model_lgbm.best_iteration
        )
        y_preds.append(
            model_lgbm.predict(X_test_df, num_iteration=model_lgbm.best_iteration)
        )
        models.append(model_lgbm)
    
        # 使用されるモデルの検証データへのloglossを配列に入れる
        # best_scoreは２次元dict, valid_1もbinary_loglossもkey
        scores.append(model_lgbm.best_score['valid_1']['binary_logloss'])

        # 特徴量重要度（gain）を保存
        feature_importances.append(
            model_lgbm.feature_importance(importance_type='gain')
        )

    #-- テストデータ予測（CV平均） --#
    proba_lgbm = np.mean(y_preds, axis=0)

    # モデルの性能確認のため、oofへのスコアも見る
    # スコアの平均を表示
    print("===CV scores===")
    print(scores)
    print("Average:", np.mean(scores))
    # OOF精度確認
    y_pred_oof = (oof_lgbm > 0.5).astype(int)
    print("OOF Accuracy (lgbm):", accuracy_score(y_train, y_pred_oof))

    # 特徴量重要度（平均）
    avg_importance = np.mean(feature_importances, axis=0)
    feature_names = X_train_df.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain_mean': avg_importance
    }).sort_values(by='importance_gain_mean', ascending=False)

    return oof_lgbm, proba_lgbm, importance_df


# -------------------------------------------
# 3-3 NeuralNetwork

# NeuralNetwork
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import HeNormal

def build_model(input_dim=18, learning_rate=0.001):
    model = Sequential()
    
    # 入力層 + BatchNormalization
    model.add(Dense(64, input_dim=input_dim, activation='relu', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # 中間層1
    model.add(Dense(32, activation='relu', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # 中間層2（少し浅く）
    model.add(Dense(16, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dropout(0.5))
    
    # 出力層
    model.add(Dense(1, activation='sigmoid'))
    
    # オプティマイザ
    optimizer = Adam(learning_rate=learning_rate)
    
    # モデルのコンパイル
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# アンサンブル学習用の弱いモデル
def build_weak_model(input_dim=18, learning_rate=0.01):
    model = Sequential()

    # 小さな1層のみ＋正則化
    model.add(Dense(8, input_dim=input_dim, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dropout(0.3))

    # 出力層
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

#from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
#from sklearn.metrics import accuracy_score

# NeuralNetworkのパラメータ
early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,  # 小さな改善を無視　val_accuracyなら1e-3
        patience=10,     # 10エポック改善なければ停止
        verbose=1,
        restore_best_weights=True  # 推奨：最良モデルに戻す
    )
# アンサンブル学習用の弱いパラメータ（早めに止める）
early_stop_weak = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,  # 小さな改善を認めない
        patience=5,      # 短くして早めに打ち切る
        verbose=0,
        restore_best_weights=True
    )

# NeuralNetworkをcvで実行する関数
def nn_predict(X_train_df, y_train, X_test_df, build_model_fn, early_stopping_cb):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    oof_nn = np.zeros(len(X_train_df))  # X_trainへの正値予測確率を入れる
    y_preds = []    # 各分割ごとの正値予測確率を入れる
    histories = []  # 各分割ごとの学習履歴

    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train_df, y_train)):
        X_tr, X_va = X_train_df.iloc[train_idx], X_train_df.iloc[valid_idx]
        y_tr, y_va = y_train[train_idx], y_train[valid_idx]

        # 渡された build_model_fn を使ってモデル作成
        model_nn = build_model_fn(input_dim=X_tr.shape[1])
        # 最初のfoldのみモデル構造を表示
        if fold_id == 0:
            model_nn.summary()

        history = model_nn.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping_cb],
            verbose=0
        )
        histories.append(history)

        y_preds.append(model_nn.predict(X_test_df).flatten())
        oof_nn[valid_idx] = model_nn.predict(X_va).flatten()

    #-- テストデータ予測（CV平均） --#
    proba_nn = np.mean(y_preds, axis=0)

    # 学習履歴の可視化（省略可）
    for fold_id, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'Model Accuracy (fold {fold_id})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

    # OOF精度表示
    y_pred_oof = (oof_nn > 0.5).astype(int)
    print("OOF Accuracy (NN):", accuracy_score(y_train, y_pred_oof))

    return oof_nn, proba_nn


# -------------------------------------------------------------------------
# 4) １層目モデルの実行
# -------------------------------------------------------------------------

# 提出ファイルのひな型
sub = pd.read_csv('../input/titanic/gender_submission.csv')

print('\n----------- [rf_6] session start -----------\n')
oof_rf_6, proba_rf_6, avg_importance = rf_predict(X_train_6,
                                                  y_train,
                                                  X_test_6,
                                                  params=rf_params)
sub['Survived'] = (proba_rf_6 > 0.5).astype(int)
sub.to_csv('sub_rf_6.csv', index=False)

print('\n----------- [nn_6] session start -----------\n')
oof_nn_6, proba_nn_6 = nn_predict(X_train_6,
                                  y_train,
                                  X_test_6,
                                  build_model_fn=build_model,
                                  early_stopping_cb=early_stop)
sub['Survived'] = (proba_nn_6 > 0.5).astype(int)
sub.to_csv('sub_nn_6.csv', index=False)

print('\n----------- [lgbm_12] session start -----------\n')
oof_lgbm_12, proba_lgbm_12, importance_df = lgbm_predict(X_train_12,
                                                         y_train,
                                                         X_test_12,
                                                         params=lgbm_params)
sub['Survived'] = (proba_lgbm_12 > 0.5).astype(int)
sub.to_csv('sub_lgbm_12.csv', index=False)

print('\n----------- [nn_12] session start -----------\n')
oof_nn_12, proba_nn_12 = nn_predict(X_train_12, 
                                    y_train,
                                    X_test_12,
                                    build_model_fn=build_model,
                                    early_stopping_cb=early_stop)
sub['Survived'] = (proba_nn_12 > 0.5).astype(int)
sub.to_csv('sub_nn_12.csv', index=False)

print('\n----------- [rf_20] session start -----------\n')
oof_rf_20, proba_rf_20, avg_importance = rf_predict(X_train_20,
                                                    y_train,
                                                    X_test_20,
                                                    params=rf_params)
sub['Survived'] = (proba_rf_20 > 0.5).astype(int)
sub.to_csv('sub_rf_20.csv', index=False)

print('\n----------- [lgbm_20] session start -----------\n')
oof_lgbm_20, proba_lgbm_20, importance_df = lgbm_predict(X_train_20,
                                                         y_train,
                                                         X_test_20,
                                                         params=lgbm_params)
sub['Survived'] = (proba_lgbm_20 > 0.5).astype(int)
sub.to_csv('sub_lgbm_20.csv', index=False)

print('\n----------- [nn_20] session start -----------\n')
oof_nn_20, proba_nn_20 = nn_predict(X_train_20,
                                    y_train,
                                    X_test_20,
                                    build_model_fn=build_model,
                                    early_stopping_cb=early_stop)
sub['Survived'] = (proba_nn_20 > 0.5).astype(int)
sub.to_csv('sub_nn_20.csv', index=False)


# -------------------------------------------------------------------------
# 5) １層目の出力をアンサンブルする（２層目モデル）
# -------------------------------------------------------------------------

test_preds = pd.DataFrame({'rf_6': proba_rf_6,
                           'nn_6': proba_nn_6,
                           'lgbm_12': proba_lgbm_12,
                           'nn_12': proba_nn_12,
                           'rf_20': proba_rf_20,
                           'lgbm_20': proba_lgbm_20,
                           'nn_20': proba_nn_20})
print(test_preds.corr())


# -------------------------------------------
# 5-1 WeightedAverageEnsemble（重み付き平均）

# 各出力に対する重み（和が１になるように調整）
weights = [0.1, 0.1, 0.15, 0.15, 0.17, 0.17, 0.16]

meta_proba_wae = np.dot(test_preds, weights)
sub['Survived'] = meta_proba_wae.astype(int)
sub.to_csv('sub_ensemble_wae.csv', index=False)


# 学習データへの予測値
train_preds = pd.DataFrame({'rf_6': oof_rf_6,
                           'nn_6': oof_nn_6,
                           'lgbm_12': oof_lgbm_12,
                           'nn_12': oof_nn_12,
                           'rf_20': oof_rf_20,
                           'lgbm_20': oof_lgbm_20,
                           'nn_20': oof_nn_20})
# 学習データへの正解ラベル: y_train

print(train_preds.corr())

# train_predsを６個の特徴量で作ったが、相関を見て５個の特徴量に絞る。
# ここからは rf_20 はカットする。

train_preds = pd.DataFrame({'rf_6': oof_rf_6,
                           'nn_6': oof_nn_6,
                           'lgbm_12': oof_lgbm_12,
                           'nn_12': oof_nn_12,
                           'lgbm_20': oof_lgbm_20,
                           'nn_20': oof_nn_20})
test_preds = pd.DataFrame({'rf_6': proba_rf_6,
                           'nn_6': proba_nn_6,
                           'lgbm_12': proba_lgbm_12,
                           'nn_12': proba_nn_12,
                           'lgbm_20': proba_lgbm_20,
                           'nn_20': proba_nn_20})

# -------------------------------------------
# 5-2 LogisticRegressionアンサンブル

from sklearn.linear_model import LogisticRegression

# メタモデル：過学習を最大限に抑えたロジスティック回帰
meta_model = LogisticRegression(
    penalty='l2',             # L2正則化（重みをゼロに近づける）
    #solver='liblinear',       # 特徴量が少ないときやL1/L2で安定
    solver='lbfgs',           # より一般的・高速なソルバーに変更
    fit_intercept=False,      # オフセット項なし（モデルを単純に）
    C=0.01,                   # 正則化を強く（小さいほどペナルティが強い）
    max_iter=1000,            # 十分な収束を保証（エラー防止用）
    random_state=0
)
meta_model.fit(train_preds, y_train)

weights = meta_model.coef_.flatten()  # meta_model.coef_はshape (1, n_features)
weights = weights / weights.sum()  # 合計が1になるようにスケーリング
print("Optimized Model weights:", weights)

#-- テストデータに適用して最終出力を生成 --#
meta_proba_logi = np.dot(test_preds, weights)
sub['Survived'] = (meta_proba_logi > 0.5).astype(int)
sub.to_csv('sub_ensemble_logi.csv', index=False)


# -------------------------------------------
# 5-3 RandomForestアンサンブル

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# メタモデル：ランダムフォレスト
meta_oof_rf, meta_proba_rf, avg_importance = rf_predict(train_preds,
                                                        y_train,
                                                        test_preds,
                                                        params=rf_weak_params)

# 重みの代わりにフィーチャーインポータンスを見る
print("Feature importances (model weights):")
for name, imp in zip(train_preds.columns, avg_importance):
    print(f"{name}: {imp:.4f}")

#-- テストデータに適用して最終出力を生成 --#
sub['Survived'] = (meta_proba_rf > 0.5).astype(int)
sub.to_csv('sub_ensemble_rf.csv', index=False)


# -------------------------------------------
# 5-4 LightGBMアンサンブル

# paramsにアンサンブル用のweak_paramsを使用
meta_oof_lgbm, meta_proba_lgbm, importance_df = lgbm_predict(train_preds,
                                                             y_train,
                                                             test_preds,
                                                             params=lgbm_weak_params)
sub['Survived'] = (meta_proba_lgbm > 0.5).astype(int)
sub.to_csv('sub_ensemble_lgbm.csv', index=False)
print(importance_df)


# -------------------------------------------
# 5-5 NeuralNetworkアンサンブル

# アンサンブル用のパラメータbuild_weak_model, early_stop_weakを使用
meta_oof_nn, meta_proba_nn = nn_predict(train_preds,
                                        y_train,
                                        test_preds,
                                        build_model_fn=build_weak_model,
                                        early_stopping_cb=early_stop_weak)
sub['Survived'] = (meta_proba_nn > 0.5).astype(int)
sub.to_csv('sub_ensemble_nn.csv', index=False)


# -------------------------------------------------------------------------
# 6) ２層目の出力をアンサンブル（３層目モデル）
# -------------------------------------------------------------------------

meta_oof_preds = pd.DataFrame({'meta_rf': meta_oof_rf,
                               'meta_lgbm': meta_oof_lgbm,
                               'meta_nn': meta_oof_nn})
meta_proba_preds = pd.DataFrame({'meta_rf': meta_proba_rf,
                                 'meta_lgbm': meta_proba_lgbm,
                                 'meta_nn': meta_proba_nn})

print(meta_oof_preds.corr())

# メタモデル：過学習を最大限に抑えたロジスティック回帰
meta_model = LogisticRegression(
    penalty='l2',             # L2正則化（重みをゼロに近づける）
    #solver='liblinear',       # 特徴量が少ないときやL1/L2で安定
    solver='lbfgs',           # より一般的・高速なソルバーに変更
    fit_intercept=False,      # オフセット項なし（モデルを単純に）
    C=0.01,                   # 正則化を強く（小さいほどペナルティが強い）
    max_iter=1000,            # 十分な収束を保証（エラー防止用）
    random_state=0
)
meta_model.fit(meta_oof_preds, y_train)

weights = meta_model.coef_.flatten()  # meta_model.coef_はshape (1, n_features)
weights = weights / weights.sum()  # 合計が1になるようにスケーリング
print("Optimized Model weights:", weights)

#-- テストデータに適用して最終出力を生成 --#
meta_ensemble_proba = np.dot(meta_proba_preds, weights)
sub['Survived'] = (meta_ensemble_proba > 0.5).astype(int)
sub.to_csv('sub_meta_ensemble.csv', index=False)

# ↓単純平均による最終出力の方が良い結果が出た
# meta_ensemble_proba = meta_preds.apply(lambda x: x.mean() , axis=1)