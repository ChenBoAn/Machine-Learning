'''
載入葡萄酒資料
'''
from sklearn import datasets

wine_data = datasets.load_wine()

#wine_data.feature_names: 特徵名稱, wine_data.target_names: 類別名稱
#wine_data.data: 特徵資料, wine_data.target: 目標資料
print(wine_data.data.shape)
print("特徵名稱: ", wine_data.feature_names)
print("類別名稱: ", wine_data.target_names)
#共有 178 筆資料，一筆資料有 13 個特徵，且被分為 3 個類別。

'''
將資料分割成訓練資料和測試資料
'''
from sklearn.model_selection import train_test_split
train_feature, test_feature, train_target, test_target = train_test_split(wine_data.data, wine_data.target, test_size=0.3)
#訓練資料特徵   測試資料特徵   訓練資料目標   測試資料目標                   總資料特徵       總資料目標         取30%的資料作為測試資料

'''
將分割好的資料先建立一個決策樹
'''
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=4) #限制決策樹的深度為4層，避免樹生長過大
tree.fit(train_feature, train_target)

'''
接著建立一個隨機森林
'''
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, max_depth=4) #設定1個森林10棵樹
forest.fit(train_feature, train_target)

'''
利用分類器的 score() 來評估效果，此函式需傳入特徵資料和目標
它會將特徵資料進行分類，並比較與結果的差異，輸出分類的準確率
以下分別評估決策樹和隨機森林的準確率
'''
accuracy_tree = tree.score(test_feature, test_target)
accuracy_forest = forest.score(test_feature, test_target)

print("Decision Tree: ", accuracy_tree)
print("Random Forest: ", accuracy_forest)