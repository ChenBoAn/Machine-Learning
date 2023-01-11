from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#scikit-learn 的特徵資料僅允許輸入數值資料
#女生: 0, 男生: 1
feature = [[1, 32], [1, 25], [0, 26], [1, 19], [0, 28], [0, 18], [1, 17], [0, 22], [1, 29], [0, 30]]

#創造目標資料 (正確答案), 目標資料允許輸入非數值資料
#美國隊長: 0, 鋼鐵人: 1
target = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

#scikit-learn 的決策樹物件
tree = DecisionTreeClassifier(criterion='entropy') #使用 entropy 作為分類標準

#使用創造的資料生成決策樹
tree.fit(feature, target) 

#輸入特徵資料看結果
prediction = tree.predict(feature)
print(prediction)

#決策樹的分類過程
export_graphviz(tree, out_file='hero.dot',
                feature_names=['gender', 'age'], #特徵名稱
                class_names=['Captain America', 'Iron Man']) #類別名稱
