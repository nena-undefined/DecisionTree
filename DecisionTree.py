import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class _Node(object):
    def __init__(self, criterion=None, max_depth=None, random_state=None, is_reg=True):
        self.criterion    = criterion #分割基準
        self.max_depth    = max_depth #最大の木の深さ
        self.random_state = random_state #seed値
        self.is_reg       = is_reg #Trueの時回帰木，Falseの時分類木
        self.depth        = None #現在のノードの深さ
        self.left         = None #左の子ノード
        self.right        = None #右の子ノード
        self.feature      = None #分割に用いる変数番号
        self.threshold    = None #分割に用いる閾値
        self.label        = None #予測値（ラベル番号)
        self.is_leaf      = False #現在のノードが葉ノードかどうか

        
    def split_node(self, X, y, depth):
        self.depth = depth

        num_samples = len(y)
        num_features = X.shape[1]

        #データが一つの時，葉ノードにする
        if len(np.unique(y)) == 1:
            self.label = y[0]
            self.is_leaf = True
            return
        
        #停止条件
        if depth == self.max_depth:
            #回帰木では目的変数yの平均値を予測値にする         
            if self.is_reg == True:
                self.label = np.mean(y)
            #分類木では各クラスの数を数え，もっとも多いラベルを予測ラベルにする
            else:    
                class_count = {i: len(y[y==i]) for i in np.unique(y)}
                self.label = max(class_count.items(), key=lambda x:x[1])[0]
            self.is_leaf = True
            return
        
        #情報利得を初期化
        info_gain = 0.0

        if self.random_state!=None:
            np.random.seed(self.random_state)
            
        #どの変数から見ていくかscikitの仕様と合わせている
        #（同じ情報利得が得られる変数が複数ある時に効いてくる）
        f_loop_order = np.random.permutation(num_features).tolist()
        for f in f_loop_order:
            
            #変数X_fをソートし，重複しているものを削除する
            uniq_feature = np.unique(X[:, f])
            
            #分割点候補は(uniq_featureの総数 - 1)
            #spilit_pointsには各分割点で分割するときの閾値が入っている
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0

            #全ての閾値候補について探索
            for threshold in split_points:
                y_l = y[X[:, f] <= threshold] 
                y_r = y[X[:, f] >  threshold]
                if y_l.shape[0] == 0 or y_r.shape[0] == 0:
                    continue
                    
                val = self.calc_info_gain(y, y_l, y_r)
                
                #情報利得更新
                if info_gain < val:
                    info_gain = val
                    self.feature   = f
                    self.threshold = threshold

        #情報利得が0の時(良い分割点が見つからなかった時)
        if info_gain == 0.0:
            #回帰木では目的変数yの平均値を予測値にする         
            if self.is_reg == True:
                self.lebel = np.mean(y)
            #分類木では各クラスの数を数え，もっとも多いラベルを予測ラベルにする
            else:    
                class_count = {i: len(y[y==i]) for i in np.unique(y)}
                self.label = max(class_count.items(), key=lambda x:x[1])[0]
            self.is_leaf = True
            return

        #データを左右に分割
        X_l   = X[X[:, self.feature] <= self.threshold]
        y_l   = y[X[:, self.feature] <= self.threshold]
        self.left  = _Node(self.criterion, self.max_depth, self.random_state, self.is_reg)
        self.left.split_node(X_l, y_l, depth + 1)

        X_r   = X[X[:, self.feature] > self.threshold]
        y_r   = y[X[:, self.feature] > self.threshold]
        self.right = _Node(self.criterion, self.max_depth, self.random_state, self.is_reg)
        self.right.split_node(X_r, y_r, depth + 1)

    def criterion_func(self, y):
        classes = np.unique(y)
        numdata = len(y)

        #ジニ係数
        if self.criterion == "gini":
            val = 1
            for c in classes:
                p = float(len(y[y == c])) / numdata
                val -= p ** 2.0
                
        #交差エントロピー
        elif self.criterion == "entropy":
            val = 0
            for c in classes:
                p = float(len(y[y == c])) / numdata
                if p!=0.0:
                    val -= p * np.log2(p)
                    
        elif self.criterion == "mse":
            val = 0.0
            y_mean = np.mean(y)
            for i in range(numdata):
                val += (y[i] - y_mean) ** 2
            val /= numdata
        return val

    def calc_info_gain(self, y_p, y_cl, y_cr):
        cri_p  = self.criterion_func(y_p)
        cri_cl = self.criterion_func(y_cl)
        cri_cr = self.criterion_func(y_cr)
        return cri_p - len(y_cl)/float(len(y_p))*cri_cl - len(y_cr)/float(len(y_p))*cri_cr

    def predict(self, X):
        #葉ノードの時
        if self.is_leaf == True:
            return self.label
        #内部ノードの時
        else:
            if X[self.feature] <= self.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
          
class DecisionTreeC(BaseEstimator, ClassifierMixin):
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.tree          = None
        self.criterion     = criterion
        self.max_depth     = max_depth
        self.random_state  = random_state

    def fit(self, X, y):
        self.tree = _Node(self.criterion, self.max_depth, self.random_state, False)
        self.tree.split_node(X, y, 0)

    def predict(self, X):
        pred = []
        for s in X:
            pred.append(self.tree.predict(s))
        return np.array(pred)
    
class DecisionTreeR(BaseEstimator, RegressorMixin):
    def __init__(self, criterion="mse", max_depth=None, random_state=None):
        self.tree          = None
        self.criterion     = criterion
        self.max_depth     = max_depth
        self.random_state  = random_state

    def fit(self, X, y):
        self.tree = _Node(self.criterion, self.max_depth, self.random_state, True)
        self.tree.split_node(X, y, 0)

    def predict(self, X):
        pred = []
        for s in X:
            pred.append(self.tree.predict(s))
        return np.array(pred)
    
