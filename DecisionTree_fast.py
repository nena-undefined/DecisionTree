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
        self.info_fain = 0.0

        
    def split_node(self, X, y, depth):
        self.depth = depth

        num_samples = len(y)
        num_features = X.shape[1]
        num_class = np.unique(y).shape[0]

        #データが一つの時，葉ノードにする
        if num_class == 1:
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
        self.info_gain = 0.0

        if self.random_state!=None:
            np.random.seed(self.random_state)
            
        #どの変数から見ていくかscikitの仕様と合わせている
        #（同じ情報利得が得られる変数が複数ある時に効いてくる）
        f_loop_order = np.random.permutation(num_features).tolist()
        
        #初期化
        if self.criterion == "mse":
            y_sum = np.sum(y) 
        #elif self.criterion == "gini" or self.criterion == "entropy":        
        else:
            class_count = {i: len(y[y==i]) for i in np.unique(y)}
            left_class = np.empty(num_class, int)
            right_class = np.empty(num_class, int)
            
            tmp_class =  np.empty(num_class, int)
            for i in range(num_class):
                tmp_class[i] = class_count[i]
            val = self.criterion_func(tmp_class, num_samples)
        
        for f in f_loop_order:
            
            if self.criterion == "mse":
                left_sum = 0.0
                right_sum = y_sum
                left_num = 0
                right_num = num_samples
                
            #elif self.criterion == "gini"　or self.criterion == "entropy":
            else:
                for i in range(num_class):
                    left_class[i] = 0
                    right_class[i] = class_count[i]
                    

            sort_idx = np.argsort(X[:, f])
            
            i = 0
            while i < num_samples-1:
                idx = sort_idx[i]

                if self.criterion == "mse":
                    left_sum += y[idx]
                    right_sum -= y[idx]

                    threshold = X[idx, f]

                    while i+1 <= num_samples-1 and threshold == X[sort_idx[i+1], f]:
                        left_sum += y[sort_idx[i+1]]
                        right_sum -= y[sort_idx[i+1]]
                        i += 1
                        
                else:
                    left_class[y[idx]] += 1
                    right_class[y[idx]] -= 1
                    
                    threshold = X[idx, f]
                    while i+1 <= num_samples-1 and threshold == X[sort_idx[i+1], f]:
                        left_class[y[sort_idx[i+1]]] += 1
                        right_class[y[sort_idx[i+1]]] -= 1
                        i += 1
                    
                if i is num_samples-1:
                    break
                    

                left_num = i + 1
                right_num = num_samples - left_num

                if left_num is 0 or right_num is 0:
                    continue
                    
                if self.criterion == "mse":
                    #calculate info_gain (Friedman_mse)
                    info_gain = left_num * right_num * ((left_sum / float(left_num) - right_sum / float(right_num)) ** 2)
                else:
                    info_gain = val - float(left_num) / float(num_samples) * self.criterion_func(left_class, left_num) - float(right_num) / float(num_samples) * self.criterion_func(right_class, right_num)
                    
                    
                if self.info_gain < info_gain:
                    self.info_gain = info_gain
                    self.feature = f
                    self.threshold = threshold

                i += 1
            
        #情報利得が0の時(良い分割点が見つからなかった時)
        if self.info_gain == 0.0:
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

    def criterion_func(self, class_count, num_samples):
        num_class = class_count.shape[0]

        #ジニ係数
        if self.criterion == "gini":
            val = 1
            for i in range(num_class):
                p = float(class_count[i]) / num_samples
                val -= p ** 2.0
                
        #交差エントロピー
        elif self.criterion == "entropy":
            val = 0
            for i in range(num_class):
                p = float(class_count[i]) / num_samples
                if p!=0.0:
                    val -= p * np.log2(p)
                    
        return val

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
    

