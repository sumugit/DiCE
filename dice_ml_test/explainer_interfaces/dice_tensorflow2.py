"""tensorflow 2.xでCF例を生成する方法"""
"""
Module to generate diverse counterfactual explanations based on tensorflow 2.x
"""
from explainer_interfaces.explainer_base import ExplainerBase
import tensorflow as tf

import numpy as np
import random
import timeit
import copy

import diverse_counterfactuals as exp
from counterfactual_explanations import CounterfactualExplanations
from constants import ModelTypes


class DiceTensorFlow2(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method
        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        # initiating data related parameters
        super().__init__(data_interface)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()

        # initializing model related variables
        self.model = model_interface
        self.model.load_model()  # loading trained model
        # TODO: this error is probably too big - need to change it.
        if self.model.transformer.func is not None:
            raise ValueError("Gradient-based DiCE currently "
                             "(1) accepts the data only in raw categorical and continuous formats, "
                             "(2) does one-hot-encoding and min-max-normalization internally, "
                             "(3) expects the ML model the accept the data in this same format. "
                             "If your problem supports this, please initialize model class again "
                             "with no custom transformation function.")
        # number of output nodes of ML model
        #if self.model.model_type == ModelTypes.Classifier:
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.ohe_encoded_feature_names)).shape[1]

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate

    # CF生成に関する事をまとめて実行する関数
    def generate_counterfactuals(self, query_instance, total_CFs, desired_range=None, desired_class="opposite", proximity_weight=0.5,
                                 diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF",
                                 features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                 diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad",
                                 optimizer="tensorflow:adam", learning_rate=0.05, min_iter=500, max_iter=5000,
                                 project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                 init_near_query_instance=True, tie_random=False, stopping_threshold=0.5,
                                 posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear"):
        """Generates diverse counterfactual explanations

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the
                              outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to
                                 the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical
                                    variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                               Defaults to the range inferred from training data. If None, uses the parameters initialized
                               in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding
                                weights as values. Default option is "inverse_mad" where the weight for a continuous feature
                                is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training
                                set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: Tensorflow optimization algorithm. Currently tested only with "tensorflow:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence.
                                      Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                (see diverse_counterfactuals.py).
        """
        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            # if not self.data_interface.check_features_range(permitted_range):
            #     raise ValueError(
            #         "permitted range of features should be within their original range")
            # else:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            # 連続変数の摂動制限区間の両端
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        # if([total_CFs, algorithm, features_to_vary] != self.cf_init_weights):
        # CFの初期化
        self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if [yloss_type, diversity_loss_type, feature_weights] != self.loss_weights:
            # 損失項の初期化
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if [proximity_weight, diversity_weight, categorical_penalty] != self.hyperparameters:
            # 各損失項のハイパーパラメータの初期化
            self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

        # CFの探索
        final_cfs_df, test_instance_df, final_cfs_df_sparse = \
            self.find_counterfactuals(query_instance, desired_range, desired_class, optimizer,
                                      learning_rate, min_iter, max_iter, project_iter,
                                      loss_diff_thres, loss_converge_maxiter, verbose,
                                      init_near_query_instance, tie_random, stopping_threshold,
                                      posthoc_sparsity_param, posthoc_sparsity_algorithm)

        # 生成したCFを格納, 可視化するための変数
        counterfactual_explanations = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_range=desired_range,
            desired_class=desired_class)

        # 重要度スコア用に返す
        return CounterfactualExplanations(cf_examples_list=[counterfactual_explanations])

    def predict_fn(self, input_instance):
        """テスト入力データに対して予測を行う"""
        """prediction function"""
        temp_preds = self.model.get_output(input_instance).numpy()
        return np.array([preds[(self.num_output_nodes-1):] for preds in temp_preds], dtype=np.float32)

    def predict_fn_for_sparsity(self, input_instance):
        """入力データに必要な前処理を施した上で予測"""
        """prediction function for sparsity correction"""
        # テスト入力データの前処理 (one-hot-encoding, min-max-scailing)
        input_instance = self.data_interface.get_ohe_min_max_normalized_data(input_instance).values
        # モデルの予測を行う
        return self.predict_fn(tf.constant(input_instance, dtype=tf.float32))

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """CFの初期化"""
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        # 生成CF数の設定
        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        # 固定する特徴量の設定
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
        self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)
        # 摂動可能な特徴量は1.0, それ以外は0.0. 勾配を計算する際に使う 
        self.freezer = tf.constant([1.0 if ix in self.feat_to_vary_idxs else 0.0 for ix in range(len(self.minx[0]))])

        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            for ix in range(self.total_CFs):
                one_init = [[]]
                for jx in range(self.minx.shape[1]):
                    # 摂動可能区間内の一様分布からランダムに値をサンプリング
                    one_init[0].append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                # 仮のCFを追加 (期待出力になるとは限らない)
                self.cfs.append(tf.Variable(one_init, dtype=tf.float32))

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
        """損失関数の各項の初期化"""
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1/normalized_mads[feature], 2)

            feature_weights_list = []
            for feature in self.data_interface.ohe_encoded_feature_names:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = tf.constant([feature_weights_list], dtype=tf.float32)

    def update_hyperparameters(self, proximity_weight, diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """再急降下法ベースによる最適化アルゴリズムの設定"""
        """Initializes gradient-based TensorFLow optimizers."""
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

    def compute_yloss(self, desired_range, desired_class):
        """ylossの定義"""
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        # 分類問題の時
        if self.model.model_type == ModelTypes.Classifier: # --added.
            for i in range(self.total_CFs):
                if self.yloss_type == "l2_loss":
                    temp_loss = tf.pow((self.model.get_output(self.cfs[i]) - self.target_cf_class), 2)
                    temp_loss = temp_loss[:, (self.num_output_nodes-1):][0][0]
                elif self.yloss_type == "log_loss":
                    temp_logits = tf.compat.v1.log((tf.abs(
                        self.model.get_output(
                            self.cfs[i]) - 0.000001))/(1 - tf.abs(self.model.get_output(self.cfs[i]) - 0.000001)))
                    temp_logits = temp_logits[:, (self.num_output_nodes-1):]
                    temp_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=temp_logits, labels=self.target_cf_class)[0][0]
                elif self.yloss_type == "hinge_loss":
                    # 0.000001はおそらく0除算の防止
                    temp_logits = tf.compat.v1.log((tf.abs(
                        self.model.get_output(
                            self.cfs[i]) - 0.000001))/(1 - tf.abs(self.model.get_output(self.cfs[i]) - 0.000001)))
                    temp_logits = temp_logits[:, (self.num_output_nodes-1):]
                    temp_loss = tf.compat.v1.losses.hinge_loss(
                        logits=temp_logits, labels=self.target_cf_class)
                yloss += temp_loss
        # 回帰問題
        elif self.model.model_type == ModelTypes.Regressor:
            for i in range(self.total_CFs):
                if self.yloss_type == "hinge_loss":
                    temp_loss = 0.0
                    # 期待出力域に収まっていなければ罰則
                    if not desired_range[0] <= self.model.get_output(self.cfs[i]) <= desired_range[1]:
                        temp_loss = max(abs(self.model.get_output(self.cfs[i]) - desired_range[0]),
                                       abs(self.model.get_output(self.cfs[i]) - desired_range[1]))
                yloss += temp_loss            

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """重み付き絶対誤差の全特徴量分の和"""
        """Compute weighted distance between two vectors."""
        return tf.reduce_sum(tf.multiply((tf.abs(x_hat - x1)), self.feature_weights_list))

    def compute_proximity_loss(self):
        """近傍損失"""
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        # 生成CF数分だけ以下を繰り返す
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        # 特徴量の個数および生成CFで最後に除算 (2重シグマの形になっている)
        return proximity_loss/tf.cast((tf.multiply(len(self.minx[0]), self.total_CFs)), dtype=tf.float32)

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = []
        # 行列Kを作成 (各要素の距離が重み付き絶対誤差の逆数)
        if submethod == "inverse_dist":
            # iのサイズとjのサイズは同じ
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.add(
                        1.0, self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_temp_entry = tf.add(det_temp_entry, 0.0001)
                    det_entries.append(det_temp_entry)
        
        # 行列Kを作成 (各要素の距離が重み付き絶対誤差の指数の逆数)
        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.exp(
                        self.compute_dist(self.cfs[i], self.cfs[j])))
                    det_entries.append(det_temp_entry)

        # 行列の形に変換
        det_entries = tf.reshape(det_entries, [self.total_CFs, self.total_CFs])
        # 行列式を計算
        diversity_loss = tf.compat.v1.matrix_determinant(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """dpp_diversityの定義"""
        """Computes the third part (diversity) of the loss function."""
        # 生成CF数が1の時はそもそもdiversityを香料する必要がない
        if self.total_CFs == 1:
            return tf.constant(0.0)

        # 行列式の計算
        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return tf.reduce_sum(self.dpp_style(submethod))
        # 
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            # 各行列の要素の値の平均値を返す
            # 対称行列だから上半分のみ計算すればよい
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
            # 1から引いてるから値が大きいほど損失が大きい (つまり良い)
            return 1.0 - (diversity_loss/count)

    def compute_regularization_loss(self):
        """各カテゴリ変数の水準がどれか一つに強く反応するようにする損失の定義"""
        """Adds a linear equality constraints to the loss functions - to ensure all levels
           of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            # v は各カテゴリ変数の水準
            # (tf.reduce_sum(self.cfs[i][0, v[0]:v[-1]+1])が１になる, つまりone-hot-encodingに近づくように最適化
            # 各水準の値に散らばりがあると損失は大きくなる
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += tf.pow((tf.reduce_sum(self.cfs[i][0, v[0]:v[-1]+1]) - 1.0), 2)

        return regularization_loss

    def compute_loss(self, desired_range, desired_class):
        """全体の最適化問題"""
        """Computes the overall loss"""
        self.yloss = self.compute_yloss(desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        self.regularization_loss = self.compute_regularization_loss()

        # 全体の損失の計算
        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - \
            (self.diversity_weight * self.diversity_loss) + \
            (self.categorical_penalty * self.regularization_loss)
        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """CF の初期化. do_cf_initializationsは多分初回のみ実行?"""
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            one_init = []
            for i in range(len(self.minx[0])):
                # 摂動させる変数のindexのみ参照
                if i in self.feat_to_vary_idxs:
                    # テスト入力データの特徴量iに近ければ?
                    if init_near_query_instance:
                        one_init.append(query_instance[0][i]+(n*0.01))
                    # 特徴量iの摂動可能範囲の一様分布からの乱数
                    else:
                        one_init.append(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    one_init.append(query_instance[0][i])
            one_init = np.array([one_init], dtype=np.float32)
            self.cfs[n].assign(one_init)

    def round_off_cfs(self, assign=False):
        """CFの中間予測を整形する関数"""
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.numpy()
            # vは連続変数
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[0, v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                org_cont = round(org_cont, self.cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                # min-maxスケーリング
                cf[0, v] = normalized_cont  # assign the projected continuous value

            # vはカテゴリ変数の水準
            for v in self.encoded_categorical_feature_indexes:
                # 仮のCFの各カテゴリ変数で最大値となる水準のindexリスト作成
                # np.argwhereは引数の条件を満たすindexを返す
                # np.amaxは最大の要素を返す
                maxs = np.argwhere(
                    cf[0, v[0]:v[-1]+1] == np.amax(cf[0, v[0]:v[-1]+1])).flatten().tolist()
                # 最大の水準の候補が複数あればランダムにそのうちの一つを選択
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                # 最大の候補が一つであればそれを選択
                else:
                    ix = maxs[0]
                # 選択された水準を1.0にし, それ以外を0.0にする
                for vi in range(len(v)):
                    if vi == ix:
                        cf[0, v[vi]] = 1.0
                    else:
                        cf[0, v[vi]] = 0.0

            temp_cfs.append(cf)
            if assign:
                self.cfs[index].assign(temp_cfs[index])

        if assign:
            return None
        else:
            return temp_cfs

    def stop_loop(self, itr, loss_diff):
        """勾配降下法の終了タイミングを設定する関数"""
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if self.project_iter > 0 and itr > 0:
            if itr % self.project_iter == 0:
                self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        # 今の時点の損失値とその前の時点の損失値の絶対差を計算
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                # 中間CFデータの整形
                temp_cfs = self.round_off_cfs(assign=False)
                # MLモデルで予測
                test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32))[0] for cf in temp_cfs]
                
                # 分類問題の時
                if self.model.model_type == ModelTypes.Classifier: # --added.
                    # 全ての予測値が期待クラス (確率の閾値stopping_threshold) に属していたら終了
                    if self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                        self.converged = True
                        return True
                    elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                        self.converged = True
                        return True
                    # 一つでも期待クラスに属していないCFがあれば失敗
                    else:
                        return False
                # 回帰問題の時
                elif self.model.model_type == ModelTypes.Regressor:
                    if all(self.target_cf_range[0] <= i <= self.target_cf_range[1] for i in test_preds):
                        self.converged = True
                        return True
                    else:
                        return False
                    
        # lossが改悪されたらiterを0に戻す
        else:
            self.loss_converge_iter = 0
            return False

    def find_counterfactuals(self, query_instance, desired_range, desired_class, optimizer, learning_rate, min_iter,
                             max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
                             init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
                             posthoc_sparsity_algorithm):
        """勾配降下法を用いてCFを見つける"""
        """Finds counterfactuals by gradient-descent."""

        # Prepares user defined query_instance for DiCE.
        # query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encoding='one-hot')
        # query_instance = np.array([query_instance.iloc[0].values])
        # 前処理を施したテスト入力データ
        query_instance = self.data_interface.get_ohe_min_max_normalized_data(query_instance).values
        # 定数に変換
        self.x1 = tf.constant(query_instance, dtype=tf.float32)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(tf.constant(query_instance, dtype=tf.float32))[0][0]
        # 期待出力域の設定 -- added.
        if desired_range is not None:
            if desired_range[0] > desired_range[1]:
                raise ValueError("Invalid Range!")
            self.target_cf_range = desired_range
        # 期待クラスの設定
        print(self.model.model_type)
        if desired_class == "opposite" and self.model.model_type == ModelTypes.Classifier: #--added.
            desired_class = 1.0 - round(test_pred) # roundで確率を0,1に変換
            self.target_cf_class = np.array([[desired_class]], dtype=np.float32)

        # 必要なオプションの設定
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        # 分類問題の時
        if self.model.model_type == ModelTypes.Classifier: # --added.
            if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
                self.stopping_threshold = 0.25
            elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
                self.stopping_threshold = 0.75

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        # 最終的に得られるCFの格納場所
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

        # variables to backup best known CFs so far in the optimization process -
        # if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.min_dist_from_threshold = [100]*loop_find_CFs  # for backup CFs

        for loop_ix in range(loop_find_CFs):
            # CF init
            # CFの初期化
            if self.total_random_inits > 0:
                self.initialize_CFs(query_instance, False)
            else:
                self.initialize_CFs(query_instance, init_near_query_instance)

            # initialize optimizer
            # 最適化パラメータの初期化
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            ### 終了しない間次の最適化を繰り返す ###
            while self.stop_loop(iterations, loss_diff) is False:

                # compute loss and tape the variables history
                # tf.GradientTapeクラスをインスタンス化することで傾き (勾配) を求められる
                with tf.GradientTape() as tape:
                    # 全体の損失を計算
                    loss_value = self.compute_loss(desired_range, desired_class)

                # get gradients
                # 各特徴量の傾き (勾配) の取得
                grads = tape.gradient(loss_value, self.cfs)

                # freeze features other than feat_to_vary_idxs
                # 摂動不可能な特徴量の勾配は常に0
                for ix in range(self.total_CFs):
                    grads[ix] *= self.freezer

                # apply gradients and update the variables
                # 最急降下方向に各特徴量を更新
                self.optimizer.apply_gradients(zip(grads, self.cfs))

                # projection step
                for j in range(0, self.total_CFs):
                    temp_cf = self.cfs[j].numpy()
                    # 更新した各特徴量を摂動可能な区間に収まるように無理矢理整形
                    clip_cf = np.clip(temp_cf, self.minx, self.maxx)  # clipping
                    # to remove -ve sign before 0.0 in some cases
                    clip_cf = np.add(clip_cf, np.array(
                        [np.zeros([self.minx.shape[1]])]))
                    self.cfs[j].assign(clip_cf)

                # 勾配計算の記録
                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                # 今の時点の損失値とその前の時点の損失値の絶対差を計算
                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                # 現在のCFとMLモデルの予測値を保存
                temp_cfs_stored = self.round_off_cfs(assign=False)
                test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs_stored]

                # 全てのCFの予測値が閾値を超えていたら最適なCFの組み合わせと閾値を更新
                # 分類問題の時
                if self.model.model_type == ModelTypes.Classifier: # --added.
                    if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or
                    (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                        avg_preds_dist = np.mean([abs(pred[0][0]-self.stopping_threshold) for pred in test_preds_stored])
                        if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                            self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                            for ix in range(self.total_CFs):
                                self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(temp_cfs_stored[ix])
                                self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])
                # 回帰問題の時
                elif self.model.model_type == ModelTypes.Regressor: #--added.
                    if all(self.target_cf_range[0] <= i <= self.target_cf_range[1] for i in test_preds_stored):
                        avg_preds_dist = np.mean(test_preds_stored)
                        if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                            self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                            for ix in range(self.total_CFs):
                                self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(temp_cfs_stored[ix])
                                self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])

            ### 勾配計算終了 ###
            # rounding off final cfs - not necessary when inter_project=True
            # 得られたCFデータセットを整形
            self.round_off_cfs(assign=True)

            # storing final CFs
            # 最終的に得られたCFデータセットを格納
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].numpy()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations
        
        # 計算時間の記録
        self.elapsed = timeit.default_timer() - start_time

        # 予測値の記録
        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        # いくつかのCFの予測値が閾値を超えていなかったら今までの中で最適なCFの組み合わせを得る
        # 分類問題の時
        if self.model.model_type == ModelTypes.Classifier: # --added.
            if((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in self.cfs_preds)) or
            (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in self.cfs_preds))):
                for loop_ix in range(loop_find_CFs):
                    if self.min_dist_from_threshold[loop_ix] != 100:
                        for ix in range(self.total_CFs):
                            self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                            self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])
        # 回帰問題の時
        elif self.model.model_type == ModelTypes.Regressor: # --added.
            if not all(self.target_cf_range[0] <= i <= self.target_cf_range[1] for i in self.cfs_preds):
                for loop_ix in range(loop_find_CFs):
                    if self.min_dist_from_threshold[loop_ix] != 100:
                        for ix in range(self.total_CFs):
                            self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                            self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])       

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])
        final_cfs_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(cfs)
        cfs_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds]
        cfs_preds = [item for sublist in cfs_preds for item in sublist]
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)

        test_instance_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(query_instance)
        test_instance_df[self.data_interface.outcome_name] = np.array(np.round(test_pred, 3))

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and \
                'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(
                final_cfs_df_sparse, test_instance_df, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            final_cfs_df_sparse = None
        # need to check the above code on posthoc sparsity

        # if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
        #     final_cfs_sparse = copy.deepcopy(self.final_cfs)
        #     cfs_preds_sparse = copy.deepcopy(self.cfs_preds)
        #     self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(
        #           self.total_CFs, final_cfs_sparse, cfs_preds_sparse, query_instance, posthoc_sparsity_param,
        #           posthoc_sparsity_algorithm, total_random_inits=self.total_random_inits)
        # else:
        #     self.final_cfs_sparse = None
        #     self.cfs_preds_sparse = None

        # 可視化に必要な処理
        # 分類問題の時
        if self.model.model_type == ModelTypes.Classifier: # --added.
            m, s = divmod(self.elapsed, 60)
            if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or
            (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))):
                self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
                valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
                print('Diverse Counterfactuals found! total time taken: %02d' %
                    m, 'min %02d' % s, 'sec')
            else:
                self.total_CFs_found = 0
                valid_ix = []  # indexes of valid CFs
                for cf_ix, pred in enumerate(self.cfs_preds):
                    if((self.target_cf_class == 0 and pred < self.stopping_threshold) or
                    (self.target_cf_class == 1 and pred > self.stopping_threshold)):
                        self.total_CFs_found += 1
                        valid_ix.append(cf_ix)

                if self.total_CFs_found == 0:
                    print('No Counterfactuals found for the given configuation, perhaps try with different ',
                        'values of proximity (or diversity) weights or learning rate...',
                        '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
                else:
                    print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                        ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                        'values of proximity (or diversity) weights or learning rate...',
                        '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
        # 回帰問題の時
        elif self.model.model_type == ModelTypes.Regressor: # --added.
            m, s = divmod(self.elapsed, 60)
            if all(self.target_cf_range[0] <= i <= self.target_cf_range[1] for i in self.cfs_preds):
                self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
                valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
                print('Diverse Counterfactuals found! total time taken: %02d' %
                    m, 'min %02d' % s, 'sec')
            else:
                self.total_CFs_found = 0
                valid_ix = []  # indexes of valid CFs
                for cf_ix, pred in enumerate(self.cfs_preds):
                    if self.target_cf_range[0] <= pred <= self.target_cf_range[1]:
                        self.total_CFs_found += 1
                        valid_ix.append(cf_ix)

                if self.total_CFs_found == 0:
                    print('No Counterfactuals found for the given configuation, perhaps try with different ',
                        'values of proximity (or diversity) weights or learning rate...',
                        '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
                else:
                    print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                        ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                        'values of proximity (or diversity) weights or learning rate...',
                        '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)
        # returning only valid CFs
        return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse
