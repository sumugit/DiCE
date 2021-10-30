""" 遺伝的アルゴリズムによるCF生成手法 """
"""
Module to generate diverse counterfactual explanations based on genetic algorithm
This code is similar to 'GeCo: Quality Counterfactual Explanations in Real Time': https://arxiv.org/pdf/2101.01292.pdf
"""
from explainer_interfaces.explainer_base import ExplainerBase
import numpy as np
import pandas as pd
import random
import timeit
import copy
from sklearn.preprocessing import LabelEncoder

import diverse_counterfactuals as exp
from constants import ModelTypes


class DiceGenetic(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        super().__init__(data_interface, model_interface)  # initiating data related parameters

        # number of output nodes of ML model
        # 分類問題の時
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = self.model.get_num_output_nodes2(
                self.data_interface.data_df[0:1][self.data_interface.feature_names])

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        # 遺伝子の個数 (優秀な遺伝子の上位total_CFs分を生成)
        self.population_size = 5000

        # Initializing a label encoder to obtain label-encoded values for categorical variables
        self.labelencoder = {}

        self.label_encoded_data = self.data_interface.data_df.copy()

        for column in self.data_interface.categorical_feature_names:
            self.labelencoder[column] = LabelEncoder()
            self.label_encoded_data[column] = self.labelencoder[column].fit_transform(
                self.data_interface.data_df[column])

        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

    def update_hyperparameters(self, proximity_weight, sparsity_weight,
                               diversity_weight, categorical_penalty):
        """ハイパーパラメータの設定"""
        """Update hyperparameters of the loss function"""

        self.proximity_weight = proximity_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights,
                                encoding='one-hot'):
        """損失項の種類と特徴量の重みを設定"""
        """Intializes variables related to main loss function"""

        # 各損失項のハイパーパラメータ
        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]
        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        # define feature weights
        # 各特徴量の重み設定
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            # MAD_pによる除算
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=False)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1 / normalized_mads[feature], 2)

            # 特徴量重みを反映
            feature_weights_list = []
            if encoding == 'one-hot':
                for feature in self.data_interface.encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
            # こっちは多分まだ未完成
            elif encoding == 'label':
                for feature in self.data_interface.feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        # TODO: why is the weight the max value of the encoded feature
                        feature_weights_list.append(self.label_encoded_data[feature].max())
            self.feature_weights_list = [feature_weights_list]

    def do_random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range):
        """テスト入力データにランダムな摂動を加えたCFで期待出力を得るものをnum_inits分生成"""
        # ランダムに特定の特徴量を変化させて期待出力を得たCFの集合格納庫
        remaining_cfs = np.zeros((num_inits, self.data_interface.number_of_features))
        # kx is the number of valid inits found so far
        kx = 0
        precisions = self.data_interface.get_decimal_precisions()
        # remaing_cfsの数がnum_initsより小さい間繰り返す
        while kx < num_inits:
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                # 摂動可能な特徴量の時
                if feature in features_to_vary:
                    # それが連続変数であれば
                    if feature in self.data_interface.continuous_feature_names:
                        # 摂動可能範囲の一様分布からランダムに値を取得
                        one_init[jx] = np.round(np.random.uniform(
                            self.feature_range[feature][0], self.feature_range[feature][1]), precisions[jx])
                    else:
                        # カテゴリ変数であればランダムに一つの水準を選択
                        one_init[jx] = np.random.choice(self.feature_range[feature])
                # 摂動不可能な特徴量はテスト入力データの値をコピー
                else:
                    one_init[jx] = query_instance[jx]
            # ランダムに特定の特徴量を変化させたCFが有効であれば
            if self.is_cf_valid(self.predict_fn_scores(one_init)):
                remaining_cfs[kx] = one_init
                kx += 1
        return remaining_cfs

    def do_KD_init(self, features_to_vary, query_instance, cfs, desired_class, desired_range):
        """テスト入力データ, CFの特徴量が既に摂動可能範囲に収まっていればコピーしてそれ以外はランダムサンプリング"""
        cfs = self.label_encode(cfs)
        cfs = cfs.reset_index(drop=True)

        # 初期の遺伝子(CF)は行数がpopulations_sizeで特徴量の数がnumber_of_features
        self.cfs = np.zeros((self.population_size, self.data_interface.number_of_features))
        for kx in range(self.population_size):
            if kx >= len(cfs):
                break
            one_init = np.zeros(self.data_interface.number_of_features)
            # 各特徴量について
            for jx, feature in enumerate(self.data_interface.feature_names):
                # 摂動不可能であればコピー
                if feature not in features_to_vary:
                    one_init[jx] = (query_instance[jx])
                # 摂動可能であれば
                else:
                    # 連続変数であれば
                    if feature in self.data_interface.continuous_feature_names:
                        # 既に摂動可能範囲に収まっていればコピー
                        if self.feature_range[feature][0] <= cfs.iat[kx, jx] <= self.feature_range[feature][1]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            # テスト入力データが既に摂動可能範囲に収まっていればそれをコピー
                            if self.feature_range[feature][0] <= query_instance[jx] <= self.feature_range[feature][1]:
                                one_init[jx] = query_instance[jx]
                            # そうでなければ一様分布からサンプリング
                            else:
                                one_init[jx] = np.random.uniform(
                                    self.feature_range[feature][0], self.feature_range[feature][1])
                    # カテゴリ変数であれば
                    else:
                        # 既に摂動可能範囲に収まっていればコピー
                        if cfs.iat[kx, jx] in self.feature_range[feature]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            # テスト入力データが既に摂動可能範囲に収まっていればそれをコピー
                            if query_instance[jx] in self.feature_range[feature]:
                                one_init[jx] = query_instance[jx]
                            # そうでなければランダムに水準を選択
                            else:
                                one_init[jx] = np.random.choice(self.feature_range[feature])
            self.cfs[kx] = one_init
            kx += 1

        # 得られたCF間の重複を除く
        new_array = [tuple(row) for row in self.cfs]
        uniques = np.unique(new_array, axis=0)

        # ユニークなCFがpopulation_sizeと一致しなかったら重複分はランダムに生成
        if len(uniques) != self.population_size:
            remaining_cfs = self.do_random_init(
                self.population_size - len(uniques), features_to_vary, query_instance, desired_class, desired_range)
            self.cfs = np.concatenate([uniques, remaining_cfs])

    def do_cf_initializations(self, total_CFs, initialization, algorithm, features_to_vary, desired_range,
                              desired_class,
                              query_instance, query_instance_df_dummies, verbose):
        """初期のCF生成"""
        """Intializes CFs and other related variables."""
        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1  # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        self.features_to_vary = features_to_vary

        # CF initialization
        self.cfs = []
        # 一様分布によるランダムな初期化
        if initialization == 'random':
            self.cfs = self.do_random_init(
                self.population_size, features_to_vary, query_instance, desired_class, desired_range)

        # kd-treeによる初期化
        elif initialization == 'kdtree':
            # Partitioned dataset and KD Tree for each class (binary) of the dataset
            # kd-treeの作成および期待出力となるCFを全て抽出
            self.dataset_with_predictions, self.KD_tree, self.predictions = \
                self.build_KD_tree(self.data_interface.data_df.copy(),
                                   desired_range, desired_class, self.predicted_outcome_name)
            # kd-treeが作成できなかったら一様分布によるランダムな初期化
            if self.KD_tree is None:
                self.cfs = self.do_random_init(
                    self.population_size, features_to_vary, query_instance, desired_class, desired_range)

            # kd-treeが作成できたらテスト入力データに近いCFから順に必要数分取得
            else:
                # CF取得数
                num_queries = min(len(self.dataset_with_predictions), self.population_size * self.total_CFs)
                # テスト入力データに近接するCFのindex取得
                indices = self.KD_tree.query(query_instance_df_dummies, num_queries)[1][0]
                # 取得したindexに対応するCFを抽出
                KD_tree_output = self.dataset_with_predictions.iloc[indices].copy()
                # 取得したCFのkd-treeに対応した初期化
                self.do_KD_init(features_to_vary, query_instance, KD_tree_output, desired_class, desired_range)

        if verbose:
            print("Initialization complete! Generating counterfactuals...")

    def do_param_initializations(self, total_CFs, initialization, desired_range, desired_class,
                                 query_instance, query_instance_df_dummies, algorithm, features_to_vary,
                                 permitted_range, yloss_type, diversity_loss_type, feature_weights,
                                 proximity_weight, sparsity_weight, diversity_weight, categorical_penalty, verbose):
        """CF生成に関する各パラメータの初期化"""
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.feature_range = self.get_valid_feature_range(normalized=False)
        if len(self.cfs) != total_CFs:
            # CFの初期化
            self.do_cf_initializations(
                total_CFs, initialization, algorithm, features_to_vary, desired_range, desired_class,
                query_instance, query_instance_df_dummies, verbose)
        else:
            self.total_CFs = total_CFs
        # 損失関数の初期化
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        # パラメータの初期化
        self.update_hyperparameters(proximity_weight, sparsity_weight, diversity_weight, categorical_penalty)

    def _generate_counterfactuals(self, query_instance, total_CFs, initialization="kdtree", desired_range=None,
                                  desired_class="opposite", proximity_weight=0.2, sparsity_weight=0.2,
                                  diversity_weight=5.0, categorical_penalty=0.1, algorithm="DiverseCF",
                                  features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                  diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad",
                                  stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="binary",
                                  maxiterations=500, thresh=1e-2, verbose=False, is_research=False):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param initialization: Method to use to initialize the population of the genetic algorithm
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: For classification problems. Desired counterfactual class - can take 0 or 1.
                              Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are. ← 使わない
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                                Defaults to the range inferred from training data. If None, uses the parameters initialized
                                in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and
                                corresponding weights as values. Default option is "inverse_mad" where the
                                weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD)
                                of the feature's values in the training set; the weight for a categorical feature is
                                equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param maxiterations: Maximum iterations to run the genetic algorithm for.
        :param thresh: The genetic algorithm stops when the difference between the previous best loss and current
                       best loss is less than thresh
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                 (see diverse_counterfactuals.py).
        """
        self.start_time = timeit.default_timer()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        # Prepares user defined query_instance for DiCE.
        # テスト入力データの前処理
        query_instance_orig = query_instance
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance)
        query_instance = self.label_encode(query_instance)
        query_instance = np.array(query_instance.values[0])
        self.x1 = query_instance

        # find the predicted value of query_instance
        # テスト入力データの予測値
        test_pred = self.predict_fn(query_instance)
        self.test_pred = test_pred

        desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred)
        
        # テスト入力データの前処理 (one-hot-encoding)
        query_instance_df_dummies = pd.get_dummies(query_instance_orig)
        for col in pd.get_dummies(self.data_interface.data_df[self.data_interface.feature_names]).columns:
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        # 各パラメータの初期化
        self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance,
                                      query_instance_df_dummies, algorithm, features_to_vary, permitted_range,
                                      yloss_type, diversity_loss_type, feature_weights, proximity_weight,
                                      sparsity_weight, diversity_weight, categorical_penalty, verbose)

        # CF生成
        query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary,
                                                      maxiterations, thresh, verbose, is_research)

        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          test_instance_df=query_instance_df,
                                          final_cfs_df=self.final_cfs_df,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class,
                                          model_type=self.model.model_type)

    def predict_fn_scores(self, input_instance):
        """Returns prediction scores."""
        """MLモデルの予測値を出力 (離散, 回帰), 回帰問題であればpredict_fn_scores, predict_fnは同じ処理"""
        input_instance = self.label_decode(input_instance)
        return self.model.get_output(input_instance)

    def predict_fn(self, input_instance):
        """Returns actual prediction."""
        """MLモデルの予測値を出力 (確率, 回帰)"""
        input_instance = self.label_decode(input_instance)
        return self.model.get_output(input_instance, model_score=False)

    def _predict_fn_custom(self, input_instance, desired_class):
        """Checks that the maximum predicted score lies in the desired class."""
        """The reason we do so can be illustrated by
        this example: If the predict probabilities are [0, 0.5, 0,5], the computed yloss is 0 as class 2 has the same
        value as the maximum score. sklearn's usual predict function, which implements argmax, returns class 1 instead
        of 2. This is why we need a custom predict function that returns the desired class if the maximum predict
        probability is the same as the probability of the desired class."""

        input_instance = self.label_decode(input_instance)
        output = self.model.get_output(input_instance, model_score=True)
        desired_class = int(desired_class)
        maxvalues = np.max(output, 1)
        predicted_values = np.argmax(output, 1)

        # We iterate through output as we often call _predict_fn_custom for multiple inputs at once
        for i in range(len(output)):
            if output[i][desired_class] == maxvalues[i]:
                predicted_values[i] = desired_class

        return predicted_values

    def compute_yloss(self, cfs, desired_range, desired_class, is_research):
        """期待出力となるような損失関数"""
        """Computes the first part (y-loss) of the loss function."""
        # 単純な誤差 --added.
        if is_research:
            predicted_value = self.predict_fn(cfs)
            yloss = np.zeros(len(predicted_value))
            for i in range(len(predicted_value)):
                yloss[i] = self.test_pred - predicted_value[i]
        else:
            # ylossの設定
            yloss = 0.0
            # 分類問題の場合
            if self.model.model_type == ModelTypes.Classifier:
                predicted_value = np.array(self.predict_fn_scores(cfs))
                if self.yloss_type == 'hinge_loss':
                    maxvalue = np.full((len(predicted_value)), -np.inf)
                    for c in range(self.num_output_nodes):
                        if c != desired_class:
                            maxvalue = np.maximum(maxvalue, predicted_value[:, c])
                    yloss = np.maximum(0, maxvalue - predicted_value[:, int(desired_class)])

            # 回帰問題の場合   
            elif self.model.model_type == ModelTypes.Regressor:
                predicted_value = self.predict_fn(cfs)
                if self.yloss_type == 'hinge_loss':
                    yloss = np.zeros(len(predicted_value))
                    for i in range(len(predicted_value)):
                        if not desired_range[0] <= predicted_value[i] <= desired_range[1]:
                            yloss[i] = min(abs(predicted_value[i] - desired_range[0]),
                                        abs(predicted_value[i] - desired_range[1]))
        return yloss

    def compute_proximity_loss(self, x_hat_unnormalized, query_instance_normalized):
        """２つの連続変数のベクトル間距離を表す損失関数"""
        """Compute weighted distance between two vectors."""
        # Distの設定
        x_hat = self.data_interface.normalize_data(x_hat_unnormalized)
        feature_weights = np.array(
            [self.feature_weights_list[0][i] for i in self.data_interface.continuous_feature_indexes])
        # query_instance と cf の標準化l1_ノルムに特徴量の重みを掛けて足す
        product = np.multiply(
            (abs(x_hat - query_instance_normalized)[:, [self.data_interface.continuous_feature_indexes]]),
            feature_weights)
        product = product.reshape(-1, product.shape[-1])
        proximity_loss = np.sum(product, axis=1)

        # Dividing by the sum of feature weights to normalize proximity loss
        return proximity_loss / sum(feature_weights)

    def compute_sparsity_loss(self, cfs):
        """摂動した特徴量に罰則を加える損失関数"""
        """Compute weighted distance between two vectors."""
        # 元の入力ベクトルから変化した特徴量をカウント
        sparsity_loss = np.count_nonzero(cfs - self.x1, axis=1)
        return sparsity_loss / len(
            self.data_interface.feature_names)  # Dividing by the number of features to normalize sparsity loss

    def compute_loss(self, cfs, desired_range, desired_class, is_research):
        """全体の損失関数"""
        """Computes the overall loss"""
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class, is_research)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 else 0.0
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        self.loss = np.reshape(np.array(self.yloss + (self.proximity_weight * self.proximity_loss) +
                                        self.sparsity_weight * self.sparsity_loss), (-1, 1))
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss

    def mate(self, k1, k2, features_to_vary, query_instance):
        """交叉と子孫"""
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        # 子孫のための染色体
        one_init = np.zeros(self.data_interface.number_of_features)
        for j in range(self.data_interface.number_of_features):
            # 親1,2のj番目の特徴量を参照
            gp1 = k1[j]
            gp2 = k2[j]
            feat_name = self.data_interface.feature_names[j]

            # random probability
            prob = random.random()

            # 確率的に交叉
            if prob < 0.40:
                # if prob is less than 0.40, insert gene from parent 1
                one_init[j] = gp1
            elif prob < 0.80:
                # if prob is between 0.40 and 0.80, insert gene from parent 2
                one_init[j] = gp2
            # 確率20%で突然変異
            else:
                # otherwise insert random gene(mutate) for maintaining diversity
                if feat_name in features_to_vary:
                    # 連続変数ならば
                    if feat_name in self.data_interface.continuous_feature_names:
                        # 摂動可能区間の一様分布から取得
                        one_init[j] = np.random.uniform(self.feature_range[feat_name][0],
                                                        self.feature_range[feat_name][0])
                    else:
                        # カテゴリ変数ならランダムに水準を選択
                        one_init[j] = np.random.choice(self.feature_range[feat_name])
                else:
                    one_init[j] = query_instance[j]
        # 新しい子孫を返す
        return one_init

    def find_counterfactuals(self, query_instance, desired_range, desired_class,
                             features_to_vary, maxiterations, thresh, verbose, is_research):
        """遺伝的アルゴリズムによってCFを生成する"""
        """Finds counterfactuals by generating cfs through the genetic algorithm"""
        population = self.cfs.copy()
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        cfs_preds = [np.inf] * self.total_CFs
        to_pred = None

        self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
        self.query_instance_normalized = self.query_instance_normalized.astype('float')

        self.average_loss_list = []
        self.average_top_loss_list = []

        # maxiterationの数だけ以下を繰り返す
        while iterations < maxiterations and self.total_CFs > 0:
            # 全ての中間CFが期待出力となっていて且つlossの絶対差が一定の閾値を下回れば
            """             if abs(previous_best_loss - current_best_loss) <= thresh and \
                                (self.model.model_type == ModelTypes.Classifier and all(i == desired_class for i in cfs_preds) or
                                (self.model.model_type == ModelTypes.Regressor and
                                all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
                            stop_cnt += 1
                        # そうでなければ
                        else:
                            stop_cnt = 0
                        # 5step連続で全ての中間CFが期待出力となれば終了
                        if stop_cnt >= 5:
                            print("stop_cntで停止.")
                            break
            """            
            previous_best_loss = current_best_loss
            population = np.unique(tuple(map(tuple, population)), axis=0)

            # 損失値の計算
            population_fitness = self.compute_loss(population, desired_range, desired_class, is_research)
            population_fitness = population_fitness[population_fitness[:, 1].argsort()]
            #print("世代", iterations, "loss", np.average(population_fitness))
            self.average_loss_list.append(np.average([val[1] for val in population_fitness]))

            current_best_loss = population_fitness[0][1]
            # CF生成数分lossを保存
            to_pred = np.array([population[int(tup[0])] for tup in population_fitness[:self.total_CFs]])

            if self.total_CFs > 0:
                # 分類問題であれば
                if self.model.model_type == ModelTypes.Classifier:
                    cfs_preds = self._predict_fn_custom(to_pred, desired_class)
                # 回帰問題であれば
                else:
                    cfs_preds = self.predict_fn(to_pred)

            # self.total_CFS of the next generation obtained from the fittest members of current generation
            # 選択
            top_members = self.total_CFs
            # 上位の平均lossを保存
            self.average_top_loss_list.append(np.average([val[1] for val in population_fitness[:top_members]]))
            # lossのスコア上位total_CFs分だけ次の世代に受け継がれる
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])

            # rest of the next generation obtained from top 50% of fittest members of current generation
            # スコア上位total_CFs分に含まれなかったCFの数
            rest_members = self.population_size - top_members
            # スコア上位total_CFs分に含まれなかったCFは上位50%のCF同士(親)の子孫に置き換える
            new_generation_2 = np.zeros((rest_members, self.data_interface.number_of_features))
            for new_gen_idx in range(rest_members):
                # 親1
                parent1 = random.choice(population[:int(len(population) / 2)])
                # 親2
                parent2 = random.choice(population[:int(len(population) / 2)])
                # 交叉
                child = self.mate(parent1, parent2, features_to_vary, query_instance)
                new_generation_2[new_gen_idx] = child

            if self.total_CFs > 0:
                # 新しい世代はlossのスコア上位total_CFs分とそれらの親から交叉された子で構成 (データ数は同じ)
                population = np.concatenate([new_generation_1, new_generation_2])
            else:
                population = new_generation_2
            iterations += 1

        # 最終的に得られたCFの格納庫
        self.cfs_preds = []
        self.final_cfs = []
        i = 0
        # 期待出力に収まったかどうかは気にしない -- added.
        if is_research:
            while i < self.total_CFs:
                predictions =  self.predict_fn_scores(population[i])[0]
                self.final_cfs.append(population[i])
                self.cfs_preds.append(predictions)
                i += 1
        else:
            while i < self.total_CFs:
                predictions = self.predict_fn_scores(population[i])[0]
                # 期待出力になれば合格
                if self.is_cf_valid(predictions):
                    self.final_cfs.append(population[i])
                    # 分類問題の時
                    if not isinstance(predictions, float) and len(predictions) > 1:
                        self.cfs_preds.append(np.argmax(predictions))
                    # 回帰問題の時
                    else:
                        self.cfs_preds.append(predictions)
                i += 1
 
        # converting to dataframe
        query_instance_df = self.label_decode(query_instance)
        query_instance_df[self.data_interface.outcome_name] = self.test_pred
        self.final_cfs_df = self.label_decode_cfs(self.final_cfs)
        self.final_cfs_df_sparse = copy.deepcopy(self.final_cfs_df)

        if self.final_cfs_df is not None:
            self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
            self.final_cfs_df_sparse[self.data_interface.outcome_name] = self.cfs_preds
            self.round_to_precision()

        # 実行時間計測
        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            if is_research:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
            else:
                if len(self.final_cfs) == self.total_CFs:
                    print('Diverse Counterfactuals found! total time taken: %02d' %
                        m, 'min %02d' % s, 'sec')
                else:
                    print('Only %d (required %d) ' % (len(self.final_cfs), self.total_CFs),
                        'Diverse Counterfactuals found for the given configuation, perhaps ',
                        'change the query instance or the features to vary...'  '; total time taken: %02d' % m,
                        'min %02d' % s, 'sec')

        return query_instance_df

    def label_encode(self, input_instance):
        for column in self.data_interface.categorical_feature_names:
            input_instance[column] = self.labelencoder[column].transform(input_instance[column])
        return input_instance

    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values
        """
        num_to_decode = 1
        if len(labelled_input.shape) > 1:
            num_to_decode = len(labelled_input)
        else:
            labelled_input = [labelled_input]

        input_instance = []

        for j in range(num_to_decode):
            temp = {}
            for i in range(len(labelled_input[j])):
                if self.data_interface.feature_names[i] in self.data_interface.categorical_feature_names:
                    enc = self.labelencoder[self.data_interface.feature_names[i]]
                    val = enc.inverse_transform(np.array([labelled_input[j][i]], dtype=np.int32))
                    temp[self.data_interface.feature_names[i]] = val[0]
                else:
                    temp[self.data_interface.feature_names[i]] = labelled_input[j][i]
            input_instance.append(temp)
        input_instance_df = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        return input_instance_df

    def label_decode_cfs(self, cfs_arr):
        ret_df = None
        if cfs_arr is None:
            return None
        for cf in cfs_arr:
            df = self.label_decode(cf)
            if ret_df is None:
                ret_df = df
            else:
                ret_df = ret_df.append(df)
        return ret_df

    def get_valid_feature_range(self, normalized=False):
        ret = self.data_interface.get_valid_feature_range(self.feature_range, normalized=normalized)
        for feat_name in self.data_interface.categorical_feature_names:
            ret[feat_name] = self.labelencoder[feat_name].transform(ret[feat_name])
        return ret
