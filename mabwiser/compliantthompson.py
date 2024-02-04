from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union
import numpy as np
from sklearn.linear_model import SGDClassifier
from mabwiser.base_mab import BaseMAB
from mabwiser.thompson import _ThompsonSampling
from mabwiser.utils import Arm, Num, reset, argmax, _BaseRNG
import numpy as np
from scipy.special import expit
from numpy.linalg import inv

class _CompliantThompsonSampling(BaseMAB):
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str], l2_lambda: Num):
        super().__init__(rng, arms, n_jobs, backend)
        self.l2_lambda = l2_lambda
        self.compliance_model = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01)
        self.gram_matrix = {arm: np.eye(len(arms)) for arm in self.arms} 
        self.reward_parameters = {arm: np.zeros(len(arms)) for arm in self.arms}  
        self.compliance_parameters = {arm: np.zeros(len(arms)) for arm in self.arms}

    def _update_reward_model(self, arm, feature_vector, reward):
        self.gram_matrix[arm] += self.l2_lambda * np.eye(len(feature_vector))
        gram_matrix_inv = inv(self.gram_matrix[arm])

        # Sherman-Morrison formula for efficient update of the inverse Gram matrix
        u = gram_matrix_inv @ feature_vector
        v = feature_vector @ gram_matrix_inv
        self.gram_matrix[arm] = gram_matrix_inv - np.outer(u, v) / (1 + feature_vector @ gram_matrix_inv @ feature_vector)

        self.reward_parameters[arm] += self.gram_matrix[arm] @ feature_vector * (reward - feature_vector @ self.reward_parameters[arm])
    
    def _update_compliance_model(self, arm, feature_vector, compliance):
        max_iter = 20
        tolerance = 1e-6
        for _ in range(max_iter):
            p = expit(feature_vector @ self.compliance_parameters[arm])
            W = np.diag(p * (1 - p))
            z = feature_vector @ self.compliance_parameters[arm] + np.linalg.inv(W) @ (compliance - p)
            
            # Update the compliance parameters using weighted least squares
            self.compliance_parameters[arm], _, _, _ = np.linalg.lstsq(W @ feature_vector, W @ z, rcond=None)

            # Check for convergence
            if np.all(np.abs(self.compliance_parameters[arm] - self.compliance_parameters[arm]) < tolerance):
                break

    def _predict_contexts(self, contexts: np.ndarray) -> List:
        # Generate predictions for each context
        predictions = []
        for context in contexts:
            arm_expectations = {arm: expit(context @ self.compliance_parameters[arm]) * (context @ self.reward_parameters[arm]) for arm in self.arms}
            best_arm = argmax(arm_expectations)
            predictions.append(best_arm)
        
        return predictions


    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None, compliances: np.ndarray = None):
        if (compliances == None):
            compliances = np.ones(decisions.shape)
        for context, reward in zip(contexts, rewards):
            self.gram_matrix[arm] += self.l2_lambda * np.eye(len(context))
            gram_matrix_inv = inv(self.gram_matrix[arm])

            u = gram_matrix_inv @ context
            v = context @ gram_matrix_inv
            self.gram_matrix[arm] = gram_matrix_inv - np.outer(u, v) / (1 + context @ gram_matrix_inv @ context)

            self.reward_parameters[arm] += self.gram_matrix[arm] @ context * (reward - context @ self.reward_parameters[arm])

        # Update compliance model for the arm
        if compliances is not None:
            arm_compliances = compliances[decisions == arm]
            for context, compliance in zip(contexts, arm_compliances):
                self._update_compliance_model(arm, context, compliance)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray,contexts: np.ndarray,compliances: np.ndarray = None):
        if (compliances == None):
            compliances = np.ones(decisions.shape)
        # Reset Gram matrix and reward parameters
        for arm in self.arms:
            self.gram_matrix[arm] = np.eye(contexts.shape[1])
            self.reward_parameters[arm] = np.zeros(contexts.shape[1])
            self.compliance_parameters[arm] = np.zeros(contexts.shape[1])

        for decision, reward, feature_vector in zip(decisions, rewards, contexts):
            self._update_reward_model(decision, feature_vector, reward)

        for decision, compliance, feature_vector in zip(decisions, compliances, contexts):
            self._update_compliance_model(decision, feature_vector, compliance)
        
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[Arm, List[Arm]]:
        expectations = self.predict_expectations(contexts)

        # Choose arm with the highest expectation
        if isinstance(expectations, dict):
            return argmax(expectations)
        else:
            return [argmax(exp) for exp in expectations]
        
    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        self._warm_start(arm_to_features, distance_quantile)

    def partial_fit(self, decisions, rewards, contexts, compliances):
        for decision, reward, feature_vector in zip(decisions, rewards, contexts):
            self._update_reward_model(decision, feature_vector, reward)

        for decision, compliance, feature_vector in zip(decisions, compliances, contexts):
            self._update_compliance_model(decision, feature_vector, compliance)
        self._parallel_fit(decisions, rewards, contexts)

    def predict_expectations(self, contexts: Optional[np.ndarray] = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        if contexts is None:
            contexts = np.array([[1] * len(self.compliance_parameters[next(iter(self.compliance_parameters))])])

        expectations = []
        for context in contexts:
            arm_expectations = {}
            for arm in self.arms:
                compliance_prob = expit(context @ self.compliance_parameters[arm])
                expected_reward = context @ self.reward_parameters[arm]
                arm_expectations[arm] = compliance_prob * expected_reward
            expectations.append(arm_expectations)

        if len(contexts) == 1:
            return expectations[0]
        else:
            return expectations
        
    def _copy_arms(self, cold_arm_to_warm_arm):
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.reward_parameters[cold_arm] = deepcopy(self.reward_parameters[warm_arm])
            self.compliance_parameters[cold_arm] = deepcopy(self.compliance_parameters[warm_arm])

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        if self.arms:
            num_features = len(self.reward_parameters[next(iter(self.arms))])
        else:
            raise ValueError("No existing arms to infer the number of features.")

        # Initialize the necessary parameters for a new arm
        self.reward_parameters[arm] = np.zeros(num_features)
        self.compliance_parameters[arm] = np.zeros(num_features)

    def _drop_existing_arm(self, arm: Arm):
        self.reward_parameters.pop(arm)
        self.compliance_parameters.pop(arm)