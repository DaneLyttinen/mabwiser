from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union
import numpy as np
from mabwiser.base_mab import BaseMAB
from mabwiser.thompson import _ThompsonSampling
from mabwiser.utils import Arm, Num, reset, argmax, _BaseRNG
import numpy as np
from scipy.special import expit
from numpy.linalg import inv
import math

class LogisticModel(object):
    """A logistic regression model for fitting and predicting binary response data.
    
    Attributes:
        w: weights
    """
    def __init__(self):
        self.converged = False
        self.w = None
        self.nll_sequence = None

    def fit(self,X,y, iterations=25, tol=.000001):
        """
        Given a response vector (y), training data matrix (X), runs the IRLS algorithm to the specified number of iterations.
        Returns a dictionary containing the coefficients 
        """

        w = np.array([0]*X.shape[1], dtype='float64') if self.w == None else self.w
        y_bar = np.mean(y)
        w_init = math.log(y_bar/(1-y_bar))
        nll_sequence = [] if self.nll_sequence == None else self.nll_sequence
        for i in range(iterations):
            h = X.dot(w)
            p = 1/(1+np.exp(-h))
            p_adj = p
            p_adj[p_adj==1.0] = 0.99999999
            nll = -(1-y.dot(np.log(1-p_adj)))+y.dot(np.log(p_adj))
            nll_sequence += [nll]
            
            if i>1:
                if not self.converged and abs(nll_sequence[-1]-nll_sequence[-2])<tol:
                    self.converged = True
                    self.converged_k = i+1
            
            s = p*(1-p)
            S = np.diag(s)
            arb_small = np.ones_like(s, dtype='float64')*tol
            z = h + np.divide((y-p), s, out=arb_small, where=s!=0)
            Xt = np.transpose(X)
            XtS = Xt.dot(S)
            XtSX = XtS.dot(X)
            inverse_of_XtSX = np.linalg.inv(XtSX)
            inverse_of_XtSX_Xt = inverse_of_XtSX.dot(Xt)
            inverse_of_XtSX_XtS = inverse_of_XtSX_Xt.dot(S)
            w = inverse_of_XtSX_XtS.dot(z)
                                                
        self.nll = nll
        self.nll_sequence = nll_sequence                                                      
        self.w=w
        
        if not self.converged:
            print('Warning: IRLS failed to converge. Try increasing the number of iterations.')
        
        return(self)
        

    def predict(self, X, use_probability = False):
        """
        Given the fitted model and a new sample matrix, X, 
        returns an array (y) of predicted log-odds (or optionally the probabilities).
        """
        
        if not hasattr(self, 'w'):
            print('LogisticModel has not been fit.')
            return(None)
                
        pred = X.dot(self.w)
        
        if use_probability:
            odds = np.exp(pred)
            pred = odds / (1 + odds)
        
        return(pred)

class _CompliantThompsonSampling(BaseMAB):
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str], l2_lambda: Num):
        super().__init__(rng, arms, n_jobs, backend)
        self.l2_lambda = l2_lambda
        self.gram_matrix = {arm: np.eye(len(arms)) for arm in self.arms} 
        self.reward_parameters = {arm: np.zeros(len(arms)) for arm in self.arms}  
        self.compliance_parameters = {arm: np.zeros(len(arms)) for arm in self.arms}
        self.arm_to_compliance_model = dict((arm, LogisticModel()) for arm in arms)

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
            # Calculate the predicted probability
            logits = np.dot(feature_vector, self.compliance_parameters[arm])
            p = expit(logits)

            # Weight for each data point, which is just the variance of the Bernoulli distribution
            weight = p * (1 - p)

            # Calculate 'z', the adjusted response variable
            z = logits + (compliance - p) / weight

            # Update compliance parameters (using only the scalar weight)
            self.compliance_parameters[arm] += (feature_vector * weight) * (z - logits)
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
        # if (compliances == None):
        #     compliances = np.ones(decisions.shape)
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
            compliances = np.ones(contexts.shape)
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
        expectations = self.predict_expectations(contexts, is_predict=True)

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

    def predict_expectations(self, contexts: Optional[np.ndarray] = None, is_predict=False) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        if contexts is None:
            # Create a default context array with the correct shape
            default_context = np.ones((len(self.compliance_parameters[next(iter(self.compliance_parameters))]),))
            contexts = np.array([default_context])
        
        arms = deepcopy(self.arms)
        ##arms = np.array(arms)

        num_contexts = contexts.shape[0]
        arm_expectations = np.empty((num_contexts, len(arms)), dtype=float)

        for i, arm in enumerate(arms):
            # Vectorized computation for each arm
            compliance_prob = expit(contexts @ self.compliance_parameters[arm])
            expected_reward = contexts @ self.reward_parameters[arm]
            arm_expectations[:, i] = compliance_prob * expected_reward

        if is_predict:
            # Return the arm with the highest expectation for each context
            return arms[np.argmax(arm_expectations, axis=1)].tolist()
        else:
            # Return a list of dictionaries mapping each arm to its expectation
            return [dict(zip(arms, arm_expectations[i])) for i in range(num_contexts)]
        
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