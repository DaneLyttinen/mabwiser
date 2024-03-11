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
        #y_bar = np.mean(y)
        #w_init = math.log(y_bar/(1-y_bar))
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
        self.tau = 1000  # Initial exploration rounds
        self.gram_matrix_theta = {}
        self.gram_matrix_psi = {}
        self.reward_parameters = {}
        self.compliance_parameters = {}
        self.compliance_models = {arm: LogisticModel() for arm in self.arms}

    def _update_reward_model(self, arm, feature_vector, reward):
        # Update reward model using equation (1) from the paper
        self.gram_matrix_theta[arm] += np.outer(feature_vector, feature_vector.T) + self.l2_lambda * np.eye(feature_vector.shape[0])
        gram_matrix_inv_theta = np.linalg.inv(self.gram_matrix_theta[arm])
        self.reward_parameters[arm] = gram_matrix_inv_theta @ (self.reward_parameters[arm] + feature_vector * reward)

    def _update_compliance_model(self, arm, feature_vector, compliance):
        # Ensure the feature vector is in the correct shape (n_samples, n_features)
        # In this case, we are updating with one sample at a time, so reshape is needed
        X = np.array(feature_vector).reshape(1, -1)  # Reshape to 2D array if necessary
        y = np.array([compliance])  # Compliance is a scalar, so wrap it in an array

        # Check if the logistic model for this arm has been initialized
        if arm not in self.compliance_models:
            self.compliance_models[arm] = LogisticModel()

        # Fit the logistic model with the new data
        self.compliance_models[arm].fit(X, y)

        # After fitting, the model's weights are updated, and there's no need to manually update self.compliance_parameters
        # as it's implicitly handled by the LogisticModel instance.
        # However, you should update `self.compliance_parameters[arm]` if you use it elsewhere.
        # This could be the logistic regression weights for the arm after fitting.
        self.compliance_parameters[arm] = self.compliance_models[arm].w

    def _sample_parameters(self):
        # Sample reward parameters from their posterior distributions
        sampled_reward_parameters = {
            arm: np.random.multivariate_normal(self.reward_parameters[arm], inv(self.gram_matrix_theta[arm])) 
            for arm in self.arms
        }

        # "Sample" compliance parameters - in this case, use the estimated parameters directly
        # Optionally, introduce variability if needed, for example, by adding Gaussian noise based on the model's confidence intervals or another method
        sampled_compliance_parameters = {
            arm: self.compliance_models[arm].w  # Directly use logistic regression weights
            for arm in self.arms
        }

        return sampled_reward_parameters, sampled_compliance_parameters

    def _predict_contexts(self, contexts: np.ndarray) -> List:
        # Sample parameters from their posterior distributions
        sampled_reward_parameters, sampled_compliance_parameters = self._sample_parameters()

        # Generate predictions for each context
        predictions = []
        for context in contexts:
            arm_expectations = {arm: expit(context @ sampled_compliance_parameters[arm]) * (context @ sampled_reward_parameters[arm]) for arm in self.arms}
            best_arm = argmax(arm_expectations)
            predictions.append(best_arm)

        return predictions

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None, compliances: np.ndarray = None):
        feature_dim = contexts.shape[1]
        self.gram_matrix_theta[arm] = self.l2_lambda * np.eye(feature_dim)
        self.gram_matrix_psi[arm] = self.l2_lambda * np.eye(feature_dim)
        self.reward_parameters[arm] = np.zeros(feature_dim)
        self.compliance_parameters[arm] = np.ones(feature_dim)
        
        for context, reward in zip(contexts, rewards):
            self._update_reward_model(arm, context, reward)

        if compliances is not None:
            arm_compliances = compliances[decisions == arm]
            for context, compliance in zip(contexts, arm_compliances):
                self._update_compliance_model(arm, context, compliance)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray, compliances: np.ndarray = None):
        if compliances is None:
            compliances = np.ones(decisions.shape)
        # Get feature dimension from the contexts
        feature_dim = contexts.shape[1]

        # Reset Gram matrices and parameters
        for arm in self.arms:
            self.gram_matrix_theta[arm] = self.l2_lambda * np.eye(feature_dim)
            self.gram_matrix_psi[arm] = self.l2_lambda * np.eye(feature_dim)
            self.reward_parameters[arm] = np.zeros(feature_dim)
            self.compliance_parameters[arm] = np.ones(feature_dim)

        # Initial exploration phase
        # for t in range(len(decisions)):
        #     arm = self.rng.choice(self.arms)
        #     decision, reward, context, compliance = decisions[t], rewards[t], contexts[t], compliances[t]
        #     self._update_reward_model(arm, context, reward)
        #     self._update_compliance_model(arm, context, compliance)


        for arm, reward, context, compliance in zip(self.arms, rewards, contexts, compliances):
            self._update_reward_model(arm, context, reward)
            self._update_compliance_model(arm, context, compliance)

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
            # If no contexts are provided, assume a default context (e.g., a vector of ones)
            default_context = np.ones((1, len(self.arms[0].features)))  # Adjust the size according to your features
            contexts = np.array([default_context])

        num_contexts = contexts.shape[0]
        arm_expectations = np.empty((num_contexts, len(self.arms)), dtype=float)

        sampled_reward_parameters, sampled_compliance_parameters = self._sample_parameters()

        for i, arm in enumerate(self.arms):
            # Calculate expected reward for each context and arm
            for j, context in enumerate(contexts):
                # Use logistic regression model to predict compliance probability
                compliance_prob = self.compliance_models[arm].predict(context.reshape(1, -1), use_probability=True)
                
                # Compute expected reward
                expected_reward = context.dot(sampled_reward_parameters[arm])
                
                # Multiply compliance probability with expected reward
                arm_expectations[j, i] = compliance_prob * expected_reward

        if is_predict:
            # If predicting, return the arm with the highest expectation for each context
            best_arms = np.argmax(arm_expectations, axis=1)
            return [self.arms[i] for i in best_arms]
        else:
            # Otherwise, return a list of dictionaries mapping each arm to its expectation for each context
            return [dict(zip(self.arms, arm_expectations[i])) for i in range(num_contexts)]
        
    def _copy_arms(self, cold_arm_to_warm_arm):
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.reward_parameters[cold_arm] = deepcopy(self.reward_parameters[warm_arm])
            self.compliance_parameters[cold_arm] = deepcopy(self.compliance_parameters[warm_arm])

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        # Add to untrained_arms arms
        self.reward_parameters[arm] = None
        self.compliance_parameters[arm] = None

    def _drop_existing_arm(self, arm: Arm):
        self.reward_parameters.pop(arm)
        self.compliance_parameters.pop(arm)