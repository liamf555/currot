import os
import gym
import torch.nn
import numpy as np
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.spl import SelfPacedTeacherV2, SelfPacedWrapper, CurrOT
from deep_sprl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from deep_sprl.teachers.util import Subsampler
from scipy.stats import multivariate_normal


def logsumexp(x):
    xmax = np.max(x)
    return np.log(np.sum(np.exp(x - xmax))) + xmax


class PointMass2DExperiment(AbstractExperiment):
    TARGET_MEANS = np.array([[3., 0.5], [-3., 0.5]])
    TARGET_VARIANCES = np.array([np.diag([1e-4, 1e-4]), np.diag([1e-4, 1e-4])])

    LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5])
    UPPER_CONTEXT_BOUNDS = np.array([4., 8.])

    def target_log_likelihood(self, cs):
        p0 = multivariate_normal.logpdf(cs, self.TARGET_MEANS[0], self.TARGET_VARIANCES[0])
        p1 = multivariate_normal.logpdf(cs, self.TARGET_MEANS[1], self.TARGET_VARIANCES[1])

        pmax = np.maximum(p0, p1)
        # There is another factor of 0.5 since exactly half of the distribution is out of bounds
        return np.log(0.5 * 0.5 * (np.exp(p0 - pmax) + np.exp(p1 - pmax))) + pmax

    def target_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random

        if n % 2 == 0:
            s0 = rng.multivariate_normal(self.TARGET_MEANS[0], self.TARGET_VARIANCES[0], size=n // 2)
            s1 = rng.multivariate_normal(self.TARGET_MEANS[1], self.TARGET_VARIANCES[1], size=n // 2)
        else:
            s0 = rng.multivariate_normal(self.TARGET_MEANS[0], self.TARGET_VARIANCES[0], size=(n - 1) // 2)
            s1 = rng.multivariate_normal(self.TARGET_MEANS[1], self.TARGET_VARIANCES[1], size=(n + 1) // 2)

        return np.concatenate((s0, s1), axis=0)

    INITIAL_MEAN = np.array([0., 4.25])
    INITIAL_VARIANCE = np.diag(np.square([2, 1.875]))

    STD_LOWER_BOUND = np.array([0.2, 0.1875])
    KL_THRESHOLD = 8000.
    KL_EPS = 0.25
    DELTA = 4.0
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 20

    STEPS_PER_ITER = 4096
    DISCOUNT_FACTOR = 0.95
    LAM = 0.99

    AG_P_RAND = {Learner.PPO: 0.1, Learner.SAC: None}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: None}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: None}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        print("Creating environment, evaluation={}".format(evaluation))
        env = gym.make("ContextualPointMass2D-v1")
        if evaluation or self.curriculum.default():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device="cpu",
                                policy_kwargs=dict(net_arch=[128, 128, 128], activation_fn=torch.nn.Tanh)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, batch_size=128),
                    sac=dict(learning_rate=3e-4, buffer_size=10000, learning_starts=500, batch_size=64,
                             train_freq=5, target_entropy="auto"))

    def create_experiment(self):
        print("Creating experiment")
        timesteps = 200 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        print("Creating self-paced teacher")
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_dsampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(100, 2))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS,
                          wait_until_threshold=True)

    def get_env_name(self):
        return "point_mass_2d"

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
        for i in range(0, 100):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_statistics()[1]
