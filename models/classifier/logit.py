from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
import numpy as np
from models.agent import Agent
from .objective_logit import *


class ClassifierAgent(Agent):

    def __init__(self, **config):

        self.X_train = config["X_train"]
        self.prices_train = config["Y_train"]
        self.X_test = config["X_test"]
        self.prices_test = config["Y_test"]
        self.objective = config["objective"]
        self.name = config["model"]
        self.max_iter = config["max_iter"]
        self.model = self._model()
        self.Y_train, self.Y_test = self._objective()
        self._train_model()

    def _objective(self):
        switch = {
            "maxima_buy_sell": maxima_buy_sell,
            "end_buy_sell": end_buy_sell,
            "local_buy_sell": local_buy_sell
        }
        if self.objective not in switch:
            raise NotImplementedError
        f = switch[self.objective]
        return f(self.prices_train), f(self.prices_test)

    def _model(self):
        switch = {
            "AdaBoostClassifier": AdaBoostClassifier(),
            "BaggingClassifier": BaggingClassifier(),
            "ExtraTreesClassifier": ExtraTreesClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=self.max_iter),
            "LogisticRegression": LogisticRegression(max_iter=self.max_iter),
            "RidgeClassifier": RidgeClassifier(max_iter=self.max_iter),
            "SGDClassifier": SGDClassifier(max_iter=self.max_iter)
        }
        if self.name not in switch:
            raise NotImplementedError
        return switch[self.name]

    def print_infos(self):
        print(self.name, "agent")

    def _train_model(self):
        self.model = self.model.fit(self.X_train, self.Y_train)

    def predict(self, state):
        state = np.array([state])
        return self.model.predict(state)[0]

    def load_model(self, **kwargs):
        return super().load_model(**kwargs)

    def save_model(self, **kwargs):
        return super().save_model(**kwargs)
