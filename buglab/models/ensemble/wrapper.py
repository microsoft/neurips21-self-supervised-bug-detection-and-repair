import logging
import math
from itertools import tee
from typing import Dict, Iterator, List, Literal, Tuple

import numpy as np
from ptgnn.baseneuralmodel import ModuleWithMetrics

from buglab.models.basemodel import AbstractBugLabModel
from buglab.representations.data import BugLabData

LOGGER = logging.getLogger(__name__)


class EnsembleModuleWrapper(ModuleWithMetrics):
    def __init__(self, nns):
        super().__init__()
        self.__nns = nns

    @property
    def nns(self) -> Iterator[ModuleWithMetrics]:
        yield from self.__nns


class EnsembleWrapper:
    def __init__(self, models: List[AbstractBugLabModel], kind: Literal["avg", "consensus"]):
        assert all(hasattr(m, "predict") for m in models), "One of the models doesn't have a predict function."
        self._models = models
        assert kind in ("avg", "consensus")
        self._kind = kind

    def predict(
        self, data: Iterator[BugLabData], trained_nn: ModuleWithMetrics, device, parallelize: bool
    ) -> Iterator[Tuple[BugLabData, Dict[int, float], List[float]]]:
        for datapoint in data:
            sample_preds = [
                tuple(model.predict([datapoint], nn, device, parallelize))
                for d, model, nn in zip(tee(data, len(self._models)), self._models, trained_nn.nns)
            ]

            assert len(sample_preds) == len(self._models)
            if any(len(s) != 1 for s in sample_preds):
                LOGGER.warning(
                    "One of the ensemble members did not return a prediction: %s", [len(s) for s in sample_preds]
                )
                if all(len(s) == 0 for s in sample_preds):
                    continue

            sample_preds = [s[0] for s in sample_preds if len(s) == 1]
            # Unpack
            datapoints = tuple(s[0] for s in sample_preds)

            member_localization_probs = tuple(s[1] for s in sample_preds)
            member_rewrite_probs = tuple(s[2] for s in sample_preds)

            if not hasattr(self, "_kind") or self._kind == "avg":
                ensemble_localization_probs, ensemble_rewrite_probs = self._avg_ensembling(
                    member_localization_probs, member_rewrite_probs
                )
                yield datapoints[0], ensemble_localization_probs, ensemble_rewrite_probs
            elif self._kind == "consensus":
                # Do all models predict the same location? If not, predict NO_BUG
                member_predicted_locs = tuple(
                    max((l for l in loc_prob), key=lambda l: loc_prob[l]) for loc_prob in member_localization_probs
                )
                if all(member_predicted_locs[0] == m for m in member_predicted_locs[1:]):
                    ensemble_localization_probs, ensemble_rewrite_probs = self._avg_ensembling(
                        member_localization_probs, member_rewrite_probs
                    )
                else:
                    ensemble_localization_probs = {loc: -math.inf for loc in member_localization_probs[0]}
                    ensemble_localization_probs[-1] = 0  # NO_BUG gets all the probability mass
                    ensemble_rewrite_probs = member_rewrite_probs[0]  # Pick one, it doesn't matter...

                yield datapoints[0], ensemble_localization_probs, ensemble_rewrite_probs
            else:
                raise Exception(f"Unrecognized option `{self._kind}`.")

    def _avg_ensembling(self, member_localization_probs, member_rewrite_probs):
        log_weight = -np.log(len(member_rewrite_probs))
        ensemble_localization_probs = {k: v + log_weight for k, v in member_localization_probs[0].items()}
        for member_localization_prob in member_localization_probs[1:]:
            for k, v in member_localization_prob.items():
                ensemble_localization_probs[k] = np.logaddexp(ensemble_localization_probs[k], v + log_weight)
        ensemble_rewrite_probs = [v + log_weight for v in member_rewrite_probs[0]]
        for member_rewrite_prob in member_rewrite_probs[1:]:
            for i, v in enumerate(member_rewrite_prob):
                ensemble_rewrite_probs[i] = np.logaddexp(ensemble_rewrite_probs[i], v + log_weight)
        return ensemble_localization_probs, ensemble_rewrite_probs
