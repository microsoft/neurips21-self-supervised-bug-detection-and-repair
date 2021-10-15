import io
import logging
import time
from typing import Tuple

import msgpack
import torch
import zmq
from ptgnn.baseneuralmodel import AbstractNeuralModel
from torch import nn

from .data import ModelSyncData

LOGGER = logging.getLogger(__name__)


class ModelSyncClient:
    """A client asking for the latest model."""

    def __init__(self, address: str, do_not_update_before_sec: int = 0):
        self.__server_address = address

        self.__current_model_version: int = 0
        self.__current_param_version: int = 0

        self.__context = zmq.Context.instance()
        self.__socket = self.__context.socket(zmq.REQ)
        self.__socket.connect(self.__server_address)

        self.__last_model_sync = 0
        self.__do_not_update_before_sec = do_not_update_before_sec

    def ask(self, device) -> Tuple[AbstractNeuralModel, nn.Module]:
        """
        Ask for new model updates.

        This call will block until the server is available and ready to respond.
        This function might be used for initializing client which has to wait for
        the server to create a model along with its metadata.
        """
        self.__current_model_version = 0
        self.__current_param_version = 0

        LOGGER.info("Asking for model from server...")
        data, retry = None, 0
        while data is None:
            try:
                data = self.__ask_server(timeout_sec=60 * 10)  # 10min
            except TimeoutError:
                if retry < 10:
                    retry += 1
                    LOGGER.error(
                        f"Model sync server did not reply. Could be still initializing. Retrying... ({retry}/10)"
                    )
                    continue
                else:
                    raise

        LOGGER.info("Got model info from server...")

        with io.BytesIO(data.model) as sb:
            (model, neural_net) = torch.load(sb, map_location=device)
        with io.BytesIO(data.params) as sb:
            neural_net.load_state_dict(torch.load(sb, map_location=device))

        self.__current_model_version = data.model_version
        self.__current_param_version = data.params_version
        return model, neural_net

    def update_params_if_needed(self, neural_net: nn.Module, device) -> bool:
        if self.__last_model_sync + self.__do_not_update_before_sec > time.time():
            return False  # Skip asking for update

        LOGGER.info("Asking model for parameter updates...")
        try:
            data = self.__ask_server()
        except TimeoutError as e:
            LOGGER.error("Server did not respond. Continuing with the previous parameters.")
            return False

        if data.model is not None:
            raise Exception("Model has been updated. Cannot only update parameters.")
        if data.params is None:
            LOGGER.info("No parameter update from server.")
            return False  # No update
        with io.BytesIO(data.params) as sb:
            neural_net.load_state_dict(torch.load(sb, map_location=device))
        LOGGER.info("Parameters updated.")
        self.__current_param_version = data.params_version
        return True

    def __ask_server(self, timeout_sec: int = 30):
        poller = zmq.Poller()
        poller.register(self.__socket, zmq.POLLIN)
        try:
            self.__socket.send(msgpack.dumps((self.__current_model_version, self.__current_param_version)))
            if poller.poll(timeout_sec * 1000):
                return ModelSyncData(*msgpack.loads(self.__socket.recv()))
            else:
                self.__socket.close()
                self.__socket = self.__context.socket(zmq.REQ)
                self.__socket.connect(self.__server_address)
                raise TimeoutError(f"Error asking for model update. Server did not reply in {timeout_sec} seconds.")

        finally:
            self.__last_model_sync = time.time()


class MockModelSyncClient:
    """A dummy client. Useful for using fixed/pre-trained model."""

    def __init__(self, model_path: str):
        self.__model_path = model_path

    def ask(self, device) -> Tuple[AbstractNeuralModel, nn.Module]:
        return AbstractNeuralModel.restore_model(self.__model_path, device)

    def update_params_if_needed(self, neural_net, device) -> None:
        pass  # No updates
