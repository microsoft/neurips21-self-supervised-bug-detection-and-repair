import io
import logging
import time
from threading import Lock, Thread

import msgpack
import torch
import zmq
from ptgnn.baseneuralmodel import AbstractNeuralModel
from torch import nn

from .data import ModelSyncData

LOGGER = logging.getLogger(__name__)


class ModelSyncServer:
    """A server storing the latest model and parameters."""

    def __init__(self, address: str, model: AbstractNeuralModel, neural_net: nn.Module):
        self.__server_address = address

        self.__sync_lock = Lock()

        self.__current_model_version = time.time()
        self.__serialized_model = self.__serialize_model(model, neural_net)
        self.__update_nn_params(neural_net)

        self.__thread = Thread(target=self.serve, daemon=True, name="model_sync_server")
        self.__thread.start()

    def __serialize_model(self, model, neural_net) -> bytes:
        with io.BytesIO() as sb:
            torch.save((model, neural_net), f=sb)
            sb.seek(0)
            return sb.read()

    def update_parameters(self, neural_net: nn.Module) -> None:
        with self.__sync_lock:
            self.__update_nn_params(neural_net)

    def __update_nn_params(self, neural_net: nn.Module):
        with io.BytesIO() as sb:
            torch.save(neural_net.state_dict(), f=sb)
            sb.seek(0)
            self.__serialized_params = sb.read()
        self.__current_param_version = time.time()

    def serve(self):
        """The thread responding to updates."""
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(self.__server_address)

        while True:
            r_bytes = socket.recv()
            client_model_version, client_param_version = msgpack.loads(r_bytes)
            with self.__sync_lock:
                if client_model_version < self.__current_model_version:
                    model_update = self.__serialized_model
                else:
                    model_update = None
                if (
                    client_model_version < self.__current_model_version
                    or client_param_version < self.__current_param_version
                ):
                    param_update = self.__serialized_params
                else:
                    param_update = None

            returned_data = ModelSyncData(
                model_update, self.__current_model_version, param_update, self.__current_param_version
            )
            socket.send(msgpack.dumps(returned_data))
            LOGGER.info("Responded to model update request.")
