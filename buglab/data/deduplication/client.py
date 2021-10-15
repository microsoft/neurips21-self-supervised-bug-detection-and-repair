from typing import List

import msgpack
import zmq


class DuplicationClient:
    def __init__(self, server_address: str = "tcp://localhost:5555"):
        self.__context = zmq.Context.instance()
        self.__socket = self.__context.socket(zmq.REQ)
        self.__socket.connect(server_address)

    def check_if_duplicate_and_add(self, filename: str, tokens: List[str]) -> bool:
        self.__socket.send(msgpack.dumps((filename, tokens)))
        return bool(self.__socket.recv())
