from os import PathLike

from chardet import UniversalDetector


def detect_encoding_and_open(filepath: PathLike):
    detector = UniversalDetector()
    with open(filepath, "rb") as rawdata:
        detector.reset()
        for line in rawdata.readlines():
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return open(filepath, encoding=detector.result["encoding"])
