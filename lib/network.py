from abc import abstractmethod
import models.facebox_tiny


class FaceInterface(object):
    @abstractmethod
    def __call__(self, cfg):
        pass


class FaceBoxes_tiny(FaceInterface):
    def __call__(self, cfg):
        return models.facebox_tiny.FaceBoxes(phase, cfg.NUM_CLASS)