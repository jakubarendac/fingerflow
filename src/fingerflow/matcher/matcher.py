from .verify_net import VerifyNet


class Matcher:
    def __init__(self, precision, verify_net_path):
        self.__verification_module = VerifyNet(precision, verify_net_path)

    def verify(self, anchor, sample):
        return self.__verification_module.verify_fingerprints(anchor, sample)
