import torch

class BaseZOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zo_training = True

    def train(self, mode: bool=True):
        """
            First-order training
        """
        self.zo_training = False
        super().train(mode)

    def eval(self):
        """
            First-order evaluation
        """
        self.zo_training = False
        super().eval()

    def zo_train(self):
        """
            Zeroth-order training
        """
        self.zo_training = True

    def zo_eval(self):
        """
            Zeroth-order evaluation
        """
        self.zo_training = False
        super().eval()
