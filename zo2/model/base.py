import torch

class BaseZOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zo_training = True

    def zo_train(self):
        """
            Zeroth-order training
        """
        self.zo_training = True
        self.eval()

    def zo_eval(self):
        """
            Zeroth-order evaluation
        """
        self.zo_training = False
        self.eval()
