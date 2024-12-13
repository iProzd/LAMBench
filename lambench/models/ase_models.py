from lambench.models.basemodel import BaseLargeAtomModel
class ASEModel(BaseLargeAtomModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "ASE":
            raise ValueError(f"Model type {self.model_type} is not supported by ASEModel")

    def evaluate(self, datapath: str, target_name: str):
        pass