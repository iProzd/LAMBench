from lambench.models.basemodel import BaseLargeAtomModel

class DPModel(BaseLargeAtomModel):
    def evaluate(self, data, target_name: str):
        pass

    def _dp_test(self):
        pass

    def _change_bias(self):
        pass

    def _finetune(self):
        pass
