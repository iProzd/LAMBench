import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from enum import Enum


with open("lambench/models/models_config.yml", "r") as file:
    yaml_data = {v["model_name"]: k for k, v in yaml.safe_load(file).items()}

ModelEnum = Enum("ModelEnum", yaml_data)  # dpa2_241126_v2_4_0: "DP_2024Q4"


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_description: str


class LeaderboardModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    model_name: str
    model_metadata: ModelMetadata
    show_direct_task: bool
    show_finetune_task: bool
    show_calculator_task: bool

    @field_validator("model_name")
    def validate_model_name(cls, v):
        if v not in ModelEnum.__members__.keys():
            raise ValueError(
                f"Invalid model name: {v}, not in {ModelEnum.__members__.keys()}"
            )
        return v


class ResultProcessor:
    @staticmethod
    def process_results_for_one_model():
        """
        This function fetch and process the raw results from corresponding tables for one model across required tasks.
        """
        pass


if __name__ == "__main__":
    ResultProcessor.validate_models_to_show()
    ResultProcessor.process_results_for_one_model()
