import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from enum import Enum


with open("lambench/models/models_config.yml", "r") as file:
    yaml_data = {k: v["model_name"] for k, v in yaml.safe_load(file).items()}

ModelEnum = Enum("ModelEnum", yaml_data)  # DP_2024Q4: "dpa2_241126_v2_4_0"


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
        if v not in [e.value for e in ModelEnum]:
            raise ValueError(
                f"Invalid model name: {v}, not in {[e.value for e in ModelEnum]}"
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
