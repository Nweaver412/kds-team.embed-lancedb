import dataclasses
import json
from dataclasses import dataclass, field
from typing import List

import dataconf

class ConfigurationBase:
    @staticmethod
    def _convert_private_value(value: str):
        return value.replace('"#', '"pswd_')

    @staticmethod
    def _convert_private_value_inv(value: str):
        if value and value.startswith("pswd_"):
            return value.replace("pswd_", "#", 1)
        else:
            return value

    @classmethod
    def load_from_dict(cls, configuration: dict):
        json_conf = json.dumps(configuration)
        json_conf = ConfigurationBase._convert_private_value(json_conf)
        return dataconf.loads(json_conf, cls, ignore_unexpected=True)

    @classmethod
    def get_dataclass_required_parameters(cls) -> List[str]:
        return [cls._convert_private_value_inv(f.name)
                for f in dataclasses.fields(cls)
                if f.default == dataclasses.MISSING
                and f.default_factory == dataclasses.MISSING]

@dataclass
class Destination(ConfigurationBase):
    incremental_load: bool
    output_name: str
    primary_keys_array: list[str]

@dataclass
class AdditionalOptions(ConfigurationBase):
    top_p: float
    max_tokens: int
    temperature: float
    presence_penalty: float
    frequency_penalty: float
    timeout: int = 60

@dataclass
class Authentication(ConfigurationBase):
    service: str
    pswd_api_token: str
    api_token: str = ""  # stack parameter
    api_base: str = ""
    deployment_id: str = ""
    api_version: str = ""

@dataclass
class PromptOptions(ConfigurationBase):
    prompt: str

@dataclass
class PromptTemplates(ConfigurationBase):
    prompt_template: str

@dataclass
class Configuration(ConfigurationBase):
    destination: Destination
    additional_options: AdditionalOptions
    authentication: Authentication
    model: str
    embedColumn: str
    outputFormat: str
    debug: bool = False
    max_token_spend: int = 0
    prompt_templates: PromptTemplates = field(default_factory=lambda: PromptTemplates(prompt_template=""))
    prompt_options: PromptOptions = field(default_factory=lambda: PromptOptions(prompt=""))

    def __post_init__(self):
        model_mapping = {
            "small_03": "text-embedding-3-small",
            "large_03": "text-embedding-3-large",
            "ada_002": "text-embedding-ada-002"
        }
        self.model = model_mapping.get(self.model, self.model)

    @property
    def output_name(self):
        return self.destination.output_name