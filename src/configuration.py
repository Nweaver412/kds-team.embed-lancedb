import dataclasses
import json
from dataclasses import dataclass
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
class OutTableParams:
    table_name: str
    destination: str
    incremental: bool
    primary_key: list

@dataclass
class Configuration(ConfigurationBase):
    embedColumn: str
    pswd_apiKey: str
    model: str
    outputFormat: str
    output_table_name: str

    def __post_init__(self):
        model_mapping = {
            "small_03": "text-embedding-3-small",
            "large_03": "text-embedding-3-large",
            "ada_002": "text-embedding-ada-002"
        }
        self.model = model_mapping.get(self.model, self.model)

    @classmethod
    def load_from_dict(cls, configuration: dict):
        # Extract output_table_name from nested structure if present
        if 'destination' in configuration and 'output_table_name' in configuration['destination']:
            configuration['output_table_name'] = configuration['destination']['output_table_name']
        
        return super().load_from_dict(configuration)