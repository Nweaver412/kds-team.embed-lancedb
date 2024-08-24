import csv
import logging
import os
import shutil
import zipfile
import lancedb

import pyarrow as pa
import pandas as pd

from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.dao import TableOutputMapping, FileOutputMapping
from keboola.component.exceptions import UserException

from configuration import Configuration
from openai import OpenAI

class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None
        self.client = None

    def run(self):
        self.init_configuration()
        self.init_client()
        try:
            input_table = self._get_input_table()
            
            with open(input_table.full_path, 'r', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)
                
                if self._configuration.outputFormat == 'csv':
                    output_tables = self.get_output_tables_definitions()
                    
                    if not output_tables:
                        raise UserException("No output table specified for CSV.")
                    
                    output_table = output_tables[0]
                    output_mapping = TableOutputMapping.create_from_dict(output_table.to_dict())
                    
                    with open(output_mapping.full_path, 'w', encoding='utf-8', newline='') as output_file:
                        fieldnames = reader.fieldnames + ['embedding']
                        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                        writer.writeheader()
                
                        for row in reader:
                            text = row[self._configuration.embedColumn]
                            embedding = self.get_embedding(text)
                            row['embedding'] = embedding
                            writer.writerow(row)
                    
                    print(f"CSV output saved as {output_mapping.name}")
        except Exception as e:
            raise UserException(f"Error occurred during embedding process: {str(e)}")

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration = Configuration.load_from_dict(self.configuration.parameters)

    def init_client(self):
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)

    def get_embedding(self, text):
        try:
            response = self.client.embeddings.create(input=[text], model=self._configuration.model)
            return response.data[0].embedding
        except Exception as e:
            raise UserException(f"Error getting embedding: {str(e)}")

    def _get_input_table(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")
        return self.get_input_tables_definitions()[0]

    def _get_lance_schema(self, fieldnames):
        schema = pa.schema([
            (name, pa.string()) for name in fieldnames
        ] + [('embedding', pa.list_(pa.float32()))])
        return schema

    def _finalize_lance_output(self, lance_dir):
        print("Zipping the Lance directory")
        try:
            zip_path = os.path.join(self.tables_out_path, 'embeddings_lance.zip')
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(lance_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, lance_dir)
                        zipf.write(file_path, arcname)
            
            print(f"Successfully zipped Lance directory to {zip_path}")
            
            # Remove the original Lance directory
            shutil.rmtree(lance_dir)
        except Exception as e:
            print(f"Error zipping Lance directory: {e}")
            raise

    @sync_action('listColumns')
    def list_columns(self):
        """
        Sync action to fill the UI element for column selection.
        """
        self.init_configuration()
        table_id = self._get_storage_source()
        columns = self._get_table_columns(table_id)
        return [{"value": c, "label": c} for c in columns]

if __name__ == "__main__":
    try:
        comp = Component()
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)