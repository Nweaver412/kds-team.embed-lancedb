import csv
import logging
import os
import shutil
import zipfile
import lancedb
import csv
import dataclasses
import logging
from typing import List
import json
import os

from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.dao import TableDefinition
from keboola.component.exceptions import UserException

from kbcstorage.tables import Tables
from kbcstorage.client import Client

import pyarrow as pa
import pandas as pd

from configuration import Configuration

from openai import OpenAI
class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None
        self.client = None
        self.out_table_columns = []
        self.input_table_name = None

    def run(self):
        self.init_configuration()
        self.init_client()
        try:
            input_table = self._get_input_table()
            self.input_table_name = os.path.splitext(os.path.basename(input_table.full_path))[0]
            out_table = self._build_out_table(input_table)

            with open(input_table.full_path, 'r', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)
                
                if self._configuration.outputFormat == 'lance':
                    lance_dir = os.path.join(self.tables_out_path, f'embed-lancedb-{self.input_table_name}')
                    os.makedirs(lance_dir, exist_ok=True)
                    db = lancedb.connect(lance_dir)
                    schema = self._get_lance_schema(reader.fieldnames)
                    table = db.create_table("embeddings", schema=schema, mode="overwrite")
                elif self._configuration.outputFormat == 'csv':
                    out_file_name = f'embed-lancedb-{self.input_table_name}.csv'
                    out_table.full_path = os.path.join(self.tables_out_path, out_file_name)
                    with open(out_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
                        writer = csv.DictWriter(output_file, fieldnames=self.out_table_columns)
                        writer.writeheader()
                        
                        data = []
                        row_count = 0
                        for row in reader:
                            row_count += 1
                            text = row[self._configuration.embedColumn]
                            embedding = self.get_embedding(text)
                            if self._configuration.outputFormat == 'csv':
                                row['embedding'] = embedding
                                writer.writerow(row)
                            else:  # Lance
                                lance_row = {**row, 'embedding': embedding}
                                data.append(lance_row)
                            if self._configuration.outputFormat == 'lance' and row_count % 1000 == 0:
                                table.add(data)
                                data = []
                        
                        if self._configuration.outputFormat == 'lance' and data:
                            table.add(data)

                if self._configuration.outputFormat == 'lance':
                    self._finalize_lance_output(lance_dir)

            print(f"Embedding process completed. Total rows processed: {row_count}")
            print(f"Output saved in {self._configuration.outputFormat} format")

            self.write_manifest(out_table)

        except Exception as e:
            raise UserException(f"Error occurred during embedding process: {str(e)}")

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

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
            zip_file_name = f'embed-lancedb-{self.input_table_name}.zip'
            zip_path = os.path.join(self.files_out_path, zip_file_name)
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