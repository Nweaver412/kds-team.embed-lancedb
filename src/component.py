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
            print(f"Input table retrieved: {input_table.full_path}")
            
            output_table = self._get_output_table()
            print(f"Output table created: {output_table.full_path}")
            
            print("Starting embedding process")
            with open(input_table.full_path, 'r', encoding='utf-8') as input_file, \
                 open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
                
                reader = csv.DictReader(input_file)
                fieldnames = reader.fieldnames + ['embedding']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()

                row_count = 0
                for row in reader:
                    row_count += 1
                    if row_count % 100 == 0:
                        print(f"Processing row {row_count}")
                    text = row[self._configuration.embedColumn]
                    embedding = self.get_embedding(text)
                    row['embedding'] = embedding
                    writer.writerow(row)
                    
                    embedding_str = str(embedding)
                    print_embedding = embedding_str[:25] + "..." if len(embedding_str) > 25 else embedding_str
                    print(f"Embedding for row {row_count}: {print_embedding}")

            print(f"Embedding process completed. Total rows processed: {row_count}")
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
            print("Embedding received")
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            raise UserException(f"Error getting embedding: {str(e)}")

    def _get_input_table(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")
        return self.get_input_tables_definitions()[0]

    def _get_output_table(self):
        output_table = self.create_out_table_definition('embeddings.csv')
        return output_table

if __name__ == "__main__":
    try:
        comp = Component()
        comp.execute_action()
    except UserException as exc:
        print(f"UserException occurred: {str(exc)}")
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        print(f"Unexpected exception occurred: {str(exc)}")
        logging.exception(exc)
        exit(2)
