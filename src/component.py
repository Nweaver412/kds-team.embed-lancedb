import csv
import logging
import os
import shutil
import zipfile
import lancedb
import pyarrow as pa
import pandas as pd
import logging

from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.dao import TableDefinition
from keboola.component.exceptions import UserException
from kbcstorage.tables import Tables
from kbcstorage.client import Client

from configuration import Configuration
from openai import OpenAI

KEY_API_TOKEN = '#api_token'
KEY_DESTINATION = 'destination'

class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None
        self.client = None
        self.input_table_name = None

    def run(self):
        self.init_configuration()
        self.init_client()
        try:
            logging.debug(f"Configuration parameters: {self.configuration.parameters}")
            input_table = self._get_input_table()
            self.input_table_name = os.path.splitext(os.path.basename(input_table.full_path))[0]
            logging.debug(f"Input table name: {self.input_table_name}")

            with open(input_table.full_path, 'r', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)

                if self._configuration.outputFormat == 'lance':
                    lance_dir = os.path.join(self.tables_out_path, 'lance_db')
                    os.makedirs(lance_dir, exist_ok=True)
                    db = lancedb.connect(lance_dir)
                    schema = self._get_lance_schema(reader.fieldnames)
                    table = db.create_table("embeddings", schema=schema, mode="overwrite")
                elif self._configuration.outputFormat == 'csv':
                    output_table = self._get_output_table()
                    logging.debug(f"Output table full path: {output_table.full_path}")
                    output_file = open(output_table.full_path, 'w', encoding='utf-8', newline='')
                    fieldnames = reader.fieldnames + ['embedding']
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
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
                if self._configuration.outputFormat == 'csv':
                    output_file.close()
                elif self._configuration.outputFormat == 'lance':
                    self._finalize_lance_output(lance_dir)

            print(f"Embedding process completed. Total rows processed: {row_count}")
            print(f"Output saved in {self._configuration.outputFormat} format")
        except Exception as e:
            raise UserException(f"Error occurred during embedding process: {str(e)}")
        
    def init_configuration(self):
        logging.debug("Initializing configuration")
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)
        logging.debug(f"Loaded configuration: {self._configuration}")
        
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
    
    def _get_output_table(self):
        try:
            destination_config = self.configuration.parameters.get('destination', {})
            out_table_name = destination_config.get("output_table_name")
            if not out_table_name:
                out_table_name = f"embed-lancedb-{self.input_table_name}"
            out_table_name = f"{out_table_name}.csv"
            table_def = self.create_out_table_definition(out_table_name)
            logging.debug(f"Created output table definition: {out_table_name}")
            return table_def
        except Exception as e:
            logging.error(f"Failed to create output table definition: {str(e)}")
            raise UserException(f"Error creating output table: {str(e)}")
    
    def _get_lance_schema(self, fieldnames):
        schema = pa.schema([
            (name, pa.string()) for name in fieldnames
        ] + [('embedding', pa.list_(pa.float32()))])
        return schema
    
    def _get_storage_source(self) -> str:
        storage_config = self.configuration.config_data.get("storage")
        if not storage_config.get("input", {}).get("tables"):
            raise UserException("Input table must be specified.")
        source = storage_config["input"]["tables"][0]["source"]
        return source
    
    def _get_table_columns(self, table_id: str) -> list:
        client = Client(self._get_kbc_root_url(), self._get_storage_token())
        table_detail = client.tables.detail(table_id)
        columns = table_detail.get("columns")
        if not columns:
            raise UserException(f"Cannot fetch list of columns for table {table_id}")
        return columns
    
    def _build_out_table(self) -> TableDefinition:
        destination_config = self.configuration.parameters['destination']

        if not (out_table_name := destination_config.get("output_table_name")):
            out_table_name = f"embed-lancedb-{self.environment_variables.config_row_id}.csv"
        else:
            out_table_name = f"{out_table_name}.csv"
        return self.create_out_table_definition(out_table_name, columns=[])
    
    @sync_action('listColumns')
    def list_columns(self):
        """
        Sync action to fill the UI element for column selection.
        """
        self.init_configuration()
        table_id = self._get_storage_source()
        columns = self._get_table_columns(table_id)
        return [{"value": c, "label": c} for c in columns]

    def _finalize_lance_output(self, lance_dir):
        print("Zipping the Lance directory")
        try:
            destination_config = self.configuration.parameters['destination']
            
            if not (out_table_name := destination_config.get("output_table_name")):
                zip_file_name = f"embed-lancedb-{self.environment_variables.config_row_id}.zip"
            else:
                zip_file_name = f"{out_table_name}.zip"

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
            print(f"Removed original Lance directory: {lance_dir}")

        except Exception as e:
            print(f"Error in _finalize_lance_output: {e}")
            raise
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        comp = Component()
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)