import csv
import logging
import os
import shutil
import lancedb
import pyarrow as pa
from keboola.component.base import ComponentBase
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
                
                if self._configuration.outputFormat == 'lance':
                    schema = self._get_lance_schema(reader.fieldnames)
                    lance_db, lance_table = self._prepare_lance_db(schema)
                elif self._configuration.outputFormat == 'csv':
                    output_table = self._get_output_table()
                    output_file = open(output_table.full_path, 'w', encoding='utf-8', newline='')
                    fieldnames = reader.fieldnames + ['embedding']
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                
                data = []
                row_count = 0
                try:
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
                            self._insert_to_lance(lance_table, data)
                            data = []

                    if self._configuration.outputFormat == 'lance' and data:
                        self._insert_to_lance(lance_table, data)
                finally:
                    if self._configuration.outputFormat == 'csv':
                        output_file.close()
                    elif self._configuration.outputFormat == 'lance':
                        self._finalize_lance_output(lance_db)

            print(f"Embedding process completed. Total rows processed: {row_count}")
            print(f"Output saved in {self._configuration.outputFormat} format")
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

    def _get_output_table(self):
        return self.create_out_table_definition('embeddings.csv')

    def _get_lance_schema(self, fieldnames):
        schema = pa.schema([
            (name, pa.string()) for name in fieldnames
        ] + [('embedding', pa.list_(pa.float32()))])
        return schema

    def _prepare_lance_db(self, schema):
        lance_dir = os.path.join(self.tables_out_path, 'lance_db')
        os.makedirs(lance_dir, exist_ok=True)
        db = lancedb.connect(lance_dir)
        table = db.create_table("embeddings", schema=schema, mode="overwrite")
        return db, table

    def _insert_to_lance(self, table, data):
        df = pa.Table.from_pylist(data)
        table.add(df)

    def _finalize_lance_output(self, lance_db):
        lance_db.close()
        
        # zip lance db
        lance_dir = os.path.join(self.tables_out_path, 'lance_db')
        zip_path = os.path.join(self.tables_out_path, 'embeddings.lance.zip')
        shutil.make_archive(zip_path[:-4], 'zip', lance_dir)
        
        # Remove the original Lance directory
        shutil.rmtree(lance_dir)

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