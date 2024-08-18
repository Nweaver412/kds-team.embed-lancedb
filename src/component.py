import csv
import logging
from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException
from configuration import Configuration
from openai import OpenAI

class Component(ComponentBase):
    def __init__(self):
        print("Initializing Component")
        super().__init__()
        self._configuration = None
        self.client = None
        print("Component initialized")

    def run(self):
        print("Starting run method")
        self.init_configuration()
        self.init_client()
        try:
            print("Getting input table")
            input_table = self._get_input_table()
            print(f"Input table retrieved: {input_table.full_path}")
            
            print("Getting output table")
            output_table = self._get_output_table()
            print(f"Output table created: {output_table.full_path}")
            
            print("Starting embedding process")
            with open(input_table.full_path, 'r', encoding='utf-8') as input_file, \
                 open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
                
                reader = csv.DictReader(input_file)
                fieldnames = reader.fieldnames + ['embedding']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                print("CSV files opened and headers written")

                row_count = 0
                for row in reader:
                    row_count += 1
                    if row_count % 100 == 0:
                        print(f"Processing row {row_count}")
                    text = row[self._configuration.embedColumn]
                    embedding = self.get_embedding(text)
                    row['embedding'] = embedding
                    writer.writerow(row)
                    
                    # Print the embedding (first 25 characters if longer)
                    embedding_str = str(embedding)
                    print_embedding = embedding_str[:25] + "..." if len(embedding_str) > 25 else embedding_str
                    print(f"Embedding for row {row_count}: {print_embedding}")

            print(f"Embedding process completed. Total rows processed: {row_count}")
            print(f"Output saved to {output_table.full_path}")
        except Exception as e:
            print(f"Error occurred during embedding process: {str(e)}")
            raise UserException(f"Error occurred during embedding process: {str(e)}")

    def init_configuration(self):
        print("Initializing configuration")
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)
        print(f"Configuration initialized. Using model: {self._configuration.model}")

    def init_client(self):
        print("Initializing OpenAI client")
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)
        print("OpenAI client initialized")

    def get_embedding(self, text):
        print("Getting embedding for text")
        try:
            response = self.client.embeddings.create(input=[text], model=self._configuration.model)
            print("Embedding received")
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            raise UserException(f"Error getting embedding: {str(e)}")

    def _get_input_table(self):
        print("Getting input table definition")
        if not self.get_input_tables_definitions():
            print("No input tables found")
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        if len(self.get_input_tables_definitions()) > 1:
            print("Multiple input tables found")
            raise UserException("Only one input table is supported")
        print("Input table definition retrieved")
        return self.get_input_tables_definitions()[0]

    def _get_output_table(self):
        print("Creating output table definition")
        output_table = self.create_out_table_definition('embeddings.csv')
        print("Output table definition created")
        return output_table

if __name__ == "__main__":
    print("Script started")
    try:
        print("Creating Component instance")
        comp = Component()
        print("Executing Component action")
        comp.execute_action()
        print("Component action executed successfully")
    except UserException as exc:
        print(f"UserException occurred: {str(exc)}")
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        print(f"Unexpected exception occurred: {str(exc)}")
        logging.exception(exc)
        exit(2)
    print("Script completed")