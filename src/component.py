import os
import zipfile
import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException

from configuration import Configuration

MODEL_MAPPING = {
    "small_03": "text-embedding-3-small",
    "large_03": "text-embedding-3-large",
    "ada_002": "text-embedding-ada-002"
}

class EmbeddingComponent(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None
        self.client = None
        self.db = None
        self.table = None
        self.model = None
        self.vector_size = None

    def configure(self):
        self._configuration = Configuration.load_from_dict(self.configuration.parameters)
        
        api_key = self._configuration.pswd_api_key
        if not api_key:
            raise UserException("OpenAI API key is missing from the configuration.")

        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()

        self.model = MODEL_MAPPING[self._configuration.model]
        self.vector_size = self.get_vector_size(self.model)

        os.makedirs("data/out/files", exist_ok=True)
        self.db = lancedb.connect("data/out/files")

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def create_table(self):
        class Words(LanceModel):
            text: str
            vector: Vector(self.vector_size)

        self.table = self.db.create_table("embedded", schema=Words, mode="overwrite")

    def process_data(self):
        input_table = self.get_input_table_definition()
        df = pd.read_csv(input_table.full_path)

        embed_column = self._configuration.embed_column
        if embed_column not in df.columns:
            raise UserException(f"'{embed_column}' column not found in the input CSV file")

        data = []
        for count, entry in enumerate(df[embed_column], 1):
            embedding = self.get_embedding(entry)
            data.append({"text": entry, "vector": embedding})
            print(f"Added: {count}")

        print("Adding to table")
        try:
            self.table.add(data)
            print("Data added successfully")
        except Exception as e:
            raise UserException(f"Error adding data to table: {e}")

    def export_data(self):
        print("Exporting data to CSV")
        try:
            all_data = self.table.to_pandas()
            vector_df = pd.DataFrame(all_data['vector'].tolist(), columns=[f'vector_{i}' for i in range(self.vector_size)])
            export_df = pd.concat([all_data['text'], vector_df], axis=1)
            
            output_table = self.create_out_table_definition('embedded_data_with_vectors.csv')
            export_df.to_csv(output_table.full_path, index=False)
            print("Data exported successfully")
            return output_table.full_path
        except Exception as e:
            raise UserException(f"Error exporting data to CSV: {e}")

    def zip_output_files(self):
        print("Zipping the output files")
        try:
            zip_path = os.path.join(self.get_output_path(), 'embedded_output.zip')
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add the CSV file
                csv_path = self.export_data()
                arcname = os.path.basename(csv_path)
                zipf.write(csv_path, arcname)
                
                # Add the Lance directory
                lance_dir = os.path.join(self.get_output_path(), 'embedded.lance')
                for root, _, files in os.walk(lance_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, lance_dir)
                        zipf.write(file_path, os.path.join('embedded.lance', arcname))
            
            print(f"Successfully zipped output files to {zip_path}")
        except Exception as e:
            raise UserException(f"Error zipping output files: {e}")

    def get_output_path(self):
        return os.path.join(os.environ.get('KBC_DATADIR', ''), 'out', 'files')

    def run(self):
        self.configure()
        self.create_table()
        self.process_data()
        self.zip_output_files()

    def run(self):
        self.configure()
        self.create_table()
        self.process_data()
        self.export_data()
        self.zip_lance_directory()

    @staticmethod
    def get_vector_size(model_name):
        model_sizes = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        return model_sizes.get(model_name, 1536)

    def get_input_table_definition(self):
        tables = self.get_input_tables_definitions()
        if len(tables) != 1:
            raise UserException("Exactly one input table is required.")
        return tables[0]

if __name__ == "__main__":
    try:
        comp = EmbeddingComponent()
        comp.execute_action()
    except UserException as e:
        print(f"User Exception: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(2)