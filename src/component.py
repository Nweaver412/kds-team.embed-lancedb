import csv
import logging
from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException

class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self.table_rows: int = 0

        if logging.getLogger().isEnabledFor(logging.INFO):
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.ERROR)

    def run(self):
        try:
            input_table = self._get_input_table()
            self.table_rows = self.count_rows(input_table.full_path)
            print(f"Number of rows in the input file: {self.table_rows}")
        except Exception as e:
            raise UserException(f"Error occurred while counting rows: {str(e)}")

    def _get_input_table(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")
        return self.get_input_tables_definitions()[0]

    @staticmethod
    def count_rows(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader) - 1
        return row_count

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