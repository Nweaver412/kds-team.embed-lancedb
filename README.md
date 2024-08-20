# Embedding Transformation

This component allows you to embed tables using OpenAI embedding algorithms with data provided from your KBC project.

- [TOC]

---

## Configuration

### Parameters:

#### AI Service Provider: OpenAI

- **API Key (`#api_token`):** Obtain your API key from the [OpenAI platform settings](https://platform.openai.com/account/api-keys).

### Other options:
- **Column to Embed(`embed_column`)** 
- **Embedding Model (`model`):** The model that will generate the embeddings. [Learn more](https://platform.openai.com/docs/models/embeddings).
- **Output Format (`output_format`):** Determines if embeddings will be sent to a zipped lance file, or to a Keboola Table
- **Output Table Name (`output_table_name`)**
- **Output File Name (`output_file_name`)**


---

### Component Configuration Example

**Generic configuration**

```json

```

**Row configuration**

```json

```


# Development

If required, change local data folder (the `CUSTOM_FOLDER` placeholder) path to your custom path in
the `docker-compose.yml` file:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volumes:
      - ./:/code
      - ./CUSTOM_FOLDER:/data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone this repository, init the workspace and run the component with following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone git@github.com:keboola/app-transformation-lanceDB-embeddings.git
cd app-transformation-lanceDB-embeddings
docker-compose build
docker-compose run --rm dev
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the test suite and lint check using this command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
docker-compose run --rm test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration
===========

For information about deployment and integration with KBC, please refer to the
[deployment section of developers documentation](https://developers.keboola.com/extend/component/deployment/)
