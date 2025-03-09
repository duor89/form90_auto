import json
import sys
import os
from typing import Dict, Union, Optional
import logging
from functools import lru_cache

import pandas as pd
from utils import ReportingLogger, log_errors


class ReportMapper:
    # Class variables
    _mapping_cache = {}

    def __init__(self, config_file: str, reuse_mappings: bool = True):
        self.logger = ReportingLogger(__name__, "reporting.log").logger
        self.logger.info(
            f"Initializing ReportMapper with config: {config_file}")

        # Load and validate configuration
        self.config = self._load_config(config_file)

        # Initialize mapping data
        self._initialize_mappings(reuse_mappings)

    @log_errors
    def _load_config(self, config_file: str) -> dict:
        """Loads and returns JSON configuration"""
        with open(config_file, 'rb') as f:
            config = json.load(f)
        self.logger.info("Configuration loaded successfully")
        return config

    @log_errors
    def _initialize_mappings(self, reuse_mappings: bool) -> None:
        """Initializes mapping data from Excel file"""
        self.mapping_file = os.path.abspath(self.config['mapping_file'])

        # Check if mappings are already cached
        if reuse_mappings and self.mapping_file in self._mapping_cache:
            self.logger.info(f"Using cached mappings for: {self.mapping_file}")
            self.all_mappings = self._mapping_cache[self.mapping_file]
        else:
            self.logger.info(f"Loading mappings from: {self.mapping_file}")
            self.all_mappings = pd.read_excel(
                self.mapping_file,
                sheet_name=None,
                dtype=str,
                keep_default_na=False
            )
            self.all_mappings = {
                k: v.rename(columns=lambda x: str(x))
                for k, v in self.all_mappings.items()
            }

            # Cache the mappings
            if reuse_mappings:
                self._mapping_cache[self.mapping_file] = self.all_mappings

        self.logger.info("Mappings loaded successfully")

    @classmethod
    def clear_cache(self):
        """Clears the mapping cache"""
        self._mapping_cache.clear()

    def apply_mapping(self,
                      data: pd.DataFrame,
                      mapping: pd.DataFrame,
                      keys: Dict[str, Union[str, list]],
                      values: Dict[str, str],
                      sheet_name: str) -> pd.DataFrame:
        """
        Apply a mapping to the main DataFrame.

        Args:
            data: The primary data to map
            mapping: The mapping data from the Excel sheet
            keys: Dictionary containing:
                - main: Column(s) from main data
                - mapping: Column(s) from mapping data
            values: Dictionary to rename output columns
            sheet_name: Name of the mapping sheet being used

        Returns:
            DataFrame with mappings applied
        """
        mapping = mapping.rename(columns=values)

        # if mapping key is str, then convert to list of single str to merge value columns
        mapping_keys = [keys['mapping']] if isinstance(
            keys['mapping'], str) else keys['mapping']
        value_cols = list(values.values())
        cols = list(set(mapping_keys + value_cols))

        # Log the mapping operation details
        self.logger.info(
            f"Applying mapping using keys: {keys['main']} -> {keys['mapping']}")

        data = data.apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        mapping = mapping.apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        # Check for duplicates in the mapping DataFrame
        duplicates = mapping[mapping.duplicated(
            subset=mapping_keys, keep=False)]
        if not duplicates.empty:
            self.logger.info(
                f"Duplicate entries found in mapping for '{mapping_keys}':\n{len(duplicates)}")
            mapping = mapping.drop_duplicates(subset=mapping_keys)
            self.logger.info(
                f"Dropped duplicates from mapping for '{mapping_keys}'")

        df_merge = pd.merge(
            data, mapping[cols], left_on=keys['main'], right_on=keys['mapping'], how='left', suffixes=('', '_ref'))

        # Check for NaN values in mapping keys
        total_nan_values = df_merge[mapping_keys].isna().sum().sum()
        if total_nan_values > 0:
            self.logger.warning(
                f"Sheet '{sheet_name}': {total_nan_values} NaN values found in mapping key '{mapping_keys}'. The mapping sheet may be out of date.")

        return df_merge[data.columns.to_list() + list(values.values())]

    def import_data(self, sheet_name: Union[int, str] = 0) -> Optional[pd.DataFrame]:
        """Import data based on configuration"""
        data_source = self.config.get('data_source', {})

        # Add support for default options
        default_options = {
            'excel': {'engine': 'openpyxl'},
            'csv': {'encoding': 'utf-8'}
        }

        source_type = data_source.get('type', '').lower()
        options = {**default_options.get(source_type, {}),
                   **data_source.get('options', {})}

        if source_type == 'excel':
            return pd.read_excel(data_source['path'],
                                 sheet_name=sheet_name,
                                 dtype=data_source.get('dtype'),
                                 **options)
        elif source_type == 'csv':
            return pd.read_csv(data_source['path'], dtype=data_source.get('dtype', None))
        elif source_type == 'sql':
            print("SQL data source selected. Load the dataframe programmatically and pass it to the `map_data` method.")
            return None
        else:
            raise ValueError(
                f"Unsupported data source type: {data_source['type']}")

    @log_errors
    def map_data(self, main_data: pd.DataFrame) -> pd.DataFrame:
        """Apply all mappings with logging"""
        self.logger.info("Starting mapping process")
        data = main_data.copy()

        total_mappings = len(self.config['mappings'])
        for idx, mapping in enumerate(self.config['mappings'], 1):
            self.logger.info(
                f"Applying mapping {idx}/{total_mappings}: {mapping['sheet_name']}")

            sheet_name = mapping['sheet_name']
            if sheet_name not in self.all_mappings:
                self.logger.error(
                    f"Sheet '{sheet_name}' not found in {self.mapping_file}")
                raise ValueError(
                    f"Sheet '{sheet_name}' not found in {self.mapping_file}")

            df_mapping = self.all_mappings[sheet_name]
            data = self.apply_mapping(
                data, df_mapping, mapping['keys'], mapping['values'], sheet_name)

        self.logger.info("Mapping process completed successfully")
        return data

    def add_mapping(self, sheet_name: str, data: pd.DataFrame) -> None:
        """
        Adds a new mapping sheet to all_mappings without modifying existing ones

        Args:
            sheet_name: Name of the new sheet to add
            data: DataFrame containing the mapping data
        """
        # Add to instance's all_mappings
        self.all_mappings[sheet_name] = data

        # Add to cache if it exists
        if self.mapping_file in self._mapping_cache:
            self._mapping_cache[self.mapping_file][sheet_name] = data

        self.logger.info(f"Added new mapping sheet: {sheet_name}")

    @log_errors
    def reload_mappings(self) -> None:
        """
        Reloads mapping data from Excel file, forcing a refresh of the cache
        """
        self.logger.info(f"Reloading mappings from: {self.mapping_file}")

        # Clear the specific mapping from cache
        if self.mapping_file in self._mapping_cache:
            del self._mapping_cache[self.mapping_file]

        # Reinitialize mappings with reuse_mappings=True to update cache
        self._initialize_mappings(reuse_mappings=True)

        self.logger.info("Mappings reloaded successfully")


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) != 2:
        logger.error("Usage: python mapping.py <path_to_json>")
        sys.exit(1)

    json_file = sys.argv[1]
    try:
        mapper = ReportMapper(json_file)
        df_input = mapper.import_data()
        df_output = mapper.map_data(df_input)
        output_file = mapper.config['output']['file']
        df_output.to_excel(output_file, index=False)
        logger.info(f"Report successfully generated: {output_file}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
