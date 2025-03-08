import os
from pathlib import Path
from typing import Dict
from utils import ReportingLogger, log_errors
import numpy as np
from sqlalchemy import create_engine, text
import pandas as pd


logger = ReportingLogger(__name__, "reporting.log").logger


def connect_to_db():
    """Create database connection"""
    server = 'DESKTOP-R532DCO\SQLEXPRESS'

    try:
        connection_string = (
            f"mssql+pyodbc:///?"
            f"odbc_connect=DRIVER=ODBC+Driver+17+for+SQL+Server;"
            f"SERVER={server};Trusted_Connection=yes;"
        )

        # # username and password
        # connection_string = (
        #     f"mssql+pyodbc:///?"
        #     f"odbc_connect=DRIVER=ODBC+Driver+17+for+SQL+Server;"
        #     f"SERVER={server};DATABASE={database};"
        #     f"UID={username};PWD={password};"
        # )

        return create_engine(connection_string)
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None


def extract_data() -> Dict[str, pd.DataFrame]:
    """Extract all required datasets from SQL"""
    queries_path = Path(__file__).parent.parent / 'queries'
    sql_files = {
        'rt30': '731 RT30RT32 FEE BOCS v2.sql'
    }
    engine = connect_to_db()
    if engine is None:
        raise ConnectionError("Failed to connect to database")
    results = {}
    try:
        with engine.connect() as conn:

            # extract data
            for name, file in sql_files.items():
                logger.info(f"Extracting {name} data")
                with open(queries_path / file, 'r') as f:
                    query = f.read()
                results[name] = pd.read_sql(text(query), conn)

        return results
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise


def extract_test_data():
    path = os.path.join(
        os.path.dirname(__file__),
        r'../data/Form90 BOCS Working NEW 20241231 updated.xlsb')

    na_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A',
                 'n/a', '<NA>', '#NA', 'null', 'NaN', '-NaN', 'nan', '-nan', 'None', '']

    schema_dict = {
        'lot_type_fk': str,
        'ide_linkage_ref': str,
        'ide_linkage_type': str,
        'ide_sourcesys_ref': str,
        'customer_nr': str,
        'ISIN': str,
        'rv_mat_remaining': str
    }


    output = {}

    output['rt30'] = pd.read_excel(
        path,
        sheet_name='RT30',
        dtype=schema_dict,
        keep_default_na=False,
        # parse_dates=['value_date', 'maturity_date'],
        na_values=na_values
    )[['lot_type_fk',
       'ide_linkage_ref',
       'ide_linkage_type',
       'ide_sourcesys_ref',
       'dealtype',
       'customer_nr',
       'customer_name',
       'rv_coa',
       'rv_cpty_type_lastQ',
        'rv_cpty_type_thisQ',
        'rv_rel_party_type_lastQ',
        'rv_rel_party_type_thisQ',
        'rv_resident_lastQ',	
        'rv_resident_thisQ',	
        'nationality_lastQ',	
        'nationality_thisQ',	
        'domicile_lastQ',	
        'domicile_thisQ',	
        'currency',	
        'value_date',	
        'maturity_date',	
        'fx_rate_value',	
        'fx_rate_maturity',	
        'rv_mat_original',	
        'rv_mat_remaining',	
        'nominal',	
        'trade_price',	
        'ISIN',	
        'rca_bookv_thisQ',	
        'rca_ori_bookv_thisQ',	
        'rca_marketv_thisQ',	
        'rca_ori_marketv_thisQ',	
        'rca_accrint_thisQ',	
        'rca_prov_coll_thisQ',
        'rca_prov_indi_thisQ',	
        'fx_rate_thisQ',	
        'rca_bookv_lastQ',	
        'rca_ori_bookv_lastQ',	
        'rca_marketv_lastQ',	
        'rca_ori_marketv_lastQ',	
        'rca_accrint_lastQ',	
        'rca_prov_coll_lastQ',	
        'rca_prov_indi_lastQ',	
        'fx_rate_lastQ']]
    
    output['rt30'].columns = output['rt30']\
        .columns.str.split('.').str[0]

    output['rt30']['maturity_date_status'] = output['rt30']['maturity_date']
    output['rt30']['value_date_status'] = output['rt30']['value_date']

    output['rt30']['maturity_date_status'].replace({"NULL": np.nan, 2958465: 132320}, inplace=True)
    output['rt30']['value_date_status'].replace({"NULL": np.nan, 2958465: 132320}, inplace=True)

    output['rt30']['maturity_date_status'] = pd.to_datetime(output['rt30']['maturity_date_status'],origin=pd.Timestamp('1899-12-30'), unit='D')
    output['rt30']['value_date_status'] = pd.to_datetime(output['rt30']['value_date_status'],origin=pd.Timestamp('1899-12-30'), unit='D')

    return output


def main():
    """Extract data and save to CSV"""
    try:
        # Choose between real or test data
        data = extract_data()
        # data = extract_test_data()

        # Save raw data
        output_path = Path(__file__).parent.parent / 'output'

        with pd.ExcelWriter(output_path / 'raw_data.xlsx') as writer:
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=False)
                logger.info(f"Saved raw {name} data")

    except Exception as e:
        logger.error(f"Error in data extraction: {e}")
        raise


if __name__ == "__main__":
    main()
