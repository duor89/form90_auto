from pathlib import Path
import os as os
import pandas as pd
from extract import extract_data, extract_test_data
from utils import ReportingLogger, log_errors
from mapping import ReportMapper
import numpy as np
from form_90_post_mapping import *

current_rpt_period = pd.to_datetime('2024-12-31')

def load_true_aux_oth(load_file):
    schema_dict = {
        'customer_nr': str,
        'customer_name':str,
        'Counterparty': str
    }

    with pd.ExcelFile(load_file) as xls:
        true_aux_oth_df = pd.read_excel(xls, 'Counterparty type FI-AUX-OTH', dtype=schema_dict)

    return true_aux_oth_df

   
def update_and_filter_rt30(main_df, true_aux_oth_df) -> pd.DataFrame:

    if main_df is None or true_aux_oth_df is None:
        return ValueError('Inpur Dataframes are empty')
    
    main_df['customer_nr'] = main_df['customer_nr'].astype(str)
    true_aux_oth_df['customer_nr'] = true_aux_oth_df['customer_nr'].astype(str)

    customer_mask = main_df['customer_nr'].isin(true_aux_oth_df['customer_nr'])

    customer_mask = main_df['customer_nr'].isin(true_aux_oth_df['customer_nr'])

    thisQ_mask = main_df['rv_cpty_type_thisQ'] == 'FI-AUX-OTH'
    
    lastQ_mask = main_df['rv_cpty_type_lastQ'] == 'FI-AUX-OTH'

    main_df['rv_cpty_type_thisQ'] = np.where(
    (thisQ_mask & ~customer_mask),
    'FI-MM-INV-FUND',
    main_df['rv_cpty_type_thisQ']
    )

    main_df['rv_cpty_type_lastQ'] = np.where(
    (lastQ_mask & ~customer_mask),
    'FI-MM-INV-FUND',
    main_df['rv_cpty_type_lastQ']
    )

    return main_df


def main():

    curr_path = Path(__file__).resolve()
        
    output_path = curr_path.parent.parent/ 'output'

    data_path = curr_path.parent.parent/'data'

    load_file = data_path / 'Counterparty_type_FI_AUX_OTH.xlsx'

    aux_oth=load_true_aux_oth(load_file)

    rt30_raw = extract_test_data()

    rt30_df = rt30_raw.get('rt30')

    updated_rt30_df = update_and_filter_rt30(rt30_df, aux_oth)

    rt30_mapper = ReportMapper(curr_path.parent.parent/'config'/ 'RT30.json')

    df_rt30_mapped = rt30_mapper.map_data(updated_rt30_df)

    post_mapper = RT30PostMapper(df_rt30_mapped,current_rpt_period)

    processed_data['deal_type_r4'] = processed_data['dealtype'].astype(str).str.slice(-4)

    EXCLUDE_COLUMNS = ['maturity_date_status', 'value_date_status']
    
    with pd.ExcelWriter(output_path / 'raw_data.xlsx') as writer:
        # Drop excluded columns before writing
        output_data = processed_data.drop(columns=EXCLUDE_COLUMNS, errors='ignore')
        output_data.to_excel(writer, sheet_name='mapped_rt30', index=False)



if __name__ == '__main__':
    main()
