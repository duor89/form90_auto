import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class MappingConfig:
    """Configuration constants for mapping operations"""
    NULL_VALUES = ['NULL', 'NaT', '']
    MATURITY_DATES = ['12/31/9999', '9999-12-31', '2958465'] 
    SHORT_TERM_THRESHOLD = 1012000
    SPECIAL_COA = [6, 17]
    STATUS_TYPES = ['N', 'M', 'D']
    FORM90_SECTION = 'Part D - Liabilities- Debt securities held by non-residents'

class RT30PostMapper:
    """Handles post-mapping logic for RT30 and RT32 reports"""

    def __init__(self, data: pd.DataFrame, current_reporting_period,
                 mapper: Optional[object] = None):
        self._validate_input(data)
        self.data = data.copy()
        self.mapper = mapper
        self.config = MappingConfig()
        self.current_reporting_period = pd.to_datetime(current_reporting_period)

    def _validate_input(self, data: pd.DataFrame) -> None:
        required_columns = {
            'rv_resident_thisQ', 'rv_resident_lastQ',
            'nationality_thisQ', 'nationality_lastQ',
            'domicile_thisQ', 'domicile_lastQ',
            'maturity_date', 'rv_mat_original', 'nominal',
            'trade_price', 'fx_rate_value', 'fx_rate_maturity', 'fx_rate_thisQ',
            'rca_bookv_thisQ', 'rca_bookv_lastQ',
            'rca_ori_bookv_thisQ', 'rca_ori_bookv_lastQ',
            'rca_marketv_lastQ', 'ide_linkage_type',
            'Form90 COA_inter_var', 'Form90 section_inter_var',
            'Residual maturity_inter_var'
        }
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def apply_status(self) -> 'RT30PostMapper': #checked
        try:
            maturity_dates = pd.to_datetime(self.data['maturity_date_status'], errors='coerce')
            
            maturity_mask = (maturity_dates <= self.current_reporting_period) & (self.data['rca_bookv_thisQ'] == 0)
            default_mask = (maturity_dates > self.current_reporting_period) & (self.data['rca_bookv_thisQ'] == 0)
            new_mask = self.data['rca_bookv_lastQ'] == 0

            conditions = [maturity_mask, default_mask, new_mask]
            choices = ['M', 'D', 'N']
            
            self.data['Status'] = np.select(conditions, choices, default='O')
            return self
            
        except Exception as e:
            logger.error(f"Error in status calculation: {str(e)}")
            raise

    def apply_form90_coa(self) -> 'RT30PostMapper':  #checked
        try:
            linkage_mask = self.data['ide_linkage_type'] == '10'
            
            self.data['Form90_COA'] = np.where(
                linkage_mask,
                '6',
                self.data['Form90 COA_inter_var']
            )
            
            self.data['Form90 section'] = np.where(
                linkage_mask,
                self.config.FORM90_SECTION,
                self.data['Form90 section_inter_var']
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Error in Form90 COA processing: {str(e)}")
            raise

    def apply_domicile_rules(self) -> 'RT30PostMapper': #checked
        try:
            for period in ['thisQ', 'lastQ']:
                conditions = [
                    (self.data[f'rv_resident_{period}'] == 'NO') & 
                    (self.data[f'nationality_{period}'] == 'AU'),
                    
                    (self.data[f'rv_resident_{period}'] == 'NO') & 
                    self.data[f'domicile_{period}'].isin(['YY', 'XX'])
                ]
                choices = ['CN', self.data[f'nationality_{period}']]
                self.data[f'Domicile_{period}'] = np.select(
                    conditions,
                    choices,
                    default=self.data[f'domicile_{period}']
                )
            return self
        except Exception as e:
            logger.error(f"Error in domicile rules: {str(e)}")
            raise

    def apply_company_types(self) -> 'RT30PostMapper': #checked
        try:
            for period in ['thisQ', 'lastQ']:
                conditions = [
                    self.data[f'rv_resident_{period}'].isin(self.config.NULL_VALUES) | ##    NULL_VALUES = ['NULL', 'NaT', '']
                    self.data[f'Domicile_{period}'].isin(self.config.NULL_VALUES),
                    (self.data[f'rv_resident_{period}'] == 'YES') | 
                    (self.data[f'Domicile_{period}'] == 'XX')
                ]
                choices = ['', 'NA']

                self.data[f'Intercompany_type_{period}'] = np.select(
                    conditions,
                    choices,
                    default=self.data[f'Intercompany_type_{period}_inter_var']
                    )
            return self
        except Exception as e:
            logger.error(f"Error in company types: {str(e)}")
            raise

    def apply_counterparty_types(self) -> 'RT30PostMapper':  #checked
        try:
            for period in ['thisQ', 'lastQ']:
                conditions = [
                    self.data[f'rv_resident_{period}'].isin(self.config.NULL_VALUES) | ##    NULL_VALUES = ['NULL', 'NaT', '']
                    self.data[f'Domicile_{period}'].isin(self.config.NULL_VALUES),
                    (self.data[f'rv_resident_{period}'] == 'YES') | 
                    (self.data[f'Domicile_{period}'] == 'XX')
                ]
                choices = ['', 'NA']
  
                self.data[f'Counterparty_type_{period}'] = np.select(
                    conditions,
                    choices,
                    default=self.data[f'Counterparty type_{period}_inter_var']
                    )
      

            return self
        except Exception as e:
            logger.error(f"Error in Counterparty types: {str(e)}")
            raise




    def apply_customer_diff(self) -> 'RT30PostMapper': #checked 4 difference
        try:
            empty_mask = (
                self.data[['Counterparty_type_lastQ', 'Counterparty_type_thisQ',
                          'Intercompany_type_lastQ', 'Intercompany_type_thisQ']] == ''
            ).any(axis=1)
            
            same_values_mask = (
                (self.data['Counterparty_type_lastQ'] == self.data['Counterparty_type_thisQ']) &
                (self.data['Intercompany_type_lastQ'] == self.data['Intercompany_type_thisQ']) &
                (self.data['Domicile_lastQ'] == self.data['Domicile_thisQ'])
            )
            
            self.data['Customer_diff'] = np.where(empty_mask | same_values_mask, 'NO', 'YES')
            return self
        except Exception as e:
            logger.error(f"Error in customer diff: {str(e)}")
            raise

    def apply_maturities(self) -> 'RT30PostMapper':   #checked
        try:
            maturity_str = self.data['maturity_date'].astype(str)
            maturity_mask = (
                maturity_str.isin(self.config.MATURITY_DATES) |
                pd.isna(self.data['maturity_date'])
            )
            
            rv_mat = pd.to_numeric(
                self.data['rv_mat_original'].replace('NULL', '0'),
                errors='coerce'
            )

            rv_mat_str = self.data['rv_mat_original'].astype(str)
            rv_mask = (
                rv_mat_str.isin(self.config.NULL_VALUES) | 
                pd.isna(self.data['rv_mat_original'])
            )

            rv_mat_rm_str = self.data['rv_mat_remaining'].astype(str)
            rv_mat_rm_str_mask = (
                rv_mat_rm_str.isin(self.config.NULL_VALUES) | 
                pd.isna(self.data['rv_mat_remaining'])
            )
            
            self.data['Original_maturity'] = np.where(
                maturity_mask | rv_mask|(rv_mat <= self.config.SHORT_TERM_THRESHOLD),
                'Short-term',
                'Long-term'
            )
            
            self.data['Residual_maturity'] = np.where(
                maturity_mask| rv_mat_rm_str_mask,
                'Term<= 90 days',
                self.data['Residual maturity_inter_var']
            )
            return self
        except Exception as e:
            logger.error(f"Error in maturities: {str(e)}")
            raise

    def apply_nominal_conversion(self) -> 'RT30PostMapper': #checked
        try:
            self.data['nominal_converted'] = pd.to_numeric(
                self.data['nominal'].replace(['NULL', np.nan], '0'),
                errors='coerce'
            ).fillna(0)
            return self
        except Exception as e:
            logger.error(f"Error in nominal conversion: {str(e)}")
            raise

    def calculate_transactions(self) -> 'RT30PostMapper':
        try:
            customer_diff_mask = self.data['Customer_diff'] == 'YES'
            coa_dot_mask = self.data['Form90_COA'].astype(str).str.contains('\.', na=False)
            coa_special_mask = self.data['Form90_COA'].isin(self.config.SPECIAL_COA)
            status_masks = {
                status: self.data['Status'] == status 
                for status in self.config.STATUS_TYPES
            }
            bookv_positive_mask = self.data['rca_bookv_thisQ'] > 0

            trade_calc = (
                self.data['nominal_converted'] * 
                self.data['trade_price'] / 100 * 
                self.data['fx_rate_value']
            )
            ori_diff = (
                self.data['rca_ori_bookv_thisQ'] - 
                self.data['rca_ori_bookv_lastQ']
            )
            normal_calc = (
                self.data['nominal_converted'] * 
                self.data['fx_rate_value'] + 
                (self.data['rca_ori_bookv_thisQ'] - 
                 self.data['nominal_converted']) * 
                self.data['fx_rate_thisQ']
            )

            conditions = [
                customer_diff_mask,
                coa_dot_mask,
                coa_special_mask & status_masks['N'] & bookv_positive_mask,
                coa_special_mask & status_masks['N'] & ~bookv_positive_mask,
                coa_special_mask & status_masks['M'] ,
                coa_special_mask & status_masks['D'] ,
                status_masks['N'],
                status_masks['M'],
                status_masks['D']
            ]

            choices = [
                0,
                self.data['rca_bookv_thisQ'] - self.data['rca_bookv_lastQ'],
                trade_calc,
                -trade_calc,
                -self.data['nominal_converted'] * self.data['fx_rate_maturity'],
                self.data['rca_marketv_lastQ'],
                normal_calc,
                -self.data['rca_ori_bookv_lastQ'] * self.data['fx_rate_maturity'],
                -self.data['rca_bookv_lastQ']
            ]

            self.data['Transactions'] = np.select(
                conditions,
                choices,
                default=ori_diff * self.data['fx_rate_thisQ']
            )
            return self

        except Exception as e:
            logger.error(f"Error in transactions calculation: {str(e)}")
            raise
    
    def calculate_market_price_changes(self) -> 'RT30PostMapper':
        """
        Calculate Market Price Changes using vectorized operations.
        Formula:
        IF(Customer_diff="YES", 0,
        IF(OR(Form90_COA=6, Form90_COA=17),
            IF(AND(Status="N", marketv_thisQ>0),
                rca_marketv_thisQ - nominal_converted*trade_price/100*fx_rate_thisQ,
                IF(AND(Status="N", rca_marketv_thisQ<0),
                    rca_marketv_thisQ + nominal_converted*trade_price/100*fx_rate_thisQ,
                    IF(Status="M",
                    nominal_converted*fx_rate_maturity - rca_ori_marketv_lastQ*fx_rate_maturity,
                    IF(Status="O",
                        (rca_ori_marketv_thisQ - rca_ori_marketv_lastQ)*fx_rate_thisQ,
                        0)))),
            0))
        """
        try:
            # Create masks for conditions
            customer_diff_mask = self.data['Customer_diff'] == 'YES'
            coa_special_mask = self.data['Form90_COA'].isin(self.config.SPECIAL_COA)
            status_n_mask = self.data['Status'] == 'N'
            status_m_mask = self.data['Status'] == 'M'
            status_o_mask = self.data['Status'] == 'O'
            marketv_positive_mask = self.data['rca_marketv_thisQ'] > 0
            marketv_negative_mask = self.data['rca_marketv_thisQ'] < 0

            # Calculate intermediate values
            trade_value = (
                self.data['nominal_converted'] * 
                self.data['trade_price'] / 100 * 
                self.data['fx_rate_thisQ']
            )
            
            maturity_diff = (
                self.data['nominal_converted'] * self.data['fx_rate_maturity'] -
                self.data['rca_ori_marketv_lastQ'] * self.data['fx_rate_maturity']
            )
            
            market_value_diff = (
                self.data['rca_ori_marketv_thisQ'] - 
                self.data['rca_ori_marketv_lastQ']
            ) * self.data['fx_rate_thisQ']

            # Define conditions and choices
            conditions = [
                customer_diff_mask,
                coa_special_mask & status_n_mask & marketv_positive_mask,
                coa_special_mask & status_n_mask & marketv_negative_mask,
                coa_special_mask & status_m_mask,
                coa_special_mask & status_o_mask
            ]

            choices = [
                0,  # Customer_diff = "YES"
                self.data['rca_marketv_thisQ'] - trade_value,  # Status="N", marketv>0
                self.data['rca_marketv_thisQ'] + trade_value,  # Status="N", marketv<0
                maturity_diff,  # Status="M"
                market_value_diff  # Status="O"
            ]

            # Calculate market price changes
            self.data['Market_price_changes'] = np.select(
                conditions,
                choices,
                default=0
            )

            return self

        except Exception as e:
            logger.error(f"Error calculating market price changes: {str(e)}")
            raise
    
    def calculate_exchange_rate_changes(self) -> 'RT30PostMapper':
        """
        Calculate Exchange Rate Changes using vectorized operations.
        
        Formula:
        IF(Status="D", 0,
        IF(Customer_diff="YES", 0,
            IF(OR(Form90_COA=6, Form90_COA=17),
                IF(AND(Status="N", rca_bookv_thisQ>0),
                    nominal_converted*trade_price/100*(fx_rate_thisQ-fx_rate_value),
                    IF(AND(Status="N", rca_bookv_thisQ<0),
                    -nominal_converted*trade_price/100*(fx_rate_thisQ-fx_rate_value),
                    IF(Status="M",
                        rca_ori_marketv_lastQ*(fx_rate_maturity-fx_rate_lastQ),
                        IF(Status="O",
                            rca_ori_marketv_lastQ*(fx_rate_thisQ-fx_rate_lastQ),
                            0)))),
                IF(Status="N",
                    nominal_converted*(fx_rate_thisQ-fx_rate_value),
                    IF(Status="M",
                    rca_ori_bookv_lastQ*(fx_rate_maturity-fx_rate_lastQ),
                    rca_ori_bookv_lastQ*(fx_rate_thisQ-fx_rate_lastQ)))))
        """
        try:
            # Create condition masks
            status_d_mask = self.data['Status'] == 'D'
            status_n_mask = self.data['Status'] == 'N'
            status_m_mask = self.data['Status'] == 'M'
            status_o_mask = self.data['Status'] == 'O'
            customer_diff_mask = self.data['Customer_diff'] == 'YES'
            coa_special_mask = self.data['Form90_COA'].isin(self.config.SPECIAL_COA)
            bookv_positive_mask = self.data['rca_bookv_thisQ'] > 0

            # Calculate rate differences
            fx_diff_this = self.data['fx_rate_thisQ'] - self.data['fx_rate_value']
            fx_diff_maturity = self.data['fx_rate_maturity'] - self.data['fx_rate_lastQ']
            fx_diff_last = self.data['fx_rate_thisQ'] - self.data['fx_rate_lastQ']

            # Calculate intermediate values
            trade_rate_diff = (
                self.data['nominal_converted'] * 
                self.data['trade_price'] / 100 * 
                fx_diff_this
            )
            
            normal_n_calc = self.data['nominal_converted'] * fx_diff_this
            normal_m_calc = self.data['rca_ori_bookv_lastQ'] * fx_diff_maturity
            normal_o_calc = self.data['rca_ori_bookv_lastQ'] * fx_diff_last

            # Special COA conditions
            special_coa_conditions = [
                status_n_mask & bookv_positive_mask,
                status_n_mask & ~bookv_positive_mask,
                status_m_mask,
                status_o_mask
            ]
            
            special_coa_choices = [
                trade_rate_diff,
                -trade_rate_diff,
                self.data['rca_ori_marketv_lastQ'] * fx_diff_maturity,
                self.data['rca_ori_marketv_lastQ'] * fx_diff_last
            ]

            # Calculate special COA results
            special_coa_result = np.select(
                special_coa_conditions,
                special_coa_choices,
                default=0
            )

            # Normal conditions
            normal_conditions = [
                status_n_mask,
                status_m_mask
            ]
            
            normal_choices = [
                normal_n_calc,
                normal_m_calc
            ]

            # Calculate normal results
            normal_result = np.select(
                normal_conditions,
                normal_choices,
                default=normal_o_calc
            )

            # Final conditions
            final_conditions = [
                status_d_mask,
                customer_diff_mask,
                coa_special_mask
            ]
            
            final_choices = [
                0,
                0,
                special_coa_result
            ]

            # Calculate final result
            self.data['Exchange_rate_changes'] = np.select(
                final_conditions,
                final_choices,
                default=normal_result
            )

            return self

        except Exception as e:
            logger.error(f"Error calculating exchange rate changes: {str(e)}")
            raise
    
    def calculate_rounding_diff(self) -> 'RT30PostMapper':
        """
        Calculate Rounding Differences using vectorized operations.
        
        Formula:
        IF(Customer_diff="YES", 0,
        IF(Form90_COA=0, 0,
            rca_bookv_lastQ + sum(Exchange_rate_changes + Market_price_changes + Transactions) - rca_bookv_thisQ))
        """
        try:
            # Create masks
            customer_diff_mask = self.data['Customer_diff'] == 'YES'
            coa_zero_mask = self.data['Form90_COA'] == '0'
            
            # Calculate sum of changes
            changes_sum = (
                self.data['Exchange_rate_changes'] + 
                self.data['Market_price_changes'] + 
                self.data['Transactions']
            )
            
            # Calculate total difference
            total_diff = (
                self.data['rca_bookv_lastQ'] + 
                changes_sum - 
                self.data['rca_bookv_thisQ']
            )
            
            # Apply conditions
            conditions = [
                customer_diff_mask,
                coa_zero_mask
            ]
            
            choices = [
                0,  # Customer_diff = "YES"
                0   # Form90_COA = 0
            ]
            
            self.data['Rounding_diff'] = np.select(
                conditions,
                choices,
                default=total_diff
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Error calculating rounding differences: {str(e)}")
            raise

    def process_all(self) -> pd.DataFrame:
        try:
            return (self
                    .apply_domicile_rules()
                    .apply_counterparty_types()
                    .apply_company_types()
                    .apply_customer_diff()
                    .apply_maturities()
                    .apply_nominal_conversion()
                    .apply_status()
                    .apply_form90_coa()
                    .calculate_transactions()
                    .calculate_market_price_changes()
                    .calculate_exchange_rate_changes()
                    .calculate_rounding_diff()
                    .get_data())
        except Exception as e:
            logger.error(f"Error in processing chain: {str(e)}")
            raise

    def get_data(self) -> pd.DataFrame:
        return self.data
