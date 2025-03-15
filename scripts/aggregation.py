import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSummarizer_D_B:
    """Efficient RT30 Part D and B data summarization"""
    
    RESIDUAL_MATURITIES = [
        "Term<= 90 days",
        "Term> 90 days <= 6 mths",
        "Term> 6 mths <= 1 year",
        "Term>1 year <= 5 years",
        "Term> 5 years"
    ]
    
    def __init__(self, mapped_rt30_df: pd.DataFrame, currency_code_df: pd.DataFrame,
                 maturity_type: str, ISIN: str = None):
        """Initialize summarizer with configuration"""
        self.data = mapped_rt30_df
        self.currency_code = currency_code_df.iloc[:, :2]
        self.maturity_type = maturity_type
        self.ISIN = ISIN
        self._create_base_masks()
    
    def _create_base_masks(self) -> None:
        """Cache commonly used masks"""
        self.base_mask = (
            (self.data['Form90_COA'] == '6') &
            (self.data['rv_resident_thisQ'] == 'NO')
        )
        
        self.short_term_mask = self.base_mask & (self.data['Original_maturity'] == 'Short-term')
        self.long_term_mask = self.base_mask & (self.data['Original_maturity'] == 'Long-term')
        self.null_isin_mask = self.data['ISIN'].isna()
    
    def _aggregate_by_currency_residual(self, mask: pd.Series, 
                                      value_cols: Dict[str, str]) -> pd.DataFrame:
        """Aggregate data by currency and residual maturity"""
        # Check currency column name
        currency_col = 'Currency_Code' if 'Currency_Code' in self.data.columns else 'currency'
        
        result = (
            self.data[mask]
            .groupby([currency_col, 'Residual_maturity'])[list(value_cols.values())]
            .sum()
            .div(1000)
            .reset_index()
        )
        
        # Pivot residual maturity
        pivoted = pd.pivot_table(
            result,
            index=currency_col,
            columns='Residual_maturity',
            values=list(value_cols.values()),
            fill_value=0
        )
        
        # Flatten column names
        pivoted.columns = [
            f"{val}_{mat.replace(' ', '_')}"
            for val in value_cols.keys()
            for mat in pivoted.columns.levels[1]
        ]
        
        return (
            pivoted
            .reset_index()
            .rename(columns={currency_col: 'Currency_Code'})
            .pipe(self._merge_currency_codes)
            .pipe(self._remove_zeros)
        )
    
    def _merge_currency_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge with currency codes"""
        return (
            self.currency_code
            .merge(
                df, 
                how='left',
                left_on='Currency Code',  # Match ISO_currency_code_df column name
                right_on='Currency_Code'
            )
            .fillna(0)
        )
    
    def _remove_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with all zero values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[df[numeric_cols].abs().sum(axis=1) > 0]

    # @classmethod
    # def create_variants(self, mapped_rt30_df: pd.DataFrame, 
    #                    currency_code_df: pd.DataFrame) -> Dict[str, 'DataSummarizer_D_B']:
    #     """Create all variant summarizers"""
    #     variants = {
    #         '6b_short': ('Short-term', None),
    #         '7b_long': ('Long-term', None)
    #     }
        
    #     return {
    #         key: self(mapped_rt30_df, currency_code_df, maturity, isin)
    #         for key, (maturity, isin) in variants.items()
    #     }

    def calculate_6b_values(self) -> pd.DataFrame:
        """Calculate original book value (6B) by residual maturity"""
        try:
            mask = self.short_term_mask 
            
            result = self._aggregate_by_currency_residual(
                mask=mask,
                value_cols={'market_value': 'rca_marketv_thisQ'}
            )
            
            # Add metadata
            result['Maturity_Type'] = self.maturity_type
            result['Calculation_Type'] = '6B'
            
            # Add total across residual maturities
            value_cols = [col for col in result.columns if col.startswith('market_value')]
            result['Total'] = result[value_cols].sum(axis=1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market value calculation: {str(e)}")
            raise

    def calculate_7b_values(self) -> pd.DataFrame:
        """Calculate 7B values by residual maturity"""
        try:
            components = {
                'market_value': 'rca_marketv_thisQ',
                'book_value': 'rca_bookv_thisQ'
            }
            
            result = self._aggregate_by_currency_residual(
                mask=self.long_term_mask & self.null_isin_mask,
                value_cols=components
            )
            
            # Add metadata
            result['Maturity_Type'] = self.maturity_type
            result['Calculation_Type'] = '7B'
            
            # Calculate totals for each residual maturity
            for mat in self.RESIDUAL_MATURITIES:
                mat_suffix = f"_{mat.replace(' ', '_')}"
                cols = [col for col in result.columns if col.endswith(mat_suffix)]
                result[f'7B_Total{mat_suffix}'] = result[cols].sum(axis=1)
            
            # Add grand total
            total_cols = [col for col in result.columns if col.startswith('7B_Total_')]
            result['Grand_Total'] = result[total_cols].sum(axis=1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in 7B calculation: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error in 7B calculation: {str(e)}")
            raise

    def generate_summary(self) -> Dict[str, pd.DataFrame]:
        """Generate all summaries"""
        return {
            '6B': self.calculate_6b_values(),
            '7B': self.calculate_7b_values()
        }



class DataSummarizer_D_A:
    """Efficient RT30 data summarization using vectorized operations"""
    
    def __init__(self, mapped_rt30_df: pd.DataFrame, iso_codes_df: pd.DataFrame,
                 maturity_type: str, counterparty_type: str):
        """Initialize summarizer with specific configuration"""
        self.data = mapped_rt30_df
        self.iso_codes = iso_codes_df.iloc[:, :2]
        self.maturity_type = maturity_type
        self.counterparty_type = counterparty_type
        self._create_base_masks()
    
    @classmethod
    def create_variants_prt_d_a(cls, mapped_rt30_df: pd.DataFrame, 
                       iso_codes_df: pd.DataFrame) :
        """Create summarizers for all variants"""
        variants = {
            '6a_short_other': ('Short-term', 'Other non-resident counterparties'),
            '6a_short_direct': ('Short-term', 'Direct investment groups abroad'),
            '7a_long_other': ('Long-term', 'Other non-resident counterparties'),
            '7a_long_direct': ('Long-term', 'Direct investment groups abroad')

        }
        
        return {
            key: cls(mapped_rt30_df, iso_codes_df, maturity, counterparty)
            for key, (maturity, counterparty) in variants.items()
        }

    
    def _create_base_masks(self) -> None:
        """Create commonly used masks for filtering"""
        self.base_mask = (
            (self.data['Form90_COA'] == '6') &
            (self.data['Original_maturity'] == self.maturity_type)
        )
        
        self.thisq_mask = self.base_mask & (
            self.data['Intercompany_type_thisQ'] == self.counterparty_type
        )
        
        self.lastq_mask = self.base_mask & (
            self.data['Intercompany_type_lastQ'] == self.counterparty_type
        )
        
        self.au_mask = self.data['ISIN'].str.startswith('AU', na=False)
        self.na_mask = self.data['ISIN'].isna()

    def _aggregate_by_mask(self, mask: pd.Series, value_cols: Dict[str, str],
                          period: str = 'thisQ') -> pd.DataFrame:
        """Aggregate data based on mask and value columns"""
        grouped = (
            self.data[mask]
            .groupby(f'Domicile_{period}')
            [list(value_cols.values())]
            .sum()
            .div(1000)
        )
        
        grouped.columns = list(value_cols.keys())
        return grouped

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate complete summary table with all metrics"""
        try:
            aggregations = {
                'Opening_Position': {
                    'mask': self.lastq_mask & (self.au_mask | self.na_mask),
                    'cols': {'Opening_Position': 'rca_bookv_lastQ'},
                    'period': 'lastQ'
                },
                'Closing_Position': {
                    'mask': self.thisq_mask & (self.au_mask | self.na_mask),
                    'cols': {'Closing_Position': 'rca_bookv_thisQ'},
                    'period': 'thisQ'
                },
                'Liability_Changes': {
                    'mask': self.thisq_mask & (self.au_mask | self.na_mask),
                    'cols': {
                        'Liability_Increases': 'Transactions',
                        'Liability_Decreases': 'rca_bookv_lastQ'
                    },
                    'period': 'thisQ'
                },
                'Market_Exchange': {
                    'mask': self.thisq_mask & (self.au_mask | self.na_mask),
                    'cols': {
                        'Market_Price_Changes': 'Market_price_changes',
                        'Exchange_Rate_Variations': 'Exchange_rate_changes'
                    },
                    'period': 'thisQ'
                },
                'Interest': {
                    'mask': self.thisq_mask & (self.au_mask | self.na_mask),
                    'cols': {'Interest_Payable': 'rca_accrint_thisQ'},
                    'period': 'thisQ'
                }
            }
            
            # Calculate all metrics
            results = []
            for agg_name, params in aggregations.items():
                df = self._aggregate_by_mask(
                    params['mask'],
                    params['cols'],
                    params['period']
                )
                results.append(df)
            
            # Combine results and add metadata
            summary = (
                pd.concat(results, axis=1)
                .fillna(0)
                .reset_index()
                .rename(columns={'index': 'ISO_Code'})
            )
            
            final_summary = (
                self.iso_codes
                .merge(summary, how='left', left_on='ISO Code', right_on='ISO_Code')
                .fillna(0)
            )
            
            # Add metadata columns
            final_summary['Maturity_Type'] = self.maturity_type
            final_summary['Counterparty_Type'] = self.counterparty_type
            
            # Add reconciliation
            final_summary['Reconciliation'] = (
                final_summary['Opening_Position'] +
                final_summary['Liability_Increases'] +
                final_summary['Liability_Decreases'] +
                final_summary['Market_Price_Changes'] +
                final_summary['Exchange_Rate_Variations'] -
                final_summary['Closing_Position']
            )
            
            # Remove zero rows
            numeric_cols = final_summary.select_dtypes(include=[np.number]).columns
            final_summary = final_summary[final_summary[numeric_cols].abs().sum(axis=1) > 0]
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Error generating summary table: {str(e)}")
            raise

    @staticmethod
    def save_summaries(summaries: Dict[str, pd.DataFrame], output_path: Path) -> None:
        """Save all summaries to Excel"""
        with pd.ExcelWriter(output_path) as writer:
            for variant, df in summaries.items():
                df.to_excel(writer, sheet_name=variant, index=False)