from clearml import Logger
import pandas as pd


def report_text(message: str):
    Logger.current_logger().report_text(message)

def report_scalar(title: str, series: str, iteration: int, value: float):
    Logger.current_logger().report_scalar(title, series, iteration=iteration, value=value)
    
def report_single_value(name: str, value: float):
    Logger.current_logger().report_single_value(name=name, value = value)

def report_table(df: pd.DataFrame):
    # type: (Logger, int) -> ()
    """
    reporting tables to the plots section
    :param logger: The task.logger to use for sending the plots
    :param iteration: The iteration number of the current reports
    """
    # report tables

    # Report table - DataFrame with index
    Logger.current_logger().report_table("Metrics", "Some metrics values", iteration=0, table_plot=df)