import sys
import logging

def error_message_detail(error, error_detail:sys):
    """
    This function takes an error and its details and returns a formatted string
    with the error message and its details.
    """
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        """
        This function initializes the CustomException class with an error message
        and its details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        """
        This function returns the error message when the exception is raised.
        """
        return self.error_message