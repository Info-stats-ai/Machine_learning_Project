import sys
# creating a custom exception class
# this class is used to raise an exception with a detailed error message
# it is used to handle errors in the code
# it is used to display the error message in a readable format
# it is used to log the error message
# it is used to send the error message to the user
# it is used to send the error message to the developer
# it is used to send the error message to the administrator
# it is used to send the error message to the system
# it is used to send the error message to the network
# it is used to send the error message to the database
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    # this exc_tb have the information about the error
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)