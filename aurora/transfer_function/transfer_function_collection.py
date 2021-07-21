"""
Placeholder class.  Will probably evolve structure in time.
This is a container to hold two things:
1. TransferFunctionHeader
2. Dictionary of TransferFunction Objects
3.
"""


class TransferFunctionCollection(object):
    def __init__(self, **kwargs):
        self.header = kwargs.get("header", None)
        self.tf_dict = kwargs.get("tf_dict", None)

    def write_emtf_z_file(self, z_file_path):
        """

        Returns
        -------

        """
        f = open(z_file_path, "w")
        f.writelines(" **** IMPEDANCE IN MEASUREMENT COORDINATES ****\n")
        f.writelines(" ********** WITH FULL ERROR COVARINCE **********\n")

        f.close()