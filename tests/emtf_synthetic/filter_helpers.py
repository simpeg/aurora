from mt_metadata.timeseries.filters.coefficient_filter import CoefficientFilter

def make_coefficient_filter(gain=1.0, name="unit_conversion"):
    #in general, you need to add all required fields from the
    #standards.json
    coeff_filter = CoefficientFilter()
    cf = CoefficientFilter()
    cf.units_in = "digital counts"
    cf.units_out = "millivolts"
    cf.gain = gain
    cf.name = name
    return cf