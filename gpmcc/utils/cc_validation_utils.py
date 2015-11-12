import numpy 

__cctypes_list = ['normal', 'normal_uc','binomial','multinomial',
				 'lognormal','poisson','vonmises','vonmises_uc']


def validate_metadata(metadata):
	# FIXME: fill in
	pass

def validate_cctypes(cctypes):
	for cctype in cctypes:
		if cctype not in __cctypes_list:
			raise ValueError("Invalid cctype %s. Valid values: %s" % (str(cctype), str(__cctypes_list)))

def validate_data(X):
	if not isinstance(X, list):
		raise TypeError("Data should be a list.")

	if not isinstance(X[0], numpy.ndarray):
		raise TypeError("All entries in data should by numpy arrays.")
	else:
		num_rows = X[0].shape[0]

	for x in X:
		if not isinstance(x, numpy.ndarray):
			raise TypeError("All entries in data should by numpy arrays.")

		if x.shape[0] != num_rows:
			raise ValueError("All columns should have the same number of rows.")