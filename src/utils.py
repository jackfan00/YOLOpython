def find_float_arg(argv, option, defv): 
	for s in argv:
		if s == '-thresh' :
			try:
				return float(defv)
			except:
				print 'find_float_arg() Error::-thresh need to be float'
				exit()
	return defv

def find_int_arg(argv, option, defv):
	for s in argv:
		if s == '-c':
			try:
				return int(defv)
			except:
				print 'find_float_arg() Error::-c need to be integer'
				exit()
def basecfg(cfg_path):
	ss = cfg_path.split('/')
	s = ss[len(ss)-1].split('.')[0]
	return s
