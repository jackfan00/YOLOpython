import parse
import os

f = open('workingcfg.txt')
hascfg = False
for l in f:
	cfgpath = l.strip()
	if os.path.isfile(cfgpath):
		hascfg = True
		break

if hascfg:
	net = parse.parse_network_cfg(cfgpath)
else:
	print 'Error::workingcfg.txt dont contain valid cfg file'
	exit()
