import phoebe

logger = phoebe.get_basic_logger()

#output = phoebe.parse_rv('example0.rv')
#print output['componentA']
#print output['componentA'][0][-1]
#print output['componentA'][1][-1]
#print '==================='

#output = phoebe.parse_rv('example1.rv')
#print output[0][-1]
#print output[1][-1]
#print '==================='

output = phoebe.parse_rv('example3.rv',components=['A',None,'A','B','B'])
print output['A'][0][-1]
print output['B'][0][-1]
print '==================='
