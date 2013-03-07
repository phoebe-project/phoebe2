import phoebe

logger = phoebe.get_basic_logger()

output = phoebe.parse_phot('example0.phot')
print output[0][-1]
print output[1][-1]
print [i['ref'] for i in output[0]]
print [i['ref'] for i in output[1]]
print '==================='

output = phoebe.parse_phot('example1.phot')
print output[0][-1]
print output[1][-1]
print '==================='

output = phoebe.parse_phot('example2.phot')
print output[0][-1]
print output[1][-1]
print '==================='

output = phoebe.parse_phot('example3.phot')
print output['starA'][0][0]
print output['starA'][1][0]
print output['starB'][0][0]
print output['starB'][1][0]
print '==================='

output = phoebe.parse_phot('example4.phot',columns=['flux','passband','sigma','unit'])
print output[0][-1]
print output[1][-1]
print '==================='

output = phoebe.parse_phot('example4.phot',columns=['flux','passband','sigma'])
print output[0][-1]
print output[1][-1]
print '==================='
