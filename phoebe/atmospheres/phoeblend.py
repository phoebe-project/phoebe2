import phoebe

# pb = phoebe.atmospheres.passbands.Passband.load('kepler.phoenix.pb')

pb = phoebe.get_passband('Kepler:mean')
pb.compute_phoenix_response('tables/phoenix', verbose=True)
pb.compute_phoenix_intensities('tables/phoenix', verbose=True)

pb.save('kepler.blended.pb')

#pb.compute_blended_response()
#pb.version = 1.0
#pb.pbname = 'phoenix'

#pb.save('kepler.blended.pb')
