import phoebe


b = phoebe.default_star(force_build=True)
b.save('./default_star.bundle', compact=True)

b = phoebe.default_binary(force_build=True)
b.save('./default_binary.bundle', compact=True)

b = phoebe.default_binary(contact_binary=True, force_build=True)
b.save('./default_contact_binary.bundle', compact=True)
