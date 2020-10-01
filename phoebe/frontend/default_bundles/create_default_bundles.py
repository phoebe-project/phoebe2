import phoebe


print("creating default_star.bundle")
b = phoebe.default_star(force_build=True)
b.save('./default_star.bundle', compact=True, incl_uniqueid=False)

print("creating default_binary.bundle")
b = phoebe.default_binary(force_build=True)
b.save('./default_binary.bundle', compact=True, incl_uniqueid=False)

print("creating default_contact_binary.bundle")
b = phoebe.default_binary(contact_binary=True, force_build=True)
b.save('./default_contact_binary.bundle', compact=True, incl_uniqueid=False)
