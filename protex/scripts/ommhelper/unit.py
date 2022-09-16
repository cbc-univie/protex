try:
    from openmm import unit
    from openmm.unit import (angstrom, bar, dalton, elementary_charge, kelvin,
                             kilocalorie_per_mole, kilojoule_per_mole, meter,
                             nanometer, picosecond, volt)

except ImportError:
    from simtk import unit
    from simtk.unit import (angstrom, bar, dalton, elementary_charge, kelvin,
                            kilocalorie_per_mole, kilojoule_per_mole, meter,
                            nanometer, picosecond, volt)

ps = picosecond
nm = nanometer
kJ_mol = kilojoule_per_mole
kcal_mol = kilocalorie_per_mole
qe = elementary_charge
