import logging

from protex.system import ProtexSystem, ProtexTemplates
from protex.testsystems import generate_h2o_system

h2o_updates = {
    frozenset(["OH", "OH"]): {"r_max": 0.165, "prob": 1},
    frozenset(["SWM4", "OH"]): {"r_max": 0.165, "prob": 1},
    frozenset(["SWM4", "SWM4"]): {"r_max": 0.165, "prob": 1},
    frozenset(["SWM4", "H3O"]): {"r_max": 0.165, "prob": 1},
    frozenset(["H3O", "H3O"]): {"r_max": 0.165, "prob": 1},
    frozenset(["H3O", "OH"]): {"r_max": 0.165, "prob": 1},
}

# TODO
h2o_states = {
    "OH": {
        "atom_name": "O1",
    },
    "SWM4": {
        "atom_name": "HO2",
        "equivalent_atom": "HO1",
    },
}


def test_protexsystem():
    simulation = generate_h2o_system()

    templates = ProtexTemplates(states=[h2o_states], allowed_updates=h2o_updates)

    system = ProtexSystem(simulation, templates)

    print(system)
