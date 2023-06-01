import logging

from protex.system import ProtexSystem, ProtexTemplates
from protex.testsystems import generate_h2o_system
from protex.update import NaiveMCUpdate, StateUpdate

h2o_updates = {
    frozenset(["OH", "OH"]): {"r_max": 0.165, "prob": 1},
    frozenset(["SWM4", "OH"]): {"r_max": 1.165, "prob": 1},
    frozenset(["SWM4", "SWM4"]): {"r_max": 0.165, "prob": 1},
    frozenset(["SWM4", "H3O"]): {"r_max": 0.165, "prob": 1},
    frozenset(["H3O", "H3O"]): {"r_max": 0.165, "prob": 1},
    frozenset(["H3O", "OH"]): {"r_max": 0.165, "prob": 1},
}

# TODO
h2o_states = {
    "OH": {
        "atom_name": "OH2",
    },
    "SWM4": {
        "atom_name": "H1",
        "equivalent_atom": "H2",
    },
}


def test_protexsystem():
    simulation = generate_h2o_system(use_plugin=False)
    #for force in simulation.system.getForces():
    #    print(force)

    templates = ProtexTemplates(states=[h2o_states], allowed_updates=h2o_updates)

    system = ProtexSystem(simulation, templates)

    print(system)

def test_customnonbondedforce_update():
    h2o_updates = {frozenset(["SWM4", "OH"]): {"r_max": 1.165, "prob": 1}}
    h2o_states = {
    "OH": {
        "atom_name": "OH2",
    },
    "SWM4": {
        "atom_name": "H1",
        "equivalent_atom": "H2",
    },
}
    simulation = generate_h2o_system(use_plugin=False)
    templates = ProtexTemplates(states=[h2o_states], allowed_updates=h2o_updates)
    system = ProtexSystem(simulation, templates)

    oh = system.residues[2]
    h2o = system.residues[0]
    h2o_params = h2o._get_CustomNonbondedForce_parameters_at_lambda(0)
    h2o_params_l1 = h2o._get_CustomNonbondedForce_parameters_at_lambda(1)
    oh_params = oh._get_CustomNonbondedForce_parameters_at_lambda(0)
    oh_params_l1 = oh._get_CustomNonbondedForce_parameters_at_lambda(1)
    assert h2o_params_l1[0] == oh_params[0]
    assert h2o_params[0] == oh_params_l1[0]
    h2o._set_CustomNonbondedForce_parameters(h2o_params_l1)
    oh._set_CustomNonbondedForce_parameters(oh_params_l1)
    new_h2o = h2o._get_CustomNonbondedForce_parameters_at_lambda(0)
    new_oh = oh._get_CustomNonbondedForce_parameters_at_lambda(0)
    assert new_h2o[0] == oh_params_l1[0]
    assert new_oh[0] == h2o_params_l1[0]

    # would need plugin for next lines
    #update = NaiveMCUpdate(system, all_forces=True)
    #state_update = StateUpdate(update)
    #state_update.update(2)
