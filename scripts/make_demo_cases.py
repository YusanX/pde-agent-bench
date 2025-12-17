"""Generate 10 demo cases for PDEBench."""
import json
from pathlib import Path


def make_demo_cases():
    """Generate 10 demo cases covering Poisson and Heat equations."""
    cases = []
    
    # Case 1: Simple Poisson with constant kappa and manufactured solution
    cases.append({
        "id": "poisson_simple",
        "pde": {
            "type": "poisson",
            "description": "Simple Poisson with manufactured solution u=sin(pi*x)*sin(pi*y)",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.0}
            },
            "manufactured_solution": {
                "u": "sin(pi*x)*sin(pi*y)"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 16, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 5e-3,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50},
            "save_xdmf": False
        }
    })
    
    # Case 2: Poisson with higher degree
    cases.append({
        "id": "poisson_p2",
        "pde": {
            "type": "poisson",
            "description": "Poisson with P2 elements",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.0}
            },
            "manufactured_solution": {
                "u": "x**2 + y**2"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 12, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 2},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 1e-9,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 40, "ny": 40},
            "save_xdmf": False
        }
    })
    
    # Case 3: Poisson with quadrilateral mesh
    cases.append({
        "id": "poisson_quad",
        "pde": {
            "type": "poisson",
            "description": "Poisson on quadrilateral mesh",
            "coefficients": {
                "kappa": {"type": "constant", "value": 2.0}
            },
            "manufactured_solution": {
                "u": "exp(x)*cos(2*pi*y)"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 14, "cell_type": "quadrilateral"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 1e-2,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 45, "ny": 45},
            "save_xdmf": False
        }
    })
    
    # Case 4: Poisson with variable kappa (piecewise)
    cases.append({
        "id": "poisson_varied",
        "pde": {
            "type": "poisson",
            "description": "Poisson with constant kappa, smooth solution",
            "coefficients": {
                "kappa": {"type": "constant", "value": 0.5}
            },
            "manufactured_solution": {
                "u": "cos(pi*x)*cos(pi*y)"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 18, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 5e-3,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 55, "ny": 55},
            "save_xdmf": False
        }
    })
    
    # Case 5: Poisson with grid-based target
    cases.append({
        "id": "poisson_grid_target",
        "pde": {
            "type": "poisson",
            "description": "Poisson targeting grid-based metric",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.0}
            },
            "manufactured_solution": {
                "u": "sin(2*pi*x)*sin(2*pi*y)"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 20, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 1e-2,
            "metric": "rel_L2_grid"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 60, "ny": 60},
            "save_xdmf": False
        }
    })
    
    # Case 6: Heat equation simple
    cases.append({
        "id": "heat_simple",
        "pde": {
            "type": "heat",
            "description": "Heat equation with manufactured solution",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.0}
            },
            "manufactured_solution": {
                "u": "exp(-t)*sin(pi*x)*sin(pi*y)"
            },
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 12, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 5e-2,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "time.dt", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 40, "ny": 40},
            "save_xdmf": False
        }
    })
    
    # Case 7: Heat equation with longer time
    cases.append({
        "id": "heat_longer",
        "pde": {
            "type": "heat",
            "description": "Heat equation with longer time evolution",
            "coefficients": {
                "kappa": {"type": "constant", "value": 0.5}
            },
            "manufactured_solution": {
                "u": "exp(-2*t)*cos(pi*x)*cos(pi*y)"
            },
            "time": {
                "t0": 0.0,
                "t_end": 0.2,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 14, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 5e-2,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "time.dt", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 45, "ny": 45},
            "save_xdmf": False
        }
    })
    
    # Case 8: Heat with P2 elements
    cases.append({
        "id": "heat_p2",
        "pde": {
            "type": "heat",
            "description": "Heat equation with P2 elements",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.0}
            },
            "manufactured_solution": {
                "u": "exp(-t)*(x**2 + y**2)"
            },
            "time": {
                "t0": 0.0,
                "t_end": 0.05,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 10, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 2},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 3e-2,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "time.dt", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 35, "ny": 35},
            "save_xdmf": False
        }
    })
    
    # Case 9: Heat with quadrilateral mesh
    cases.append({
        "id": "heat_quad",
        "pde": {
            "type": "heat",
            "description": "Heat equation on quadrilateral mesh",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.5}
            },
            "manufactured_solution": {
                "u": "exp(-3*t)*sin(pi*x)*cos(pi*y)"
            },
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 12, "cell_type": "quadrilateral"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 5e-2,
            "metric": "rel_L2_fe"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "time.dt", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 40, "ny": 40},
            "save_xdmf": False
        }
    })
    
    # Case 10: Heat with grid-based target
    cases.append({
        "id": "heat_grid_target",
        "pde": {
            "type": "heat",
            "description": "Heat equation targeting grid-based metric",
            "coefficients": {
                "kappa": {"type": "constant", "value": 1.0}
            },
            "manufactured_solution": {
                "u": "exp(-t)*cos(2*pi*x)*cos(2*pi*y)"
            },
            "time": {
                "t0": 0.0,
                "t_end": 0.15,
                "dt": 0.03,
                "scheme": "backward_euler"
            }
        },
        "domain": {"type": "unit_square"},
        "mesh": {"resolution": 16, "cell_type": "triangle"},
        "fem": {"family": "Lagrange", "degree": 1},
        "bc": {
            "dirichlet": {"on": "all", "value": "u"}
        },
        "targets": {
            "target_error": 1e-2,
            "metric": "rel_L2_grid"
        },
        "expose_parameters": ["mesh.resolution", "fem.degree", "time.dt", "ksp.type", "ksp.rtol", "pc.type"],
        "output": {
            "format": "npz",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50},
            "save_xdmf": False
        }
    })
    
    return cases


def main():
    """Generate and save all demo cases."""
    # scripts/ is at project root, cases/ is at project root
    cases_dir = Path(__file__).parent.parent / 'cases' / 'demo'
    cases_dir.mkdir(parents=True, exist_ok=True)
    
    cases = make_demo_cases()
    
    for case in cases:
        case_file = cases_dir / f"{case['id']}.json"
        with open(case_file, 'w') as f:
            json.dump(case, f, indent=2)
        print(f"Generated: {case_file}")
    
    print(f"\nTotal: {len(cases)} demo cases generated in {cases_dir}")


if __name__ == '__main__':
    main()

