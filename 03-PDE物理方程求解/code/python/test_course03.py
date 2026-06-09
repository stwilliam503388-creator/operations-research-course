"""Unit tests for course 03 PDE examples.

These tests focus on deterministic setup routines and basic physical
properties, without turning the simulations into heavy numerical workloads.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import case03_heat
import case04_wave
from capstone import ChipPackage


class HeatEquationTests(unittest.TestCase):
    def test_set_parameters_contains_expected_keys(self) -> None:
        params = case03_heat.set_parameters()
        self.assertTrue({"L", "alpha", "T_left", "T_initial", "nx", "nt", "nt_implicit"} <= set(params))

    def test_build_mesh_satisfies_stability_condition(self) -> None:
        params = case03_heat.set_parameters()
        mesh = case03_heat.build_mesh(params)
        r = params["alpha"] * mesh["dt"] / mesh["dx"] ** 2
        self.assertGreater(mesh["dx"], 0.0)
        self.assertGreater(mesh["dt"], 0.0)
        self.assertLessEqual(r, 0.5)

    def test_initial_condition_is_uniform_ambient_temperature(self) -> None:
        params = case03_heat.set_parameters()
        mesh = case03_heat.build_mesh(params)
        u0 = case03_heat.initial_condition(mesh["x"], params)
        np.testing.assert_allclose(u0, params["T_initial"])

    def test_explicit_solution_preserves_boundary_and_hot_to_cold_profile(self) -> None:
        params = case03_heat.set_parameters()
        mesh = case03_heat.build_mesh(params)
        with contextlib.redirect_stdout(io.StringIO()):
            result = case03_heat.solve_explicit(params, mesh)
        u_final = result["u_final"]
        self.assertAlmostEqual(u_final[0], params["T_left"], places=10)
        self.assertAlmostEqual(u_final[-1], u_final[-2], places=10)
        self.assertTrue(np.all(np.diff(u_final) <= 1e-8))
        self.assertGreaterEqual(u_final[0], u_final[-1])


class WaveEquationTests(unittest.TestCase):
    def test_set_parameters_contains_expected_keys(self) -> None:
        params = case04_wave.set_parameters()
        self.assertTrue({"L", "c", "source_x0", "source_sigma", "source_amplitude", "nx", "nt", "boundary_type"} <= set(params))

    def test_build_mesh_satisfies_courant_limit(self) -> None:
        params = case04_wave.set_parameters()
        mesh = case04_wave.build_mesh(params)
        self.assertGreater(mesh["dx"], 0.0)
        self.assertGreater(mesh["dt"], 0.0)
        self.assertLessEqual(mesh["courant"], 1.0)
        self.assertTrue(case04_wave.check_cfl(mesh))

    def test_initial_condition_is_gaussian_centered_near_source(self) -> None:
        params = case04_wave.set_parameters()
        mesh = case04_wave.build_mesh(params)
        u0 = case04_wave.initial_condition(mesh["x"], params)
        peak_idx = int(np.argmax(u0))
        self.assertAlmostEqual(mesh["x"][peak_idx], params["source_x0"], delta=mesh["dx"])
        self.assertGreater(u0[peak_idx], 0.0)
        self.assertLess(u0[0], u0[peak_idx])
        self.assertLess(u0[-1], u0[peak_idx])
        self.assertAlmostEqual(u0[0], 0.0, places=8)
        self.assertAlmostEqual(u0[-1], 0.0, places=8)


class CapstoneTests(unittest.TestCase):
    def test_chip_package_exposes_expected_material_and_geometry_attributes(self) -> None:
        chip = ChipPackage()
        for attr in [
            "Lx", "Ly", "Lz", "k_si", "rho_si", "cp_si", "E_si", "nu_si",
            "alpha_si", "k_fr4", "rho_fr4", "cp_fr4", "E_fr4", "nu_fr4",
            "alpha_fr4", "Q_hotspot", "T_ambient", "h_conv", "Nx", "Ny", "dx", "dy",
        ]:
            self.assertTrue(hasattr(chip, attr), msg=f"missing attribute {attr}")
        self.assertGreater(chip.Lx, 0.0)
        self.assertGreater(chip.Ly, 0.0)
        self.assertGreater(chip.k_si, chip.k_fr4)
        self.assertAlmostEqual(chip.dx, chip.Lx / chip.Nx, places=12)
        self.assertAlmostEqual(chip.dy, chip.Ly / chip.Ny, places=12)


if __name__ == "__main__":
    unittest.main()
