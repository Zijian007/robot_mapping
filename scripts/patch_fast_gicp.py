"""Apply the small, version-checked Python binding fix in the private source tree."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "src/python/main.cpp"
source = path.read_text()
if 'm.attr("mapping_gil_released")' not in source:
    source = source.replace('PYBIND11_MODULE(pygicp, m) {',
        'PYBIND11_MODULE(pygicp, m) {\n  m.attr("mapping_gil_released") = true;')
    old = '.def("get_fitness_score", [] (LsqRegistration& reg, const double max_range) { return reg.getFitnessScore(max_range); })'
    new = '''.def("has_converged", [](LsqRegistration& reg) { return reg.hasConverged(); })
    .def("get_fitness_score", [] (LsqRegistration& reg, const double max_range) {
      py::gil_scoped_release release;
      return reg.getFitnessScore(max_range);
    })'''
    if old not in source or '        reg.align(aligned, initial_guess);' not in source:
        raise SystemExit("Unsupported fast_gicp binding revision; patch was not applied")
    source = source.replace(old, new).replace('        reg.align(aligned, initial_guess);',
        '        { py::gil_scoped_release release; reg.align(aligned, initial_guess); }')
    path.write_text(source)
if '.def("has_converged"' not in source or 'py::gil_scoped_release' not in source:
    raise SystemExit("Incomplete fast_gicp binding patch")
print(f"[third_party] convergence/GIL binding patch ready: {path}")
