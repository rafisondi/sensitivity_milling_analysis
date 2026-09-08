# Meshes (not required)

The robot model is built with `pinocchio.buildModelFromUrdf`, which parses only
the kinematic and inertial parameters from the URDF. The visual `.stl` meshes
referenced by `../Staeubli-Huynh 1.urdf` (`meshes/Base.stl`, `meshes/J1.stl`, …)
are **not loaded** by this simulation and are therefore not shipped in the repo.

The 3D visualisation in `dexel_sim/main_rotated.py` draws the robot as a link
polyline with matplotlib and does not need the meshes either.

If you want to load the geometry (e.g. with `buildGeomFromUrdf` or a Meshcat
viewer), drop the `.stl` files into this folder — the URDF already references
them with relative paths.
