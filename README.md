# AME-EEE-598-Taichi-MPM

## Dependencies

```bash
pip install -r requirements.txt
```

## Snowfall particles

```bash
python src/snowfall_particles/snowfall_simulate.py --config configs/default.yml
python src/snowfall_particles/snowfall_simulate.py --offline --output outputs/snow.npz
python src/snowfall_particles/visualize_output.py outputs/snow.npz
```

### Configuration

- Default settings are in [`configs/default.yml`](configs/default.yml)
- You can edit `n_grid` / `steps` / `dt` / `n_particles`, gravity, rendering options, obstacle lists, and more there

### Grid SDF cache

- **Location**: Same directory as the mesh file (alongside the mesh)
- **Naming**: `<mesh_filename>.sdf_res<res>.<key>.npz`
  - Example: `assets/sphere.obj.sdf_res128.0123abcd4567ef89.npz`
- **Key**: Derived from mesh content SHA256 together with `(sdf_res, scale, center)`; a new file is created when content or parameters change
- **Clearing cache**: Delete the corresponding `.npz` files

## Walking tree controller

## Skeleton to tree-shaped mesh

## TODO

- [ ] environment.yml
