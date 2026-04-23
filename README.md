# AME-EEE-598-Taichi-MPM

## Dependencies

```bash
pip install -r requirements.txt
```

## Snowfall Particles

```bash
python src/snowfall_particles/main.py --config configs/default.yml
```

### Config

- 默认配置在 [`configs/default.yml`](configs/default.yml)
- 你可以在里面改 `n_grid/steps/dt/n_particles`、重力、渲染参数、obstacle 列表等

### Mesh SDF cache

- **缓存文件位置**：与 mesh 文件同目录（紧挨 mesh）
- **命名规则**：`<mesh_filename>.sdf_res<res>.<key>.npz`
  - 例：`assets/sphere.obj.sdf_res128.0123abcd4567ef89.npz`
- **key 组成**：mesh 内容 SHA256 + (sdf_res, scale, center) 共同派生（内容或参数变了就会生成新文件）
- **清理缓存**：直接删除对应的 `.npz` 文件即可

## Walking Tree Controller

## Skeleton to Tree Mesh

## TODO
- [] environment.yml