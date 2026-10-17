# promptmap for dbt users

| promptmap concept | dbt analogy |
|---|---|
| `.prompt` file | `.sql` model file |
| `ref('model')` | `{{ ref('model') }}` |
| `ref('model.key[0]')` | `ref()` plus a JSON path, like `col -> 'key' -> 0` |
| `config(each='model.key[*]')` | `json_each` / `jsonb_array_elements`: one row per item |
| `models/` directory | `models/` directory |
| `global.prompt` | `query-comment` in `dbt_project.yml` |
| SQLite `runs` table | dbt `run_results.json` |
| SQLite `model_results` table | dbt `model` timing artifacts |
