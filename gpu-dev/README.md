# LightAutoML-GPU

**environment.yml with RAPIDS-0.19 under cuda-11.0**

```bash
source ../lama_venv/bin/activate

conda env update -p ../lama_venv --file ./environment.yml 
```

**possible issues:**

- if it complains about cupy, uninstall the installed versions and install using pip (cupy-cuda110).
