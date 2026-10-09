```bash
docker build -t tinyemb:v1.0.0 .
docker stop tinyemb && docker rm tinyemb
docker run -it --name tinyemb -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  tinyemb:v1.0.0
```