cd ..
docker run -it -p -d 8080:8080 -v "${PWD}/conf:/opt/darknet_server/conf" --runtime=nvidia --name darknet_python_server  darknet_python_server
