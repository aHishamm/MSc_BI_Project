# MSc_BI_Project
This is my project for BI 
#### Environment Setup with Conda 
```bash
conda create -n biproject python=3.12
conda activate biproject 
pip install -r requirements.txt
```
#### Deployment as a Docker Container 
- To create a Docker container, a Dockerfile is provided. Make sure Docker Desktop is installed. The Dockerfile contains the following
```bash
FROM python:3.10-slim
ADD . .
RUN pip install -r requirements.txt 
EXPOSE 7250
ENTRYPOINT ["streamlit","run"] 
CMD ["./dashboard.py","--server.headless","true","--server.fileWatcherType","none","--browser.gatherUsageStats","false","--server.port=7250","--server.address=0.0.0.0"]
```
- To build the Docker image from the Dockerfile, run the following command in the terminal or powershell: 
```bash
docker build -t biprojectimage . 
```

- To run a Docker container with the name 'biprojectcontainer' on port 7250 
```bash
docker run -p 7250:7250 --name biprojectcontainer biprojectimage
```
#### HuggingFace Deployment
The project is hosted on HuggingFace and is available via the following link: https://huggingface.co/spaces/ahishamm/biproject
#### Project Previews
###### Preview 1 
![Screenshot from 2023-04-07 19-12-13](https://user-images.githubusercontent.com/40188935/230633005-e58f5b64-c0cc-4c3b-83f3-d4126c186fa6.png)
###### Preview 2
![Screenshot from 2023-04-07 19-12-28](https://user-images.githubusercontent.com/40188935/230633019-3bd8145c-2388-4627-b16a-d39ded51fc62.png)
###### Preview 3
![Screenshot from 2023-04-07 19-13-00](https://user-images.githubusercontent.com/40188935/230633038-d32ba73e-7fdf-4178-a386-0b8114578554.png)
###### Preview 4
![Screenshot from 2023-04-07 19-13-07](https://user-images.githubusercontent.com/40188935/230633051-77979088-30d3-45dc-a8d1-7d5cad35457a.png)
###### Preview 5
![Screenshot from 2023-04-07 19-13-18](https://user-images.githubusercontent.com/40188935/230633152-acae3487-07cf-449a-8d61-98f858aeb7c2.png)
