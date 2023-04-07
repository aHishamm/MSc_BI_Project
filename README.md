# MSc_BI_Project
This is my project for BI 
#### Environment Setup with Conda 
```bash
conda create -n biproject python=3.12
conda activate biproject 
pip install -r requirements.txt
```
### Deployment as a Docker Container 
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
