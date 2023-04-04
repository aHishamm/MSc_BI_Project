FROM python:3.10-slim
ADD . .
RUN pip install -r requirements.txt 
EXPOSE 7250
ENTRYPOINT ["streamlit","run"] 
CMD ["./dashboard.py","--server.headless","true","--server.fileWatcherType","none","--browser.gatherUsageStats","false","--server.port=7250","--server.address=0.0.0.0"]
