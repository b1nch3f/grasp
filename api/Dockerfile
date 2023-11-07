FROM python:3.11

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/api

# Install requirements, including from Gemfury
ADD ./ /opt/api/
RUN pip install --upgrade pip
RUN pip install -r /opt/api/requirements.txt

RUN chmod +x /opt/api/run.sh
RUN chown -R api-user:api-user ./

USER api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]
