FROM rocm/dev-ubuntu-24.04:6.2
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN pip3 install --break-system-packages --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

ADD . .

CMD [ "python3", "main.py" ]
