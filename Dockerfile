FROM animcogn/face_recognition:cpu-latest
LABEL authors="chiyizi"

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -i http://pypi.douban.com/simple/ -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python3", "./main.py"]